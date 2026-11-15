import math
import os
import pickle
import socket
import tempfile
import time
import uuid

from sklearn.metrics import\
    roc_auc_score as auc_score,\
    f1_score, average_precision_score as ap_score
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.multiprocessing as mp
from torch.optim import Adam, Adadelta

from loaders.tdata import TData
from models.euler_detector import DetectorEncoder, DetectorRecurrent
from models.euler_predictor import PredictorEncoder, PredictorRecurrent
from models.utils import _remote_method_async, _remote_method, LocalRRef
from utils import get_score, get_optimal_cutoff

try:
    from gids_eval.cdr import build_prediction_rows, write_prediction_rows
    from gids_eval.process_monitor import maybe_profile_phase
except Exception:
    build_prediction_rows = None
    write_prediction_rows = None
    maybe_profile_phase = None

DEFAULT_DDP_PORT = 22034
DEFAULT_RPC_PORT = 22206

DEFAULT_TR = {
    'lr': 0.01,
    'epochs': 3,
    'min': 1,
    'patience': 5,
    'nratio': 1,
    'val_nratio': 1,
}

WORKER_ARGS = [32,32]
RNN_ARGS = [32,32,16,1]

WORKERS=4
W_THREADS=8
M_THREADS=2

TMP_FILE = 'tmp_cic.dat'
MODEL_PATH = 'model_save.pkl'
SCORE_FILE = 'scores.txt'
PREDICTION_INDEX = 0

LOAD_FN = None

torch.set_num_threads(1)
DEVICE = torch.device('cpu')
DEVICES = []

def _local_mode_enabled() -> bool:
    raw = os.environ.get("EG1_EULER_LOCAL_MODE", "")
    return raw.strip().lower() not in {"", "0", "false", "no"}

LOCAL_MODE = _local_mode_enabled()

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def _is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
        except OSError:
            return False
        return True

def _pick_ddp_port():
    ddp_port = int(os.environ.get('DDP_PORT', DEFAULT_DDP_PORT))
    if not _is_port_available(ddp_port):
        ddp_port = _find_free_port()
    return str(ddp_port)

def _pick_rpc_init_method():
    env_init = os.environ.get('RPC_INIT_METHOD')
    if env_init:
        if env_init.startswith('tcp://'):
            try:
                port = int(env_init.rsplit(':', 1)[1])
            except ValueError:
                port = _find_free_port()
                env_init = f"tcp://127.0.0.1:{port}"
            else:
                if not _is_port_available(port):
                    port = _find_free_port()
                    env_init = f"tcp://127.0.0.1:{port}"
        return env_init

    rpc_port_env = os.environ.get('RPC_PORT')
    if rpc_port_env:
        try:
            rpc_port = int(rpc_port_env)
        except ValueError:
            rpc_port = _find_free_port()
        if not _is_port_available(rpc_port):
            rpc_port = _find_free_port()
        return f"tcp://127.0.0.1:{rpc_port}"

    rpc_store_dir = os.environ.get('RPC_FILESTORE_DIR', tempfile.gettempdir())
    os.makedirs(rpc_store_dir, exist_ok=True)
    store_name = f"rpc_store_{uuid.uuid4().hex}"
    return f"file://{os.path.join(rpc_store_dir, store_name)}"

def _get_cuda_devices():
    return []

def get_work_units(num_workers, start, end, delta, isTe):
    slices_needed = math.ceil((end-start) / delta)

    per_worker = [slices_needed // num_workers] * num_workers

    remainder = slices_needed % num_workers
    if remainder:
        for i in range(num_workers, num_workers-remainder, -1):
            per_worker[i-1]+=1

    load_threads = W_THREADS*2 if isTe else W_THREADS

    min_threads = min(8, load_threads*num_workers)
    t_per_worker = max(1, min_threads//num_workers)

    print("Tasks: %s" % str(per_worker))
    kwargs = []
    prev = start

    for i in range(num_workers):
            end_t = min(prev + delta*per_worker[i], end)
            kwargs.append({
                'start': prev,
                'end': end_t,
                'delta': delta,
                'is_test': isTe,
                'jobs': t_per_worker
            })
            prev = end_t

    return kwargs

def init_workers(num_workers, start, end, delta, isTe, worker_constructor, worker_args):
    kwargs = get_work_units(num_workers, start, end, delta, isTe)

    rrefs = []
    for i in range(len(kwargs)):
        rrefs.append(
            rpc.remote(
                'worker'+str(i),
                worker_constructor,
                args=(LOAD_FN, kwargs[i], *worker_args),
                kwargs={'head': i==0}
            )
        )

    return rrefs

def init_empty_workers(num_workers, worker_constructor, worker_args):
    empty = {'jobs': 0, 'start': None, 'end': None}

    rrefs = [
        rpc.remote(
            'worker'+str(i),
            worker_constructor,
            args=(LOAD_FN, empty, *worker_args),
            kwargs={'head': i==0}
        )
        for i in range(num_workers)
    ]

    return rrefs

def init_procs(rank, world_size, ddp_port, rpc_init_method, model_path, tmp_file, rnn_constructor, rnn_args, worker_constructor, worker_args,
                times, just_test, lambda_param, impl, load_fn, tr_args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = ddp_port
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'

    global DEVICES, DEVICE, MODEL_PATH, TMP_FILE
    MODEL_PATH = model_path
    TMP_FILE = tmp_file
    DEVICES = _get_cuda_devices()
    DEVICE = torch.device('cpu')
    os.environ['TP_ENABLE_SHM'] = '0'
    os.environ['TP_DISABLE_SHM'] = '1'
    os.environ['TP_TRANSPORTS'] = 'uv'
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method=rpc_init_method
    rpc_backend_options.rpc_timeout = 0
    try:
        rpc_backend_options.transports = ['uv']
    except Exception:
        pass

    global LOAD_FN
    LOAD_FN = load_fn

    if rank == world_size-1:
        torch.set_num_threads(M_THREADS)
        rpc.init_rpc(
            'master', rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

        if just_test:
            rrefs = init_empty_workers(
                world_size-1,
                worker_constructor,
                worker_args
            )

            rnn = rnn_constructor(*rnn_args).to(DEVICE)
            model = DetectorRecurrent(rnn, rrefs) if impl=='DETECT'\
                else PredictorRecurrent(rnn, rrefs)
            model = model.to(DEVICE)

            states = pickle.load(open(MODEL_PATH, 'rb'))
            model.load_states(*states['states'])
            h0 = states['h0']
            if isinstance(h0, torch.Tensor):
                h0 = h0.to(DEVICE)
            tpe = 0
            tr_time = 0

        else:
            rrefs = init_workers(
                world_size-1,
                times['tr_start'], times['tr_end'], times['delta'], False,
                worker_constructor, worker_args
            )

            tmp = time.time()
            if maybe_profile_phase:
                with maybe_profile_phase("train"):
                    model, h0, tpe = train(rrefs, tr_args, rnn_constructor, rnn_args, impl)
            else:
                model, h0, tpe = train(rrefs, tr_args, rnn_constructor, rnn_args, impl)
            tr_time = time.time() - tmp

        h0, zs = get_cutoff(model, h0, times, tr_args, lambda_param)
        stats = []

        for te_start,te_end in times['te_times']:
            test_times = {
                'te_start': te_start,
                'te_end': te_end,
                'delta': times['delta']
            }
            st = test(model, h0, test_times, rrefs)
            for s in st:
                s['TPE'] = tpe
                s['tr_time'] = tr_time
                s.setdefault('train_time', tr_time)
                s.setdefault('test_time', s.get('FwdTime', 0))
                s.setdefault('precision', s.get('P', 0.0))
                s.setdefault('recall', s.get('TPR', 0.0))

            stats += st

    else:
        torch.set_num_threads(W_THREADS)

        dist.init_process_group(
            'gloo', rank=rank,
            world_size=world_size-1
        )

        rpc.init_rpc(
            'worker'+str(rank),
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )

    rpc.shutdown()

    if rank == world_size-1:
        pickle.dump(stats, open(TMP_FILE, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

def train(rrefs, kwargs, rnn_constructor, rnn_args, impl):
    rnn = rnn_constructor(*rnn_args).to(DEVICE)
    model = DetectorRecurrent(rnn, rrefs) if impl=='DETECT'\
        else PredictorRecurrent(rnn, rrefs)
    model = model.to(DEVICE)

    opt = DistributedOptimizer(
        Adam, model.parameter_rrefs(), lr=kwargs['lr']
    )

    times = []
    best = (model.save_states(), 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        model.train()
        with dist_autograd.context() as context_id:
            print("forward")
            st = time.time()
            zs = model.forward(TData.TRAIN)
            loss = model.loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'])

            print("backward")
            dist_autograd.backward(context_id, loss)

            print("step")
            opt.step(context_id)

            elapsed = time.time()-st
            times.append(elapsed)
            l = torch.stack(loss).sum()
            print('[%d] Loss %0.4f  %0.2fs' % (e, l.item(), elapsed))

        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)
            if p.numel() == 0 or n.numel() == 0:
                print("\tValidation edge set empty; falling back to train split", end='')
                p, n = model.score_edges(zs, TData.TRAIN, kwargs['nratio'])

            auc,ap = get_score(p,n)
            print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')

            tot = auc+ap
            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= kwargs['min']:
                    no_progress += 1

            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break

    model.load_states(*best[0])

    zs, h0 = model(TData.TEST, include_h=True)

    states = {'states': best[0], 'h0': h0}
    f = open(MODEL_PATH, 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)

    return model, h0, tpe

def train_local(model, kwargs):
    opt = Adam(model.parameter_rrefs(), lr=kwargs['lr'])

    times = []
    best = (model.save_states(), 0)
    no_progress = 0
    for e in range(kwargs['epochs']):
        model.train()
        opt.zero_grad()
        print("forward")
        st = time.time()
        zs = model.forward(TData.TRAIN)
        loss = model.loss_fn(zs, TData.TRAIN, nratio=kwargs['nratio'])

        print("backward")
        l = torch.stack(loss).sum()
        l.backward()

        print("step")
        opt.step()

        elapsed = time.time()-st
        times.append(elapsed)
        print('[%d] Loss %0.4f  %0.2fs' % (e, l.item(), elapsed))

        model.eval()
        with torch.no_grad():
            zs = model.forward(TData.TRAIN, no_grad=True)
            p,n = model.score_edges(zs, TData.VAL)
            if p.numel() == 0 or n.numel() == 0:
                print("\tValidation edge set empty; falling back to train split", end='')
                p, n = model.score_edges(zs, TData.TRAIN, kwargs['nratio'])

            auc,ap = get_score(p,n)
            print("\tValidation: AP: %0.4f  AUC: %0.4f" % (ap, auc), end='')

            tot = auc+ap
            if tot > best[1]:
                print('*\n')
                best = (model.save_states(), tot)
                no_progress = 0
            else:
                print('\n')
                if e >= kwargs['min']:
                    no_progress += 1

            if no_progress == kwargs['patience']:
                print("Early stopping!")
                break

    model.load_states(*best[0])

    zs, h0 = model(TData.TEST, include_h=True)

    states = {'states': best[0], 'h0': h0}
    f = open(MODEL_PATH, 'wb+')
    pickle.dump(states, f, protocol=pickle.HIGHEST_PROTOCOL)

    tpe = sum(times)/len(times)
    print("Exiting train loop")
    print("Avg TPE: %0.4fs" % tpe)

    return model, h0, tpe

def get_cutoff(model, h0, times, kwargs, lambda_param):
    Encoder = DetectorEncoder if isinstance(model, DetectorRecurrent)\
        else PredictorEncoder
    empty_z = torch.empty((0,), device=DEVICE)

    def _load_remote_train_cutoff():
        _remote_method(
            Encoder.load_new_data,
            model.gcns[0],
            LOAD_FN,
            {
                'start': times['tr_start'],
                'end': times['tr_end'],
                'delta': times['delta'],
                'jobs': 2,
                'is_test': False,
            }
        )
        train_zs = _remote_method(
            Encoder.forward,
            model.gcns[0],
            TData.TRAIN,
            True
        )
        if getattr(train_zs, "size", None) and train_zs.size(0) == 0:
            return torch.empty((0,), device=DEVICE), torch.empty((0,), device=DEVICE)
        return _remote_method(
            Encoder.score_edges,
            model.gcns[0],
            train_zs,
            TData.TRAIN,
            kwargs['nratio']
        )

    _remote_method(
        Encoder.load_new_data,
        model.gcns[0],
        LOAD_FN,
        {
            'start': times['val_start'],
            'end': times['val_end'],
            'delta': times['delta'],
            'jobs': 2,
            'is_test': True
        }
    )

    model.eval()
    zs = _remote_method(
        Encoder.forward,
        model.gcns[0],
        TData.ALL,
        True
    )

    if getattr(zs, "size", None) and zs.size(0) == 0:
        print("No validation time slices found; falling back to train split for cutoff selection")
        p, n = _load_remote_train_cutoff()
        model.cutoff = get_optimal_cutoff(p, n, fw=lambda_param) if p.numel() and n.numel() else 0.5
        print()
        return h0, empty_z

    with torch.no_grad():
        zs, h0 = model.rnn(zs, h0, include_h=True)

    p,n = _remote_method(
        Encoder.score_edges,
        model.gcns[0],
        zs, TData.ALL,
        kwargs['val_nratio']
    )

    if p.numel() == 0 or n.numel() == 0:
        print("Validation cutoff edge set empty; falling back to train split for cutoff selection")
        p, n = _load_remote_train_cutoff()

    model.cutoff = get_optimal_cutoff(p,n,fw=lambda_param) if p.numel() and n.numel() else 0.5

    print()
    return h0, zs[-1] if getattr(zs, "size", None) and zs.size(0) > 0 else empty_z

def test(model, h0, times, rrefs):
    Encoder = DetectorEncoder if isinstance(model, DetectorRecurrent)\
        else PredictorEncoder

    ld_args = get_work_units(
        len(rrefs),
        times['te_start'],
        times['te_end'],
        times['delta'],
        True
    )

    print("Loading test data")

    dont_use = 0
    for ld in ld_args:
        if ld['start'] == ld['end']:
            dont_use += 1
        else:
            break

    futs = [
        _remote_method_async(
            Encoder.load_new_data,
            rrefs[i],
            LOAD_FN,
            ld_args[i+dont_use]
        ) for i in range(len(rrefs)-dont_use)
    ]
    model.num_workers = len(futs)

    [f.wait() for f in futs]
    stats = []

    print("Embedding Test Data...")
    with torch.no_grad():
        model.eval()
        s = time.time()
        zs = model.forward(TData.TEST, h0=h0, no_grad=True)
        ctime = time.time()-s

    print("Scoring")
    if maybe_profile_phase:
        with maybe_profile_phase(f"test_{PREDICTION_INDEX}"):
            score_output = model.score_all(zs)
    else:
        score_output = model.score_all(zs)
    if len(score_output) == 4:
        scores, labels, weights, campaigns = score_output
    else:
        scores, labels, weights = score_output
        campaigns = None
    stats.append(
        score_stats(
            model.__class__.__name__,
            scores, labels, weights, campaigns, model.cutoff, ctime
        )
    )

    model.num_workers = len(rrefs)
    return stats

def score_stats(title, scores, labels, weights, campaigns, cutoff, ctime):
    global PREDICTION_INDEX
    scores = torch.cat(scores, dim=0).detach()
    labels = torch.cat(labels, dim=0).clamp(max=1)
    weights = torch.cat(weights, dim=0)

    device = scores.device
    classified = torch.zeros(labels.size(), device=device)
    classified[scores <= cutoff] = 1

    p_mask = labels == 1
    tpr = classified[p_mask].mean()
    tp = classified[p_mask].sum()
    fn = p_mask.sum() - tp

    f_mask = labels == 0
    fpr = classified[f_mask].mean()
    fp = classified[f_mask].sum()
    tn = f_mask.sum() - fp
    del p_mask, f_mask

    scores_cpu = (1 - scores).detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()
    classified_cpu = classified.detach().cpu().numpy()

    auc = auc_score(labels_cpu, scores_cpu)
    ap = ap_score(labels_cpu, scores_cpu)
    f1 = f1_score(labels_cpu, classified_cpu)
    tp_fp = (tp + fp).float().item()
    tp_fn = (tp + fn).float().item()
    precision = tp.float().item() / tp_fp if tp_fp else 0.0
    fnr = fn.float().item() / tp_fn if tp_fn else 0.0
    tnr = tn.float().item() / (fp + tn).float().item() if (fp + tn).float().item() else 0.0
    recall = tpr.float().item()
    threshold = float(cutoff)

    print(title)
    print("Learned Cutoff %0.4f" % cutoff)
    print("TPR: %0.4f, FPR: %0.4f" % (tpr, fpr))
    print("TP: %d  FP: %d" % (tp, fp))
    print("F1: %0.8f" % f1)
    print("AUC: %0.4f  AP: %0.4f\n" % (auc,ap))

    predictions_path = os.environ.get("GIDS_PREDICTIONS_PATH")
    dataset = os.environ.get("GIDS_JOB_DATASET", title)
    if predictions_path and build_prediction_rows and write_prediction_rows:
        prediction_rows = build_prediction_rows(
            dataset=dataset,
            scores=scores_cpu,
            labels=labels_cpu,
            weights=weights.detach().cpu().numpy(),
            predicted=classified_cpu,
            campaign=campaigns,
            eval_index=PREDICTION_INDEX,
        )
        write_prediction_rows(predictions_path, prediction_rows, overwrite=(PREDICTION_INDEX == 0))
    PREDICTION_INDEX += 1

    return {
        'Model': title,
        'threshold': threshold,
        'TPR':tpr.item(),
        'FPR':fpr.item(),
        'FNR': fnr,
        'TNR': tnr,
        'TP':tp.item(),
        'FP':fp.item(),
        'FN':fn.item(),
        'TN':tn.item(),
        'P':precision,
        'precision': precision,
        'recall': recall,
        'F1':f1,
        'AUC':auc,
        'AP': ap,
        'FwdTime':ctime,
        'test_time': ctime
    }

def run_all_local(rnn_constructor, rnn_args, worker_constructor, worker_args, times, just_test, lambda_param, impl, load_fn, tr_args):
    print("Running EULER in local single-process mode")

    global LOAD_FN
    LOAD_FN = load_fn

    if just_test:
        worker = worker_constructor(LOAD_FN, {'jobs': 0, 'start': None, 'end': None}, *worker_args, head=True)
        rrefs = [LocalRRef(worker)]

        rnn = rnn_constructor(*rnn_args).to(DEVICE)
        model = DetectorRecurrent(rnn, rrefs) if impl == 'DETECT'\
            else PredictorRecurrent(rnn, rrefs)
        model = model.to(DEVICE)

        states = pickle.load(open(MODEL_PATH, 'rb'))
        model.load_states(*states['states'])
        h0 = states['h0']
        if isinstance(h0, torch.Tensor):
            h0 = h0.to(DEVICE)
        elif isinstance(h0, (tuple, list)):
            h0 = tuple(t.to(DEVICE) for t in h0)
        tpe = 0
        tr_time = 0
    else:
        kwargs = get_work_units(1, times['tr_start'], times['tr_end'], times['delta'], False)[0]
        worker = worker_constructor(LOAD_FN, kwargs, *worker_args, head=True)
        rrefs = [LocalRRef(worker)]

        rnn = rnn_constructor(*rnn_args).to(DEVICE)
        model = DetectorRecurrent(rnn, rrefs) if impl == 'DETECT'\
            else PredictorRecurrent(rnn, rrefs)
        model = model.to(DEVICE)

        tmp = time.time()
        if maybe_profile_phase:
            with maybe_profile_phase("train"):
                model, h0, tpe = train_local(model, tr_args)
        else:
            model, h0, tpe = train_local(model, tr_args)
        tr_time = time.time() - tmp

    h0, zs = get_cutoff(model, h0, times, tr_args, lambda_param)
    stats = []

    for te_start,te_end in times['te_times']:
        test_times = {
            'te_start': te_start,
            'te_end': te_end,
            'delta': times['delta']
        }
        st = test(model, h0, test_times, rrefs)
        for s in st:
            s['TPE'] = tpe
            s['tr_time'] = tr_time
            s.setdefault('train_time', tr_time)
            s.setdefault('test_time', s.get('FwdTime', 0))
            s.setdefault('precision', s.get('P', 0.0))
            s.setdefault('recall', s.get('TPR', 0.0))

        stats += st

    return stats

def run_all(workers, rnn_constructor, rnn_args, worker_constructor,
            worker_args, delta, just_test, lambda_param, impl, load_fn,
            tr_start, tr_end, val_times, te_times, tr_args):

    if val_times is None:
        val = max((tr_end - tr_start) // 20, delta*2)
        val_start = tr_end-val
        val_end = tr_end
        tr_end = val_start
    else:
        val_start = val_times[0]
        val_end = val_times[1]

    max_workers = int((tr_end-tr_start) // delta)
    workers = max(min(max_workers, workers), 1)

    times = {
        'tr_start': tr_start,
        'tr_end': tr_end,
        'val_start': val_start,
        'val_end': val_end,
        'te_times': te_times,
        'delta': delta
    }

    print(times)

    if LOCAL_MODE:
        return run_all_local(
            rnn_constructor,
            rnn_args,
            worker_constructor,
            worker_args,
            times,
            just_test,
            lambda_param,
            impl,
            load_fn,
            tr_args,
        )

    ddp_port = _pick_ddp_port()
    rpc_init_method = _pick_rpc_init_method()
    model_path = MODEL_PATH
    tmp_file = TMP_FILE
    print(f"Using DDP port {ddp_port}, RPC init {rpc_init_method}")

    world_size = workers+1
    mp.spawn(
        init_procs,
        args=(
            world_size,
            ddp_port,
            rpc_init_method,
            model_path,
            tmp_file,
            rnn_constructor,
            rnn_args,
            worker_constructor,
            worker_args,
            times,
            just_test,
            lambda_param,
            impl,
            load_fn,
            tr_args
        ),
        nprocs=world_size,
        join=True
    )

    stats = pickle.load(open(TMP_FILE, 'rb'))

    print(stats)
    return stats

if __name__ == '__main__':
    print("Please run this file using run.py")
