import torch.distributed.rpc as rpc

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs, timeout=0.0)

def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs, timeout=0.0)

def _param_rrefs(module):
    rrefs = []
    for param in module.parameters():
        rrefs.append(
            rpc.RRef(param)
        )

    return rrefs

def _decoder_rrefs(module):
    rrefs = []
    for param in module.decoder_parameters():
        rrefs.append(
            rpc.RRef(param)
        )

    return rrefs
