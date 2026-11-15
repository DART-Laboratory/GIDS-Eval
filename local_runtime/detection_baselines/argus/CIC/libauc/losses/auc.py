
import torch
import torch.nn.functional as F
from .surrogate import squared_loss, squared_hinge_loss, logistic_loss

def _check_tensor_shape(inputs, shape=(-1, 1)):
    input_shape = inputs.shape
    target_shape = shape
    if len(input_shape) != len(target_shape):
        inputs = inputs.reshape(target_shape)
    return inputs

def _get_surrogate_loss(backend='squared_hinge'):
    if backend == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif backend == 'squared':
       surr_loss = squared_loss
    elif backend == 'logistic':
       surr_loss = logistic_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss

class AUCMLoss_V2(torch.nn.Module):
    def __init__(self, margin=1.0, device=None):
        super(AUCMLoss_V2, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        pos_mask = (1==y_true).float()
        neg_mask = (0==y_true).float()
        loss = self.mean((y_pred - self.a)**2*pos_mask) +\
               self.mean((y_pred - self.b)**2*neg_mask) +\
               2*self.alpha*(self.margin + self.mean(y_pred*neg_mask) - self.mean(y_pred*pos_mask)) -\
               self.alpha**2
        return loss

class AUCM_MultiLabel_V2(torch.nn.Module):
    def __init__(self, margin=1.0, num_classes=10, device=None):
        super(AUCM_MultiLabel_V2, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.num_classes = num_classes
        self.a = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    @property
    def get_a(self):
        return self.a.mean()

    @property
    def get_b(self):
        return self.b.mean()

    @property
    def get_alpha(self):
        return self.alpha.mean()

    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true):
        total_loss = 0
        for idx in range(self.num_classes):
            y_pred_i = _check_tensor_shape(y_pred[:, idx], (-1,1))
            y_true_i = _check_tensor_shape(y_true[:, idx], (-1,1))
            pos_mask = (1==y_true_i).float()
            neg_mask = (0==y_true_i).float()
            loss = self.mean((y_pred_i - self.a[idx])**2*pos_mask) +\
                   self.mean((y_pred_i - self.b[idx])**2*neg_mask) +\
                   2*self.alpha[idx]*(self.margin + self.mean(y_pred_i*neg_mask) - self.mean(y_pred_i*pos_mask)) -\
                   self.alpha[idx]**2
            total_loss += loss
        return total_loss

class CompositionalAUCLoss_V2(torch.nn.Module):
    def __init__(self, margin=1.0, backend='ce', device=None):
        super(CompositionalAUCLoss_V2, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.L_AVG = F.binary_cross_entropy_with_logits
        self.backend = 'ce'

    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        if self.backend == 'ce':
           self.backend = 'auc'
           return self.L_AVG(y_pred, y_true)
        else:
           self.backend = 'ce'
           y_pred = torch.sigmoid(y_pred)
           pos_mask = (1==y_true).float()
           neg_mask = (0==y_true).float()
           self.L_AUC = self.mean((y_pred - self.a)**2*pos_mask) +\
                        self.mean((y_pred - self.b)**2*neg_mask) +\
                        2*self.alpha*(self.margin + self.mean(y_pred*neg_mask) - self.mean(y_pred*pos_mask) ) -\
                        self.alpha**2
           return self.L_AUC

class APLoss(torch.nn.Module):

    def __init__(self, pos_len=None, margin=1.0, gamma=0.99, surrogate_loss='squared_hinge', device=None):
        super(APLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        else:
            self.device = device
        self.u_all = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, y_pred, y_true, index_p):
        CHECKMEORY=False
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        index_p = index_p[index_p>=0]

        pos_mask = (y_true == 1).flatten()
        f_ps = y_pred[pos_mask]
        mat_data = y_pred.flatten().repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        pos_sur_loss = sur_loss * pos_mask
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_all[index_p] = (1 - self.gamma) * self.u_all[index_p] + self.gamma * (sur_loss.mean(1, keepdim=True)).detach()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_pos[index_p] = (1 - self.gamma) * self.u_pos[index_p] + self.gamma * (pos_sur_loss.mean(1, keepdim=True)).detach()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        p = (self.u_pos[index_p] - (self.u_all[index_p]) * pos_mask) / (self.u_all[index_p] ** 2)

        loss = torch.mean(p * sur_loss)

        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        return loss

class pAUC_CVaR_Loss(torch.nn.Module):
    def __init__(self,
                 pos_len,
                 num_neg=None,
                 margin=1.0,
                 alpha=0,
                 beta=0.2,
                 surrogate_loss='squared_hinge'):
        super(pAUC_CVaR_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eta = 1.0
        self.num_neg = num_neg
        self.pos_len = pos_len
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).cuda()
        self.margin = margin
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def set_coef(self, eta):
        self.eta = eta

    def update_coef(self, decay_factor):
        self.eta = self.eta/decay_factor

    @property
    def get_coef(self):
        return self.eta

    def forward(self, y_pred, y_true, index_p, auto=True):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))

        f_ps = y_pred[y_true == 1].reshape(-1, 1)
        f_ns = y_pred[y_true == 0].reshape(-1, 1)
        f_ps = f_ps.repeat(1, len(f_ns))
        f_ns = f_ns.repeat(1, len(f_ps))
        index_p = index_p[index_p>=0]
        if auto:
           self.num_neg = (y_true == 0).float().sum()
           assert self.num_neg > 0
        loss = self.surrogate_loss(self.margin, f_ps - f_ns.transpose(0,1))
        p = loss > self.u_pos[index_p]
        self.u_pos[index_p] = self.u_pos[index_p]-self.eta/self.pos_len*(1 - p.sum(dim=1, keepdim=True)/(self.beta*self.num_neg))
        p.detach_()
        loss = torch.mean(p * loss) / self.beta
        return loss

class pAUC_DRO_Loss(torch.nn.Module):
    def __init__(self,
                 pos_len,
                 gamma=0.1,
   margin=1.0,
                 Lambda=1.0,
                 surrogate_loss='squared_hinge',
                 device=None):
        super(pAUC_DRO_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.u_pos = torch.tensor([0.0]*pos_len).view(-1, 1).to(self.device)
        self.margin = margin
        self.gamma = gamma
        self.Lambda = Lambda

        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def set_coef(self, gamma):
        self.gamma = gamma

    def update_coef(self, decay_factor):
        self.gamma /= decay_factor

    @property
    def get_coef(self):
        return self.gamma

    def forward(self, y_pred, y_true, index_p):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))

        f_ps = y_pred[y_true == 1].reshape(-1,1)
        f_ns = y_pred[y_true == 0].reshape(-1)
        index_p = index_p[index_p>=0]
        mat_data = f_ns.repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))
        exp_loss = torch.exp(sur_loss/self.Lambda)
        self.u_pos[index_p] = (1 - self.gamma) * self.u_pos[index_p] + self.gamma * (exp_loss.mean(1, keepdim=True).detach())
        p = exp_loss/self.u_pos[index_p]
        p.detach_()
        loss = torch.mean(p * sur_loss)
        return loss

class tpAUC_KL_Loss(torch.nn.Module):
    def __init__(self,
                 pos_len,
   tau=1.0,
                 Lambda=1.0,
                 margin=1.0,
                 surrogate_loss='squared_hinge',
   device=None):
        super(tpAUC_KL_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.gamma0 = 0.9
        self.gamma1 = 0.9
        self.Lambda = Lambda
        self.tau = tau
        self.u_pos = torch.tensor([0.0]*pos_len).view(-1, 1).to(self.device)
        self.w = 0.0
        self.margin = margin

        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def set_coef(self, gamma0, gamma1):
        self.gamma0 = gamma0
        self.gamma1 = gamma1

    def update_coef(self, decay_factor):
        self.gamma0 = self.gamma0/decay_factor
        self.gamma1 = self.gamma1/decay_factor

    @property
    def get_coef(self):
        return (self.gamma0, self.gamma1)

    def forward(self, y_pred, y_true, index_p):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))

        f_ps = y_pred[y_true == 1].reshape(-1, 1)
        f_ns = y_pred[y_true == 0].reshape(-1, 1)
        f_ps = f_ps.repeat(1, len(f_ns))
        f_ns = f_ns.repeat(1, len(f_ps))
        index_p = index_p[index_p>=0]
        loss = self.surrogate_loss(self.margin, f_ps - f_ns.transpose(0,1))

        exp_loss = torch.exp(loss/self.Lambda).detach_()
        self.u_pos[index_p] = (1 - self.gamma0) * self.u_pos[index_p] + self.gamma0 * (exp_loss.mean(1, keepdim=True))
        self.w = (1 - self.gamma1) * self.w + self.gamma1 * (torch.pow(self.u_pos[index_p], self.Lambda/self.tau).mean())
        p = torch.pow(self.u_pos[index_p], self.Lambda/self.tau - 1) * exp_loss/self.w
        p.detach_()
        loss = torch.mean(p * loss)
        return loss

class pAUCLoss(torch.nn.Module):
    def __init__(self, backend='SOPA', **kwargs):
        super(pAUCLoss, self).__init__()
        assert backend in ['SOPA', 'SOPAs', 'SOTA'], 'Keyword is not found!'
        print ('Backend loss: %s'%backend)
        self.backend = backend
        self.loss_fn = self.get_loss(backend, **kwargs)

    def get_loss(self, backend='SOPA', **kwargs):
        if backend == 'SOPA':
           loss = pAUC_CVaR_Loss(**kwargs)
        elif backend == 'SOPAs':
           loss = pAUC_KLDRO_Loss(**kwargs)
        elif backend == 'SOTA':
           loss = tpAUC_KL_Loss(**kwargs)
        else:
            raise ValueError('Out of options!')
        return loss

    def forward(self, y_pred, y_true, index_p):
        return self.loss_fn(y_pred, y_true, index_p)

pAUC_Loss = pAUCLoss

AUCM_MultiLabel_V2
AUCMLoss_V2
CompositionalAUCLoss_V2
