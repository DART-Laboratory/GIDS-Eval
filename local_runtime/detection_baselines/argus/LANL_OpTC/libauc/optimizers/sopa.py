import math
import torch

class SOPA(torch.optim.Optimizer):

    def __init__(self, model, loss_fn,
                 mode = 'adam',
                 eta=1.0, lr=1e-3, weight_decay=0,
                 betas=(0.9, 0.999), eps=1e-8, amsgrad=False,
                 momentum=0.9, nesterov=False, dampening=0
                ):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        try:
            params = model.parameters()
        except:
            params = model

        self.params = params
        self.lr = lr
        self.mode = mode.lower()
        self.loss_fn = loss_fn
        self.loss_fn.set_coef(eta)
        self.steps = 0

        defaults = dict(lr=lr, betas=betas, eps=eps, momentum=momentum, nesterov=nesterov, dampening=dampening,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(SOPA, self).__init__(self.params, defaults)

    def __setstate__(self, state):
      super(SOPA, self).__setstate__(state)
      for group in self.param_groups:
          if self.mode == 'sgd':
             group.setdefault('nesterov', False)
          elif self.mode == 'adam':
             group.setdefault('amsgrad', False)
          else:
             NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self.mode == 'adam':
              self.lr = group['lr']
              for i, p in enumerate(group['params']):
                  if p.grad is None:
                      continue
                  grad = p.grad
                  if grad.is_sparse:
                      raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                  amsgrad = group['amsgrad']
                  state = self.state[p]

                  if len(state) == 0:
                      state['step'] = 0

                      state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                      state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                      if amsgrad:

                          state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                  exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                  if amsgrad:
                      max_exp_avg_sq = state['max_exp_avg_sq']
                  beta1, beta2 = group['betas']
                  state['step'] += 1
                  bias_correction1 = 1 - beta1 ** state['step']
                  bias_correction2 = 1 - beta2 ** state['step']
                  if group['weight_decay'] != 0:
                      grad = grad.add(p, alpha=group['weight_decay'])
                  exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                  exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                  if amsgrad:

                      torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                      denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                  else:
                      denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                  step_size = group['lr'] / bias_correction1
                  p.addcdiv_(exp_avg, denom, value=-step_size)
            elif self.mode == 'sgd':
              weight_decay = group['weight_decay']
              momentum = group['momentum']
              dampening = group['dampening']
              nesterov = group['nesterov']
              self.lr = group['lr']
              for p in group['params']:
                  if p.grad is None:
                      continue
                  d_p = p.grad
                  if weight_decay != 0:
                      d_p = d_p.add(p, alpha=weight_decay)
                  if momentum != 0:
                      param_state = self.state[p]
                      if 'momentum_buffer' not in param_state:
                          buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                      else:
                          buf = param_state['momentum_buffer']
                          buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                      if nesterov:
                          d_p = d_p.add(buf, alpha=momentum)
                      else:
                          d_p = buf
                  p.add_(d_p, alpha=-group['lr'])
        self.steps += 1
        return loss

    def update_lr(self, decay_factor=None, coef_decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            print ('Reducing lr to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
        if coef_decay_factor != None:
            self.loss_fn.update_coef(coef_decay_factor)
            print ('Reducing eta/gamma to %.5f @ T=%s!!' %(self.loss_fn.get_coef, self.steps))
