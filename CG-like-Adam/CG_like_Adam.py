import math
from optimizer import Optimizer



class CG_like_Adam(Optimizer):
    def __init__(self, params, lr:float=1e-3, betas:tuple=(0.9, 0.999), eps:float=1e-8, 
                a:float=1/2+1e-5, b:float=0, lambada:float=1, weight_decay:float=0, 
                lambada1:float=2, amsgrad:bool=True, gammatype:str='DY'):
        ## necessary explanation of some parameters ##
        # lambada: set 1 by default so that for all t, beta_1t = beta_11.
        # b: set 0 by default so that for all t, alpha_t (lr_t) = lr.
        # lambada1: a parameter for HZ conjugate coefficient.
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 < betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 1/2 <= a:
            raise ValueError("Invalid a value: {}".format(a))
        if not 0.0 <= b:
            raise ValueError("Invalid b value: {}".format(b))
        if not 0.0 < lambada <= 1:
            raise ValueError("Invalid lambada value: {}".format(lambada))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 1/4 < lambada1:
            raise ValueError("Invalid lambada1 value: {}".format(lambada1))

        defaults = dict(lr=lr, betas=betas, eps=eps, a=a, b=b, lambada=lambada, 
                        weight_decay=weight_decay, lambada1=lambada1, amsgrad=amsgrad, gammatype=gammatype)
        super(CG_like_Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # Gradient
                grad = p.grad.data 
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['a'], state['b'] = group['a'], group['b']
                    # Exponential moving average of gradient values
                    # Initialize m_0 = 0
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_() 
                    # Exponential moving average of squared gradient values
                    # Initialize v_0 = 0 
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                    # Initialize g_previous = 0
                    state['grad_previous'] = grad.new().resize_as_(grad).zero_()
                    state['v_hat_previous'] = grad.new().resize_as_(grad).zero_()
                    # Initialize conjugate gradient direction d_0 = 0
                    state['d_t'] = grad.new().resize_as_(grad).zero_() 
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                a, b = state['a'], state['b']
                beta11, beta2 =group['betas']

                # Update step t
                state['step'] += 1 
                
                if group['weight_decay'] != 0:
                    grad = grad.add_(p.data, alpha = group['weight_decay'])
                
                # Decay the first and second moment running average coefficient
                bias_correction1 = 1 - beta11 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Calculate the conjugate coefficient gamma_t
                if state['step'] == 1:
                    gamma_t = 0
                    theta_t = 0
                    y_t = grad.new().resize_as_(grad).zero_()
                else:
                    grad_pre = state['grad_previous']
                    d_t_pre = state['d_t']
                    # FR
                    if group['gammatype'] == 'FR':
                        gamma_t = grad.mul(grad).sum(0).sum() / (grad_pre.mul(grad_pre).sum(0).sum() + group['eps'])
                    # MFR
                    elif group['gammatype'] == 'MFR':
                        gamma_t = grad.mul(grad).sum(0).sum() / (grad_pre.mul(grad_pre).sum(0).sum() + group['eps'])
                        theta_t = grad.mul(d_t_pre).sum(0).sum() / (grad_pre.mul(grad_pre).sum(0).sum() + group['eps'])
                    # PRP
                    elif group['gammatype'] == 'PRP':
                        y_t = grad - grad_pre
                        gamma_t = grad.mul(y_t).sum(0).sum() / (grad_pre.mul(grad_pre).sum(0).sum() + group['eps'])
                    # MPRP
                    elif group['gammatype'] == 'MPRP':
                        y_t = grad - grad_pre
                        gamma_t = grad.mul(y_t).sum(0).sum() / (grad_pre.mul(grad_pre).sum(0).sum() + group['eps'])
                        theta_t = grad.mul(d_t_pre).sum(0).sum() / (grad_pre.mul(grad_pre).sum(0).sum() + group['eps'])
                    # HS
                    elif group['gammatype'] == 'HS':
                        # gamma_t = 1e-3
                        y_t = grad - grad_pre
                        gamma_t = grad.mul(y_t).sum(0).sum() / (d_t_pre.mul(y_t).sum(0).sum() + group['eps'])
                    # DY
                    elif group['gammatype'] == 'DY':
                        y_t = grad - grad_pre
                        gamma_t = grad.mul(grad).sum(0).sum() / (d_t_pre.mul(y_t).sum(0).sum() + group['eps'])
                    # HZ
                    elif group['gammatype'] == 'HZ':
                        y_t = grad - grad_pre
                        temp = d_t_pre.mul(y_t).sum(0).sum()
                        gamma_t = grad.mul(y_t).sum(0).sum() / (temp + group['eps']) - group['lambada1']*(y_t.mul(y_t).sum(0).sum() / (temp**2 + group['eps']))*(grad.mul(d_t_pre).sum(0).sum())
                    else:
                        raise Exception("Unknow Gamma type: "+str(group['gammatype']))
                
                # Calculate d_t
                if group['gammatype'] == 'MFR':
                    state['d_t'].mul_(-gamma_t).add_(grad.mul(1+theta_t))
                elif group['gammatype'] == 'MPRP':
                    state['d_t'].mul_(-gamma_t).add_(grad).add_(y_t.mul(theta_t))
                else:
                    state['d_t'].mul_(- gamma_t / state['step']**a).add_(grad)
                # Assert the element of d_t not including nan or inf
                if state['d_t'].isnan().any() or state['d_t'].isinf().any():
                    raise ValueError("Invalid d_t value [inf or nan]: {}".format(state['d_t']))
                
                # Calculate beta_1t
                state['beta1t'] = beta11 * group['lambada']**(state['step']-1) # beta_1t的取值范围为[0,1)
                # Calculate m_t
                exp_avg.mul_(state['beta1t']).add_(state['d_t'], alpha=1-state['beta1t'])
                # Assert the element of m_t not including nan or inf
                if exp_avg.isnan().any() or exp_avg.isinf().any():
                    raise ValueError("Invalid exp_avg value [inf or nan]: {}".format(exp_avg))
                
                # Calculate v_t
                exp_avg_sq.mul_(beta2).addcmul_(state['d_t'], state['d_t'], value = 1-beta2) # 实验2
                # # Assert the element of v_t not including nan or inf
                if exp_avg_sq.isnan().any() or exp_avg_sq.isinf().any():
                   raise ValueError("Invalid exp_avg_sq value [inf or nan]: {}".format(exp_avg_sq))
                
                if group['amsgrad']:
                    # Unbiased estimation
                    v_hat_t = exp_avg_sq.div(bias_correction2)
                    # Calculate v_hat_t_max
                    v_hat_t_max = v_hat_t.maximum(state['v_hat_previous'])
                    if v_hat_t_max.isnan().any() or v_hat_t_max.isinf().any():
                        raise ValueError("Invalid v_hat_t_max value [inf or nan]: {}".format(v_hat_t_max))
                    # save v_hat_t_max to v_hat_previous
                    state['v_hat_previous'] = v_hat_t_max.clone().detach()
                    # Calculate v_hat_t
                    v_hat_t = v_hat_t_max.sqrt().add_(group['eps'])
                else:
                    # Calculate v_hat_t
                    v_hat_t = (exp_avg_sq.sqrt().div(math.sqrt(bias_correction2))).add_(group['eps'])
                    if v_hat_t.isnan().any() or v_hat_t.isinf().any():
                        raise ValueError("Invalid v_hat_t value [inf or nan]: {}".format(v_hat_t))
                
                # save grad_t-1
                state['grad_previous'] = grad.clone().detach()
                
                p.data.addcdiv_(exp_avg, v_hat_t, value = - group['lr']/(state['step']**b * bias_correction1))
                if p.data.isnan().any() or p.data.isinf().any():
                    raise ValueError("Invalid pamara value [inf or nan]: {}".format(p.data))
        return loss