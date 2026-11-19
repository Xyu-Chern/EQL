import copy
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from agent.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)

from agent.model import Model


    
def unpack_q(q_func, state, action):
        res = q_func(state, action)
        if isinstance(res, tuple):
            return res
        elif torch.is_tensor(res):
            if res.dim() > 1:
                res = res.mean(dim=-1, keepdim=True)
            return res, res
        else:
            raise ValueError("Unexpected Q-function output type: {}".format(type(res)))

class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, noise_ratio,
                 beta_schedule='vp', n_timesteps=1000,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True,
                 behavior_sample=16, eval_sample=512, deterministic=False, guidance_step=10, guidance_rate=1.0):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = Model(state_dim, action_dim)

        self.max_noise_ratio = noise_ratio
        self.noise_ratio = noise_ratio

        self.behavior_sample = behavior_sample
        self.eval_sample = eval_sample
        self.deterministic = deterministic

        self.guidance_step = guidance_step
        self.guidance_rate = guidance_rate
       
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
       
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio


    @torch.no_grad()
    def p_sample_loop(self, state, shape):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

        return x

    @torch.no_grad()
    def p_sample_loop_dsg(self, state, shape, q_func):
        '''
        DSG Sample

        Returns:
            a single action 
        '''

        guidance_step = self.guidance_step
        guidance_rate = self.guidance_rate

        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)


        for t in reversed(range(self.n_timesteps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            eps_pred = self.model(x, timesteps, state)
            x0_pred = (
                extract(self.sqrt_recip_alphas_cumprod, timesteps, x.shape) * x -
                extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x.shape) * eps_pred
            )

            # Compute μ_theta(x_t)
            mu_theta = (
                extract(self.posterior_mean_coef1, timesteps, x.shape) * x0_pred +
                extract(self.posterior_mean_coef2, timesteps, x.shape) * x
            )

            sigma_t = torch.sqrt(extract(self.posterior_variance, timesteps, x.shape))
            eps_std = torch.randn_like(x)
            
            if t < guidance_step and q_func is not None:
                x_t = x.clone().detach().requires_grad_(True)
                # print("using dsg sampling")
                with torch.enable_grad():
                    eps_pred = self.model(x_t, timesteps, state)
                    x0_pred = (
                        extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t -
                        extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * eps_pred
                    )

                    q1, q2 = unpack_q(q_func, state, x0_pred)
                    q = -torch.min(q1, q2).sum()
                grad = torch.autograd.grad(q, x_t, retain_graph=False, create_graph=False)[0]
                d_star = -math.sqrt(self.action_dim) * sigma_t * grad / (grad.norm(dim=-1, keepdim=True) + 1e-8)
                d_sample = sigma_t * eps_std
                d_mixed = d_sample + guidance_rate * (d_star - d_sample)
    
                r = math.sqrt(self.action_dim) * sigma_t
                x = mu_theta + r * d_mixed / (d_mixed.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                x = mu_theta + sigma_t * eps_std

        x0_pred = x0_pred.detach()
       
        return x0_pred


    @torch.no_grad()
    def sample(self, state, eval=False, q_func=None, normal=False):
        if self.deterministic:
            self.noise_ratio = 0 if eval else self.max_noise_ratio
        else:
            self.noise_ratio = self.max_noise_ratio

        if normal:
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.p_sample_loop(state, shape)
            action.clamp_(-1., 1.)
            return action

        if eval:
            raw_batch_size = state.shape[0]
            state = state.repeat(self.eval_sample, 1)
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.p_sample_loop(state, shape)
            action.clamp_(-1., 1.)
            q = q_func.q(state, action)
            action = action.view(self.eval_sample, raw_batch_size, -1).transpose(0,1)
            q = q.view(self.eval_sample, raw_batch_size, -1).transpose(0,1)
            action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1,1,self.action_dim)
            return action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)
        else:
            raw_batch_size = state.shape[0]
            state = state.repeat(self.behavior_sample, 1)
            batch_size = state.shape[0]
            shape = (batch_size, self.action_dim)
            action = self.p_sample_loop(state, shape)
            action.clamp_(-1., 1.)
            q = q_func.q(state, action)
            action = action.view(self.behavior_sample, raw_batch_size, -1).transpose(0,1)
            q = q.view(self.behavior_sample, raw_batch_size, -1).transpose(0,1)
            action_idx = torch.argmax(q, dim=1, keepdim=True).repeat(1,1,self.action_dim)
            return action.gather(dim=1, index=action_idx).view(raw_batch_size, -1)

    # ------------------------------------------ training ------------------------------------------#
    @torch.no_grad()
    def sample_dsg(self, state, q_func):
        """
        Sample one DSG-guided action for each state in the batch.

        Returns:
            old_state: (B, state_dim)
            action: (B, action_dim)
            (q, v): q-value and batch average
            (mean, std): stats across batch
        """
        self.noise_ratio = self.max_noise_ratio
        old_state = state  

        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)

        # DSG guide 
        action = self.p_sample_loop_dsg(state, shape, q_func)
        action.clamp_(-1., 1.)

        # Compute Q 
        q1, q2 = unpack_q(q_func, state, action)
        q = torch.min(q1, q2)

        # keep returns with CGPO
        mean = q.mean()
        std = q.std()
        v = q.mean(dim=1, keepdim=True)

        if q.ndim > 1:
            # quantile critic 
            q = q.mean(dim=tuple(range(1, q.ndim)))
        return old_state, action, (q.view(batch_size, 1), v), (mean, std)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss


    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, eval=False, q_func=None, normal=False):
        return self.sample(state, eval, q_func, normal)

