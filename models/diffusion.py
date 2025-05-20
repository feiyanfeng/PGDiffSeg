import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.model import get_denoise_model
import cv2
import matplotlib.pyplot as plt
import math

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        # scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas

    """
    def __init__(
        self,
        model,
        # img_size,
        img_channels,
        betas,
        device
    ):
        super().__init__()

        self.model = model
        self.img_channels = img_channels
        self.device = device
        self.num_timesteps = len(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    @torch.no_grad()
    def remove_noise(self, x, y, t):

        return (
            (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, y, t)) *
            extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    @torch.no_grad()
    def sample(self, y, see=False):
        x = torch.randn_like(y)  # TODO: device
        if see:
            print('sample, the first:', torch.min(x), torch.max(x))
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=self.device).repeat(y.shape[0])
            x = self.remove_noise(x, y, t_batch)
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            if see:
                print(f'sample, in time {t}:', torch.min(x), torch.max(x))
        
        return x.cpu().detach()

            
 
    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, y, t):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, y, t)

        return F.mse_loss(estimated_noise, noise)  # TODO: 

    def forward(self, x, y):
        b, c, h, w = x.shape
        device = x.device        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, y, t)


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):

    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    if schedule == "linear2":
        betas = torch.linspace(start * 1000 / num_timesteps, end * 1000 / num_timesteps, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
            range(num_timesteps)])
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
            range(num_timesteps)])
    return betas

def diffusion_model(args):
    model = get_denoise_model(args.model, args.classify)
    #model.to(args.device)
    input_channels = args.model.input_channels
    args = args.diffusion
    betas = make_beta_schedule(schedule=args.beta_schedule,
                            num_timesteps=args.timesteps,
                            start=args.beta_start, 
                            end=args.beta_end)

    return GaussianDiffusion(model, input_channels, betas, args.device) # (args.img_size, args.img_size),
