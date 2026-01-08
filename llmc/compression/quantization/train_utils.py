import os
import sys
import time
from math import inf

import torch
import torch.nn as nn
from loguru import logger


class AvgMeter:
    def __init__(self):
        self.num = 0
        self.s = 0
        self.m = 0

    def update(self, value):
        self.num += 1
        prev = value - self.m
        self.m = self.m + (value - self.m) / self.num
        now = value - self.m
        self.s = self.s + prev * now

    def get(self):
        # assert self.num > 1
        return round(self.m, 4), round(self.s / (self.num - 1), 5)


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = (
            truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        )
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class LossFunction:
    def __init__(self, method='mse', reduction='mean', dim=0, sw_num_projections=128,
                 sw_block_size=None, hybrid_weights=None):
        """
        Loss function wrapper supporting multiple loss methods.

        Args:
            method: Loss method name ('mse', 'l2', 'dist', 'kl', 'sliced_wasserstein',
                    'sw', 'mse_sw', 'l2_sw')
            reduction: Reduction method for standard losses ('mean', 'sum', 'none')
            dim: Dimension for distribution loss
            sw_num_projections: Number of random projections for Sliced-Wasserstein distance
                               (default: 128, recommended: 128-256)
            sw_block_size: Block size for SW distance. If None, use token-level computation.
                          If specified, treats each block as one sample.
            hybrid_weights: Dict with keys 'base' and 'sw' for hybrid loss weighting.
                          Recommended to sum to 1.0. E.g., {'base': 0.9, 'sw': 0.1}
        """
        self.method = method
        self.reduction = reduction
        self.dim = dim
        self.sw_num_projections = sw_num_projections
        self.sw_block_size = sw_block_size

        # Set default hybrid weights if using hybrid loss (sum to 1.0)
        if hybrid_weights is None:
            self.hybrid_weights = {'base': 0.9, 'sw': 0.1}
        else:
            self.hybrid_weights = hybrid_weights

    def l2_loss(self, x, y):
        return (x - y).pow(2).sum(-1).mean()

    def sliced_wasserstein_loss(self, y_fp, y_quant):
        """
        Compute Sliced-Wasserstein distance between FP and quantized outputs.

        Args:
            y_fp: Full-precision output, typically [batch, seq_len, hidden_dim] or [N, D]
            y_quant: Quantized output with same shape as y_fp

        Returns:
            Scalar tensor representing SW distance
        """
        # Handle different input shapes
        if y_fp.dim() == 2:
            # Already flattened: [N, D]
            y_fp_flat = y_fp.float()
            y_quant_flat = y_quant.float()
        elif y_fp.dim() == 3:
            batch_size, seq_len, hidden_dim = y_fp.shape

            if self.sw_block_size is None:
                # Token-level: N = batch*seq_len, d = hidden_dim
                y_fp_flat = y_fp.reshape(-1, hidden_dim).float()
                y_quant_flat = y_quant.reshape(-1, hidden_dim).float()
            else:
                # Block-level: N = batch*num_blocks, d = block_size*hidden_dim
                assert seq_len % self.sw_block_size == 0, (
                    f"seq_len ({seq_len}) must be divisible by "
                    f"sw_block_size ({self.sw_block_size})"
                )
                num_blocks = seq_len // self.sw_block_size
                # Reshape: [batch, seq_len, hidden_dim]
                #       -> [batch, num_blocks, block_size, hidden_dim]
                #       -> [batch*num_blocks, block_size*hidden_dim]
                y_fp_flat = y_fp.reshape(
                    batch_size, num_blocks, self.sw_block_size * hidden_dim
                ).reshape(-1, self.sw_block_size * hidden_dim).float()
                y_quant_flat = y_quant.reshape(
                    batch_size, num_blocks, self.sw_block_size * hidden_dim
                ).reshape(-1, self.sw_block_size * hidden_dim).float()
        else:
            raise ValueError(
                f"Unsupported tensor dimension: {y_fp.dim()}. "
                f"Expected 2D [N, D] or 3D [batch, seq_len, hidden_dim]"
            )

        N, proj_dim = y_fp_flat.shape

        # Sample random projection vectors: [proj_dim, n_projections]
        u = torch.randn(
            proj_dim, self.sw_num_projections,
            device=y_fp.device, dtype=torch.float32
        )
        # L2-normalize each projection vector
        u = u / (u.norm(dim=0, keepdim=True) + 1e-8)

        # Project both FP and quant outputs onto all random directions in parallel
        # proj_fp / proj_quant: [N, n_projections]
        proj_fp = y_fp_flat @ u
        proj_quant = y_quant_flat @ u

        # Sort along the sample dimension (dim=0) for each projection independently
        # sorted_fp / sorted_quant: [N, n_projections]
        sorted_fp, _ = torch.sort(proj_fp, dim=0)
        sorted_quant, _ = torch.sort(proj_quant, dim=0)

        # Compute 1D Wasserstein-1 distance for each projection:
        # Take elementwise |diff|, mean over samples N -> [n_projections]
        w1_per_proj = torch.abs(sorted_fp - sorted_quant).mean(dim=0)

        # Final SW distance is the mean across projections -> scalar
        return w1_per_proj.mean()

    def __call__(self, f_out, q_out):
        # L2 Loss
        if self.method == 'l2':
            return self.l2_loss(f_out, q_out)

        # MSE Loss
        elif self.method == 'mse':
            mse_loss = nn.MSELoss(reduction=self.reduction)
            return mse_loss(f_out, q_out)

        # Distribution Loss
        elif self.method == 'dist':
            mse_loss = nn.MSELoss(reduction=self.reduction)

            channel_num = f_out.shape[-1]
            f_out = f_out.reshape(-1, channel_num)
            q_out = q_out.reshape(-1, channel_num)

            mean_error = mse_loss(f_out.mean(dim=self.dim), q_out.mean(dim=self.dim))
            std_error = mse_loss(f_out.std(dim=self.dim), q_out.std(dim=self.dim))
            return mean_error + std_error

        # KL divergence Loss
        elif self.method == 'kl':
            kl_loss = nn.KLDivLoss(reduction=self.reduction)
            return kl_loss(f_out, q_out)

        # Sliced-Wasserstein Loss
        elif self.method == 'sliced_wasserstein' or self.method == 'sw':
            return self.sliced_wasserstein_loss(f_out, q_out)

        # Hybrid: MSE + Sliced-Wasserstein
        elif self.method == 'mse_sw':
            mse = nn.MSELoss(reduction=self.reduction)(f_out, q_out)
            sw = self.sliced_wasserstein_loss(f_out, q_out)
            return self.hybrid_weights['base'] * mse + self.hybrid_weights['sw'] * sw

        # Hybrid: L2 + Sliced-Wasserstein
        elif self.method == 'l2_sw':
            l2 = self.l2_loss(f_out, q_out)
            sw = self.sliced_wasserstein_loss(f_out, q_out)
            return self.hybrid_weights['base'] * l2 + self.hybrid_weights['sw'] * sw

        else:
            raise ValueError(f"Unknown loss method: {self.method}")


class NativeScalerWithGradNormCount:
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        retain_graph=False,
    ):
        self._scaler.scale(loss).backward(
            create_graph=create_graph, retain_graph=retain_graph
        )
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = self.ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def ampscaler_get_grad_norm(self, parameters, norm_type=2.0):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        norm_type = float(norm_type)
        if len(parameters) == 0:
            return torch.tensor(0.0)
        device = parameters[0].grad.device
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(
                torch.stack(
                    [
                        torch.norm(p.grad.detach(), norm_type).to(device)
                        for p in parameters
                    ]
                ),
                norm_type,
            )
        return total_norm
