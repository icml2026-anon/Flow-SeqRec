"""Flow Matching module: learns a continuous-time ODE velocity field
that transports a noise distribution to the target item embedding.

The key idea is to define a probability path p_t from a simple prior
p_0 (Gaussian) to the data distribution p_1, and learn the velocity
field v_theta(x, t, cond) that generates this path. At inference time
a single Euler step suffices for high-quality recommendations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding following the diffusion literature."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t -- (batch,) in [0, 1]"""
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class VelocityNetwork(nn.Module):
    """Neural network that predicts the velocity field v_theta(x_t, t, cond).

    Architecture: condition and time embeddings are injected via
    adaptive layer normalization (FiLM-style modulation).

    Parameters
    ----------
    data_dim : int
        Dimension of the item embedding space.
    cond_dim : int
        Dimension of the conditioning vector (user representation).
    hidden_dim : int
        Hidden layer dimension.
    num_layers : int
        Number of residual blocks.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        self.time_embed = TimeEmbedding(hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        self.blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
            )
            # FiLM modulation: produces scale and shift from time + condition
            self.film_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                )
            )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity at (x_t, t) conditioned on user representation.

        Args:
            x_t: (batch, data_dim) -- noisy point on the ODE trajectory
            t: (batch,) -- time in [0, 1]
            cond: (batch, cond_dim) -- user representation

        Returns:
            v: (batch, data_dim) -- predicted velocity
        """
        t_emb = self.time_embed(t)  # (B, hidden_dim)
        c_emb = self.cond_proj(cond)  # (B, hidden_dim)
        h = self.input_proj(x_t)  # (B, hidden_dim)

        for block, film in zip(self.blocks, self.film_layers):
            # FiLM conditioning
            film_input = torch.cat([t_emb, c_emb], dim=-1)  # (B, hidden_dim*2)
            film_params = film(film_input)  # (B, hidden_dim*2)
            scale, shift = film_params.chunk(2, dim=-1)  # each (B, hidden_dim)
            # Residual block with modulation
            h = h + block(h) * (1 + scale) + shift

        return self.output_proj(h)


class FlowMatchingModule(nn.Module):
    """Flow Matching training and inference module.

    During training, we sample a random time t ~ U(0,1), construct the
    interpolated point x_t = (1-t)*x_0 + t*x_1, and regress the velocity
    network output to the ground-truth velocity (x_1 - x_0).

    During inference, we start from noise x_0 ~ N(0, I) and integrate
    the learned velocity field using a single Euler step (or multi-step).

    Parameters
    ----------
    data_dim : int
        Item embedding dimension.
    cond_dim : int
        User representation dimension.
    hidden_dim : int
        Velocity network hidden dim.
    num_layers : int
        Velocity network depth.
    sigma_min : float
        Minimum noise scale for the prior.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.sigma_min = sigma_min

        self.velocity_net = VelocityNetwork(
            data_dim=data_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def compute_loss(
        self,
        x_1: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Conditional flow matching loss.

        Args:
            x_1: (batch, data_dim) -- target item embedding
            cond: (batch, cond_dim) -- user representation

        Returns:
            loss: scalar MSE between predicted and true velocity
        """
        batch_size = x_1.size(0)
        device = x_1.device

        # Sample noise (source distribution)
        x_0 = torch.randn_like(x_1)

        # Sample time uniformly
        t = torch.rand(batch_size, device=device)

        # Optimal transport conditional path: x_t = (1 - t)*x_0 + t*x_1
        t_expand = t.unsqueeze(-1)  # (B, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        # Ground-truth velocity for OT path
        v_target = x_1 - x_0

        # Predicted velocity
        v_pred = self.velocity_net(x_t, t, cond)

        loss = F.mse_loss(v_pred, v_target)
        return loss

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """Generate item embeddings via ODE integration (Euler method).

        Args:
            cond: (batch, cond_dim) -- user representations
            num_steps: number of Euler steps (1 for single-step inference)

        Returns:
            x: (batch, data_dim) -- predicted item embeddings
        """
        batch_size = cond.size(0)
        device = cond.device

        # Start from Gaussian noise
        x = torch.randn(batch_size, self.data_dim, device=device)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = self.velocity_net(x, t, cond)
            x = x + v * dt

        return x
