# model.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class AEProj(nn.Module):
    """
    AE projector with residual and post-MLP layers.
    """
    def __init__(self, d_in=64, d_out=128, hidden=256, p_drop=0.0, use_gate=True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_out),
        )
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = nn.Parameter(torch.tensor([-4.0]))
        self.post_mlp = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        z_main = self.proj(x)
        z_res = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(x)
        if self.use_gate:
            z_base = z_main + torch.sigmoid(self.gate) * z_res
        else:
            z_base = z_main + z_res
        z_base = F.normalize(z_base, dim=-1)
        z = self.post_mlp(z_base)
        return F.normalize(z, dim=-1)

class AEProjSH(nn.Module):
    """
    AE projector with spherical-harmonics location conditioning.

    The AE branch keeps the original AEProj stem and fuses a projected SH
    location code in hidden space through a near-zero initialized gate.
    """

    def __init__(
        self,
        d_in=64,
        d_out=128,
        hidden=256,
        p_drop=0.0,
        use_gate=True,
        *,
        sh_degree: int = 8,
        pos_hidden: int = 256,
        rho_init: float = 1e-3,
        origin_left: float = -180.0,
        origin_bottom: float = -84.0,
        resolution_deg: float = 360.0 / 400752.0,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, d_out),
        )
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = nn.Parameter(torch.tensor([-4.0]))

        self.ae_to_hidden = nn.Sequential(
            nn.Linear(d_out, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
        )
        self.sh_degree = int(sh_degree)
        if self.sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        pe_dim = (self.sh_degree + 1) ** 2
        self.pos_proj = nn.Sequential(
            nn.Linear(pe_dim, int(pos_hidden)),
            nn.GELU(),
            nn.Linear(int(pos_hidden), hidden),
        )
        self.ae_ln = nn.LayerNorm(hidden)
        self.pos_ln = nn.LayerNorm(hidden)
        self.rho = nn.Parameter(torch.tensor(float(rho_init)))
        self.out_proj = nn.Linear(hidden, d_out)

        self.register_buffer("origin_left", torch.tensor(float(origin_left)), persistent=True)
        self.register_buffer("origin_bottom", torch.tensor(float(origin_bottom)), persistent=True)
        self.register_buffer("resolution_deg", torch.tensor(float(resolution_deg)), persistent=True)
        self.register_buffer("sh_norms", self._build_sh_norms(self.sh_degree), persistent=False)

    @staticmethod
    def _build_sh_norms(degree: int) -> torch.Tensor:
        norms = torch.zeros((degree + 1, degree + 1), dtype=torch.float32)
        for l in range(degree + 1):
            for m in range(l + 1):
                log_ratio = math.lgamma(l - m + 1) - math.lgamma(l + m + 1)
                norms[l, m] = math.sqrt(((2 * l + 1) / (4.0 * math.pi)) * math.exp(log_ratio))
        return norms

    def _spatial_xy_to_latlon(self, spatial_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_xy.ndim != 2 or spatial_xy.shape[-1] != 2:
            raise ValueError("spatial_xy must have shape [batch, 2] with x/y grid indices.")
        xy = spatial_xy.to(dtype=self.resolution_deg.dtype, device=self.resolution_deg.device)
        lon = self.origin_left + (xy[:, 0] + 0.5) * self.resolution_deg
        lat = self.origin_bottom + (xy[:, 1] + 0.5) * self.resolution_deg
        return lat, lon

    def _real_spherical_harmonics(self, lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
        dtype = lat_deg.dtype
        device = lat_deg.device
        degree = self.sh_degree
        lat = torch.deg2rad(lat_deg.to(dtype=dtype, device=device))
        lon = torch.deg2rad(lon_deg.to(dtype=dtype, device=device))
        x = torch.sin(lat).clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
        one_minus_x2 = torch.clamp(1.0 - x * x, min=0.0)

        legendre: dict[tuple[int, int], torch.Tensor] = {(0, 0): torch.ones_like(x)}
        for m in range(1, degree + 1):
            coeff = 1.0
            for k in range(1, m + 1):
                coeff *= -(2 * k - 1)
            legendre[(m, m)] = float(coeff) * one_minus_x2.pow(0.5 * m)
        for m in range(0, degree):
            legendre[(m + 1, m)] = (2 * m + 1) * x * legendre[(m, m)]
        for m in range(0, degree + 1):
            for l in range(m + 2, degree + 1):
                legendre[(l, m)] = (
                    (2 * l - 1) * x * legendre[(l - 1, m)]
                    - (l + m - 1) * legendre[(l - 2, m)]
                ) / float(l - m)

        norms = self.sh_norms.to(device=device, dtype=dtype)
        sqrt2 = math.sqrt(2.0)
        parts = []
        for l in range(degree + 1):
            parts.append(norms[l, 0] * legendre[(l, 0)])
            for m in range(1, l + 1):
                base = sqrt2 * norms[l, m] * legendre[(l, m)]
                parts.append(base * torch.cos(float(m) * lon))
                parts.append(base * torch.sin(float(m) * lon))
        return torch.stack(parts, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        spatial_xy: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
        pos_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_main = self.proj(x)
        z_res = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(x)
        if self.use_gate:
            z_base = z_main + torch.sigmoid(self.gate) * z_res
        else:
            z_base = z_main + z_res
        z_base = F.normalize(z_base, dim=-1)
        h_ae = self.ae_to_hidden(z_base)

        if pos_hidden is None:
            pos_hidden = self.encode_position(
                spatial_xy=spatial_xy,
                latlon=latlon,
                dtype=x.dtype,
                device=x.device,
            )
        else:
            pos_hidden = pos_hidden.to(dtype=x.dtype, device=x.device)
        h = self.ae_ln(h_ae) + self.rho * pos_hidden
        z = self.out_proj(h)
        return F.normalize(z, dim=-1)

    def encode_position(
        self,
        spatial_xy: torch.Tensor | None = None,
        latlon: torch.Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = self.resolution_deg.device
        if dtype is None:
            dtype = self.resolution_deg.dtype
        if latlon is not None:
            if latlon.ndim != 2 or latlon.shape[-1] != 2:
                raise ValueError("latlon must have shape [batch, 2] with latitude/longitude in degrees.")
            lat = latlon[:, 0].to(dtype=dtype, device=device)
            lon = latlon[:, 1].to(dtype=dtype, device=device)
        elif spatial_xy is not None:
            lat, lon = self._spatial_xy_to_latlon(spatial_xy.to(device=device))
            lat = lat.to(dtype=dtype)
            lon = lon.to(dtype=dtype)
        else:
            raise ValueError("encode_position requires spatial_xy or latlon.")

        pos = self._real_spherical_harmonics(lat, lon)
        h_pos = self.pos_proj(pos.to(dtype=dtype, device=device))
        return self.pos_ln(h_pos)

class TextProj(nn.Module):
    def __init__(self, d_in: int = 384, d_out: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(t), dim=-1)
