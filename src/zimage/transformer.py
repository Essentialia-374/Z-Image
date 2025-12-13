"""Z-Image Transformer."""

import os
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

_LINEAR_ATTN_DEFAULT_FEATURES = int(os.environ.get("ZIMAGE_LINEAR_ATTN_FEATURES", "128"))
_LINEAR_ATTN_SEED = int(os.environ.get("ZIMAGE_LINEAR_ATTN_SEED", "0"))
_LINEAR_ATTN_EPS = float(os.environ.get("ZIMAGE_LINEAR_ATTN_EPS", "1e-6"))

from config import (
    ADALN_EMBED_DIM,
    FREQUENCY_EMBEDDING_SIZE,
    MAX_PERIOD,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
    SEQ_MULTI_OF,
)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)

def _as_key_padding_mask(
    attn_mask: Optional[torch.Tensor],
    b: int,
    s: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert an attention mask to a (B, S) bool mask where True = valid token."""
    if attn_mask is None:
        return None

    m = attn_mask

    # Normalize common broadcasted shapes to (B, S).
    if m.dim() == 4 and m.shape[-1] == s:      # (B, 1, 1, S)
        m = m[:, 0, 0, :]
    elif m.dim() == 3 and m.shape[-1] == s:    # (B, 1, S)
        m = m[:, 0, :]
    elif m.dim() != 2:
        raise ValueError(f"Unsupported attention_mask shape: {tuple(attn_mask.shape)}")

    if m.shape != (b, s):
        raise ValueError(f"attention_mask shape {tuple(m.shape)} does not match (B,S)=({b},{s})")

    if m.dtype == torch.bool:
        return m.to(device=device)

    return (m > 0).to(device=device)


def _gaussian_orthogonal_random_matrix(
    nb_rows: int,
    nb_cols: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a (nb_rows, nb_cols) float32 orthogonal random features matrix (Performer/FAVOR+)."""
    # Generate on CPU for deterministic results across CPU/CUDA RNG.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    nb_full_blocks = nb_rows // nb_cols
    blocks = []

    for _ in range(nb_full_blocks):
        a = torch.randn((nb_cols, nb_cols), generator=gen, device="cpu", dtype=torch.float32)
        q, _ = torch.linalg.qr(a, mode="reduced")
        blocks.append(q.T)

    remaining = nb_rows - nb_full_blocks * nb_cols
    if remaining > 0:
        a = torch.randn((nb_cols, nb_cols), generator=gen, device="cpu", dtype=torch.float32)
        q, _ = torch.linalg.qr(a, mode="reduced")
        blocks.append(q.T[:remaining])

    mat = torch.cat(blocks, dim=0)

    # Row-wise scaling (chi-like).
    row_norms = torch.randn((nb_rows, nb_cols), generator=gen, device="cpu", dtype=torch.float32).norm(dim=1)
    mat = mat * row_norms[:, None]

    return mat.to(device=device)


def _favor_feature_map(
    x: torch.Tensor,
    projection: torch.Tensor,
    is_query: bool,
    eps: float,
) -> torch.Tensor:
    """FAVOR+ positive random features. x: (B,H,S,D), projection: (M,D) -> (B,H,S,M)."""
    b, h, s, d = x.shape
    m = projection.shape[0]

    x = x * (d ** -0.25)
    x_proj = torch.einsum("bhsd,md->bhsm", x, projection)

    x_sq = 0.5 * (x * x).sum(dim=-1, keepdim=True)
    x_proj = x_proj - x_sq

    # Stabilize exponentials.
    if is_query:
        x_proj = x_proj - x_proj.amax(dim=-1, keepdim=True)
    else:
        x_proj = x_proj - x_proj.amax(dim=2, keepdim=True)

    x_phi = torch.exp(x_proj) * (m ** -0.5)
    return x_phi + eps


def linear_attention_favor(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    num_features: int = _LINEAR_ATTN_DEFAULT_FEATURES,
    seed: int = _LINEAR_ATTN_SEED,
    eps: float = _LINEAR_ATTN_EPS,
    projection_cache: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FAVOR+ linear attention. Returns (B, S, H, D)."""
    b, s, h, d = q.shape
    _, _, hk, dk = k.shape
    assert dk == d, "k head_dim mismatch"
    assert v.shape == (b, s, hk, d), "v shape mismatch"

    # Expand KV heads for GQA.
    if hk != h:
        if h % hk != 0:
            raise ValueError(f"Cannot expand kv heads: n_heads={h}, n_kv_heads={hk}")
        rep = h // hk
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

    q_bhsd = q.permute(0, 2, 1, 3)
    k_bhsd = k.permute(0, 2, 1, 3)
    v_bhsd = v.permute(0, 2, 1, 3)

    valid = _as_key_padding_mask(attention_mask, b=b, s=s, device=q.device)
    valid_f = valid.to(dtype=q_bhsd.dtype).unsqueeze(1).unsqueeze(-1) if valid is not None else None

    # Use float32 for stability (exp).
    q32, k32, v32 = q_bhsd.float(), k_bhsd.float(), v_bhsd.float()

    if projection_cache is not None:
        proj = projection_cache
        if proj.device != q.device or proj.dtype != torch.float32 or proj.shape != (num_features, d):
            raise ValueError("projection_cache has wrong device/dtype/shape")
    else:
        proj = _gaussian_orthogonal_random_matrix(
            num_features,
            d,
            seed=seed + 1009 * d + 9176 * num_features,
            device=q.device,
        )

    q_phi = _favor_feature_map(q32, proj, is_query=True, eps=eps)
    k_phi = _favor_feature_map(k32, proj, is_query=False, eps=eps)

    if valid_f is not None:
        vf32 = valid_f.float()
        k_phi = k_phi * vf32
        v32 = v32 * vf32

    kv = torch.einsum("bhsm,bhsd->bhmd", k_phi, v32)
    k_sum = k_phi.sum(dim=2)

    out = torch.einsum("bhsm,bhmd->bhsd", q_phi, kv)
    denom = torch.einsum("bhsm,bhm->bhs", q_phi, k_sum).unsqueeze(-1)
    out = out / (denom + eps)

    if valid is not None:
        out = out * valid.to(dtype=out.dtype).unsqueeze(1).unsqueeze(-1)

    return out.to(dtype=q.dtype).permute(0, 2, 1, 3).contiguous()


class ZImageAttention(nn.Module):
    _attention_backend = None

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        hidden_states = linear_attention_favor(
            query, 
            key, 
            value, 
            attention_mask=attention_mask,
            num_features=128 
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        return output


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = ROPE_THETA,
        axes_dims: List[int] = ROPE_AXES_DIMS,
        axes_lens: List[int] = ROPE_AXES_LENS,
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = ROPE_THETA):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
        else:
            if self.freqs_cis[0].device != device:
                self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=ROPE_THETA,
        t_scale=1000.0,
        axes_dims=ROPE_AXES_DIMS,
        axes_lens=ROPE_AXES_LENS,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for _, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
                if cap_padding_len > 0
                else torch.zeros((cap_ori_len,), dtype=torch.bool, device=device)
            )
            # padded feature
            all_cap_feats_out.append(
                torch.cat(
                    [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                    dim=0,
                )
                if cap_padding_len > 0
                else cap_feat
            )

            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padded_pos_ids = torch.cat(
                [
                    image_ori_pos_ids,
                    self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                    .flatten(0, 2)
                    .repeat(image_padding_len, 1),
                ],
                dim=0,
            )
            all_image_pos_ids.append(image_padded_pos_ids if image_padding_len > 0 else image_ori_pos_ids)
            # pad mask
            image_pad_mask = torch.cat(
                [
                    torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                    torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_image_pad_mask.append(
                image_pad_mask
                if image_padding_len > 0
                else torch.zeros((image_ori_len,), dtype=torch.bool, device=device)
            )
            # padded feature
            image_padded_feat = torch.cat(
                [image, image[-1:].repeat(image_padding_len, 1)],
                dim=0,
            )
            all_image_out.append(image_padded_feat if image_padding_len > 0 else image)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0)
        )

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]  # same for dynamo compatibility

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        return x, {}
