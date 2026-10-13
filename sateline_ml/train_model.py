"""Model training utilities for suspended-matter UNet and plastic spectral classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .endmember_features import (
    build_plastic_features,
    compute_background_cov,
    convolve_library_to_S2,
    load_plastic_library,
)
from .spectral_tools import S2_BANDS

DeviceLike = Union[str, torch.device]


def _mps_is_available() -> bool:
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def _device_is_available(device: torch.device) -> bool:
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        if device.index is None:
            return True
        return device.index < torch.cuda.device_count()
    if device.type == "mps":
        return _mps_is_available()
    if device.type == "cpu":
        return True
    return False


def resolve_device(requested: Optional[DeviceLike] = None) -> torch.device:
    """Pick the best available torch.device, preferring CUDA → MPS → CPU.

    If a specific device is requested but unavailable, fall back while preserving the
    requested string when valid (e.g., cuda:0).
    """

    candidates: List[torch.device] = []
    if requested is not None:
        try:
            candidates.append(torch.device(requested))
        except Exception:
            pass
    candidates.extend(torch.device(name) for name in ("cuda", "mps", "cpu"))

    for candidate in candidates:
        if _device_is_available(candidate):
            return candidate
    return torch.device("cpu")


def _ensure_nchw(x: torch.Tensor, out_channels: int, name: str) -> torch.Tensor:
    """Ensure a 4D tensor is in NCHW layout, inferring channel dim by out_channels.

    Accepts tensors that might be (N,H,W,C), (N,W,C,H) or other rare permutations
    and permutes them to (N,C,H,W). Raises if channel dim cannot be inferred.
    """
    if x.ndim != 4:
        raise ValueError(f"{name} must be 4D tensor, got shape {tuple(x.shape)}")
    n, d1, d2, d3 = x.shape
    if d1 == out_channels:
        return x
    if d3 == out_channels:
        return x.permute(0, 3, 1, 2)
    if d2 == out_channels:
        return x.permute(0, 2, 1, 3)
    if out_channels == 1 and (d2 == 1 or d3 == 1):
        if d2 == 1:
            return x.permute(0, 2, 1, 3)
        else:
            return x.permute(0, 3, 1, 2)
    raise ValueError(
        f"Cannot infer channel dimension for {name} with shape {tuple(x.shape)} and out_channels={out_channels}"
    )

# ---------------------------------------------------------------------------
# Plastic spectral model helpers
# ---------------------------------------------------------------------------


@dataclass
class PlasticModelMetadata:
    input_dim: int
    feature_names: Sequence[str]
    band_order: Sequence[str]


@dataclass
class ThresholdCalibrationResult:
    """Store per-class threshold calibration details."""

    thresholds: Dict[str, float]
    metric_name: str
    per_class_metrics: Dict[str, Dict[str, float]]
    candidate_thresholds: Sequence[float]
    class_order: Sequence[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric_name,
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            "per_class_metrics": {
                cls: {mk: float(mv) for mk, mv in metrics.items()}
                for cls, metrics in self.per_class_metrics.items()
            },
            "candidate_thresholds": [float(x) for x in self.candidate_thresholds],
            "class_order": list(self.class_order),
        }



class PlasticMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(), nn.Dropout(dropout)])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.net(x))

def build_plastic_feature_matrix(
    band_cube: np.ndarray,
    band_order: Sequence[str],
    library_csv: Path | str,
    *,
    background_mask: Optional[np.ndarray] = None,
    bg_sample: int = 200_000,
) -> Tuple[np.ndarray, Sequence[str], Dict[str, np.ndarray]]:
    """
    用“库×影像”的波段交集构建塑料特征矩阵（全图）。至少需要 3 个交集波段。
    """
    band_order = tuple(band_order) if band_order else tuple(S2_BANDS.keys())
    H, W, _ = band_cube.shape

    # 1) 加载库并卷积到 S2，然后做波段交集
    lib = load_plastic_library(library_csv)
    K_dict, sensor_band_order = convolve_library_to_S2(lib)
    available_bands = [b for b in sensor_band_order if b in band_order]
    if len(available_bands) < 3:
        raise ValueError(
            f"Not enough overlapping bands between cube and library. "
            f"Have {available_bands}, need ≥3."
        )

    idx_in_cube   = [band_order.index(b) for b in available_bands]
    idx_in_sensor = [sensor_band_order.index(b) for b in available_bands]

    # 2) 影像反射率矩阵（全图）
    X_band = band_cube.reshape(-1, band_cube.shape[2])[:, idx_in_cube].astype(np.float32)

    # 3) 背景协方差（从 water/背景里抽样）
    if background_mask is not None and np.any(background_mask):
        bg_idx = np.flatnonzero(background_mask.reshape(-1))
        if bg_idx.size > bg_sample:
            bg_idx = np.random.default_rng(0).choice(bg_idx, size=bg_sample, replace=False)
        B = X_band[bg_idx]
    else:
        B = X_band

    mu = np.nanmean(B, axis=0, keepdims=True)
    Bc = np.nan_to_num(B - mu, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (Bc.T @ Bc) / max(len(Bc) - 1, 1)
    cov = cov + (1e-4 * np.eye(cov.shape[0], dtype=np.float32))  # 正则
    try:
        Sigma_inv = np.linalg.pinv(cov).astype(np.float32)
    except Exception:
        Sigma_inv = np.eye(cov.shape[0], dtype=np.float32)

    # 4) 裁剪库到可用波段
    K_avail = {name: np.asarray(vec, dtype=np.float32)[idx_in_sensor] for name, vec in K_dict.items()}

    # 5) 构特征
    feat_df = build_plastic_features(X_band, available_bands, K_avail, sigma_inv=Sigma_inv)
    features = np.hstack([X_band, feat_df.values.astype(np.float32)])
    feature_names = [f"rho_{b}" for b in available_bands] + list(feat_df.columns)

    return features, feature_names, {
        "sensor_band_order": tuple(available_bands),
        "Sigma_inv": Sigma_inv,
        "endmembers": K_avail,
    }

def build_plastic_feature_matrix_sparse(
    band_cube: np.ndarray,
    band_order: Sequence[str],
    library_csv: Path | str,
    sel_rows: np.ndarray,
    sel_cols: np.ndarray,
    *,
    background_mask: Optional[np.ndarray] = None,
    bg_sample: int = 200_000,
) -> Tuple[np.ndarray, Sequence[str], Dict[str, np.ndarray]]:
    """
    稀疏版：只在 (sel_rows, sel_cols) 上构建特征；背景协方差仍可用 background_mask 抽样估计。
    至少需要 3 个库×影像交集波段。
    """
    band_order = tuple(band_order) if band_order else tuple(S2_BANDS.keys())
    H, W, _ = band_cube.shape

    # 1) 库到 S2 & 波段交集
    lib = load_plastic_library(library_csv)
    K_dict, sensor_band_order = convolve_library_to_S2(lib)
    available_bands = [b for b in sensor_band_order if b in band_order]
    if len(available_bands) < 3:
        raise ValueError(
            f"Not enough overlapping bands between cube and library. "
            f"Have {available_bands}, need ≥3."
        )

    idx_in_cube   = [band_order.index(b) for b in available_bands]
    idx_in_sensor = [sensor_band_order.index(b) for b in available_bands]

    # 2) 选中像素的光谱
    X_sel = band_cube[sel_rows, sel_cols][:, idx_in_cube].astype(np.float32)

    # 3) 背景协方差（抽样）
    if background_mask is not None and np.any(background_mask):
        bg_idx = np.flatnonzero(background_mask.reshape(-1))
        if bg_idx.size > 0:
            if bg_idx.size > bg_sample:
                bg_idx = np.random.default_rng(0).choice(bg_idx, size=bg_sample, replace=False)
            bg_rows = (bg_idx // W).astype(np.int64)
            bg_cols = (bg_idx %  W).astype(np.int64)
            B = band_cube[bg_rows, bg_cols][:, idx_in_cube].astype(np.float32)
        else:
            B = X_sel
    else:
        B = X_sel

    mu = np.nanmean(B, axis=0, keepdims=True)
    Bc = np.nan_to_num(B - mu, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (Bc.T @ Bc) / max(len(Bc) - 1, 1)
    cov = cov + (1e-4 * np.eye(cov.shape[0], dtype=np.float32))
    try:
        Sigma_inv = np.linalg.pinv(cov).astype(np.float32)
    except Exception:
        Sigma_inv = np.eye(cov.shape[0], dtype=np.float32)

    # 4) 裁剪库到可用波段
    K_avail = {name: np.asarray(vec, dtype=np.float32)[idx_in_sensor] for name, vec in K_dict.items()}

    # 5) 特征
    feat_df = build_plastic_features(X_sel, available_bands, K_avail, sigma_inv=Sigma_inv)
    features = np.hstack([X_sel, feat_df.values.astype(np.float32)])
    feature_names = [f"rho_{b}" for b in available_bands] + list(feat_df.columns)

    return features, feature_names, {
        "sensor_band_order": tuple(available_bands),
        "Sigma_inv": Sigma_inv,
        "endmembers": K_avail,
    }


def train_plastic_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = True,
    hidden: Sequence[int] = (128, 64),
    dropout: float = 0.1,
) -> Tuple[PlasticMLP, List[float]]:
    torch_device = resolve_device(device)
    model = PlasticMLP(features.shape[1], hidden=hidden, dropout=dropout).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    X = torch.from_numpy(features).float().to(torch_device)
    y = torch.from_numpy(labels).float().unsqueeze(1).to(torch_device)

    history: List[float] = []
    model.train()
    for epoch in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"[PlasticMLP] epoch {epoch:04d} loss={loss.item():.6f}")
    return model, history


def predict_plastic(model: PlasticMLP, features: np.ndarray, *, device: Optional[str] = None) -> np.ndarray:
    runtime_device = resolve_device(device) if device is not None else next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(features).float().to(runtime_device)
        prob = model(X).squeeze(1).cpu().numpy()
    return prob


def save_plastic_model(path: Path | str, model: PlasticMLP, metadata: PlasticModelMetadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_plastic_model(path: Path | str, *, device: Optional[str] = None) -> Tuple[PlasticMLP, PlasticModelMetadata]:
    load_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=load_device)
    metadata = PlasticModelMetadata(**checkpoint["metadata"])
    model = PlasticMLP(metadata.input_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(load_device)
    model.eval()
    return model, metadata

# ---------------------------------------------------------------------------
# Suspended-matter UNet utilities
# ---------------------------------------------------------------------------


class FloatingDataset(Dataset):
    def __init__(
        self,
        images: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        valid_masks: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        self.images = images
        self.masks = masks
        if valid_masks is None:
            valid_masks = [np.ones(image.shape[:2], dtype=np.float32) for image in images]
        self.valid_masks = valid_masks

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.images)

    def __getitem__(self, idx: int):  # type: ignore[override]
        img_np = self.images[idx]
        mask_np = self.masks[idx]
        valid_np = self.valid_masks[idx]

        if img_np.ndim != 3:
            raise ValueError(f"Expected image patch to be HxWxC, got shape {img_np.shape}")

        image = torch.from_numpy(img_np).permute(2, 0, 1).float()

        if mask_np.ndim == 2:
            mask_np = mask_np[np.newaxis, ...]
        elif mask_np.ndim == 3 and mask_np.shape[0] > 8 and mask_np.shape[-1] <= 8:
            mask_np = np.transpose(mask_np, (2, 0, 1))
        mask = torch.from_numpy(mask_np).float()

        if valid_np.ndim == 3:
            valid_np = valid_np[..., 0]
        valid = torch.from_numpy(valid_np).float()
        return image, mask, valid


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.outc = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u1 = self.up(e2)
        # Handle odd spatial sizes by padding/cropping to match skip connection
        if u1.shape[-2:] != e1.shape[-2:]:
            dh = e1.size(2) - u1.size(2)
            dw = e1.size(3) - u1.size(3)
            if dh != 0 or dw != 0:
                pad_left = max(dw // 2, 0)
                pad_right = max(dw - pad_left, 0)
                pad_top = max(dh // 2, 0)
                pad_bottom = max(dh - pad_top, 0)
                if pad_left or pad_right or pad_top or pad_bottom:
                    u1 = F.pad(u1, [pad_left, pad_right, pad_top, pad_bottom])
                # If we overshot in any direction, crop back
                if u1.size(2) > e1.size(2) or u1.size(3) > e1.size(3):
                    u1 = u1[:, :, : e1.size(2), : e1.size(3)]
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.outc(d1)


def sample_training_patches(
    features: np.ndarray,
    label_stack: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tile: int = 256,
    stride: int = 256,
    min_valid_frac: float = 0.6,
    max_patches: int = 1024,
    pos_frac: float = 0.6,
    seed: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    H, W, _ = features.shape
    rng = np.random.default_rng(seed)
    pos_idxs: List[Tuple[int, int]] = []
    neg_idxs: List[Tuple[int, int]] = []
    labels = label_stack
    for r in range(0, max(1, H - tile + 1), stride):
        for c in range(0, max(1, W - tile + 1), stride):
            v_patch = valid_mask[r : r + tile, c : c + tile]
            if v_patch.size == 0 or np.mean(v_patch) < min_valid_frac:
                continue
            m_patch = labels[:, r : r + tile, c : c + tile]
            if np.any(m_patch > 0.5):
                pos_idxs.append((r, c))
            else:
                neg_idxs.append((r, c))

    rng.shuffle(pos_idxs)
    rng.shuffle(neg_idxs)

    n_pos = min(int(max_patches * pos_frac), len(pos_idxs))
    n_neg = min(max_patches - n_pos, len(neg_idxs))
    chosen = pos_idxs[:n_pos] + neg_idxs[:n_neg]
    rng.shuffle(chosen)

    images_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    valid_list: List[np.ndarray] = []
    for r, c in chosen:
        images_list.append(features[r : r + tile, c : c + tile, :])
        masks_list.append(labels[:, r : r + tile, c : c + tile])
        valid_list.append(valid_mask[r : r + tile, c : c + tile].astype(np.float32))

    return images_list, masks_list, valid_list


def train_unet_from_patches(
    dataset: Dataset,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
    batch_size: int = 1,
    pos_weight: Optional[Union[np.ndarray, torch.Tensor, Sequence[float]]] = None,
) -> Tuple[UNet, List[float], torch.device]:
    device_obj = resolve_device(device)
    sample = dataset[0]
    channels_in = sample[0].shape[0]
    channels_out = sample[1].shape[0]
    model = UNet(in_channels=channels_in, out_channels=channels_out).to(device_obj)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    if pos_weight is None:
        masks_stack = torch.stack([dataset[i][1] for i in range(len(dataset))])  # type: ignore[index]
        flat = masks_stack.reshape(len(dataset), channels_out, -1)
        pos_counts = flat.sum(dim=2)  # (N,C)
        pos_totals = pos_counts.sum(dim=0)  # (C,)
        total_pixels = float(flat.shape[0] * flat.shape[2])
        neg_totals = total_pixels - pos_totals
        if float(pos_totals.sum()) < 1.0:
            pos_weight_array = np.ones(channels_out, dtype=np.float32)
        else:
            pos_weight_array = (neg_totals / (pos_totals + 1e-4)).clamp(min=1.0).cpu().numpy().astype(np.float32)
            pos_weight_array = np.clip(pos_weight_array, 1.0, 1e4)
    elif isinstance(pos_weight, torch.Tensor):
        pos_weight_array = pos_weight.detach().cpu().numpy().astype(np.float32)
    else:
        pos_weight_array = np.asarray(pos_weight, dtype=np.float32)

    pos_weight_array = np.atleast_1d(pos_weight_array).astype(np.float32)
    if pos_weight_array.ndim != 1:
        pos_weight_array = pos_weight_array.reshape(-1)
    if pos_weight_array.shape[0] not in (1, channels_out):
        raise ValueError(
            f"pos_weight must have length 1 or match channel count ({channels_out}), got shape {pos_weight_array.shape}"
        )
    if pos_weight_array.shape[0] == 1 and channels_out > 1:
        pos_weight_array = np.repeat(pos_weight_array, channels_out)

    pos_weight_tensor = torch.as_tensor(pos_weight_array, dtype=torch.float32, device=device_obj)
    # Use base BCEWithLogits without pos_weight and apply positive-class weights manually
    base_criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # _ensure_nchw now defined at module level

    history: List[float] = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (images, masks, valid) in enumerate(dataloader):
            images = images.to(device_obj)
            masks = masks.to(device_obj)
            valid = valid.to(device_obj).unsqueeze(1)

            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            if masks.ndim == 4 and masks.shape[1] != model.outc.out_channels and masks.shape[-1] == model.outc.out_channels:
                masks = masks.permute(0, 3, 1, 2)
            if masks.shape[1] != model.outc.out_channels:
                raise ValueError(
                    f"Mask channel dimension mismatch: expected {model.outc.out_channels}, got {masks.shape[1]}"
                )

            if valid.shape[-2:] != masks.shape[-2:]:
                valid = torch.nn.functional.interpolate(valid, size=masks.shape[-2:], mode="nearest")

            optimizer.zero_grad()
            logits = model(images)
            # Canonicalize to NCHW for both tensors
            logits = _ensure_nchw(logits, model.outc.out_channels, name="logits").contiguous()
            masks = _ensure_nchw(masks, model.outc.out_channels, name="masks").contiguous()
            # Align spatial dims if needed
            if masks.shape[-2:] != logits.shape[-2:]:
                masks = torch.nn.functional.interpolate(masks, size=logits.shape[-2:], mode="nearest")
                if valid.shape[-2:] != logits.shape[-2:]:
                    valid = torch.nn.functional.interpolate(valid, size=logits.shape[-2:], mode="nearest")
            # One-time debug
            if epoch == 0 and batch_idx == 0:
                try:
                    print(
                        f"[DEBUG] shapes images={tuple(images.shape)} logits={tuple(logits.shape)} masks={tuple(masks.shape)} valid={tuple(valid.shape)}"
                    )
                except Exception:
                    pass
            # Base per-element BCE with logits
            base_loss = base_criterion(logits, masks)
            # Build positive weighting map: weight=1 for negatives, =pos_weight for positives
            pw = pos_weight_tensor.view(1, model.outc.out_channels, 1, 1)
            weight_map = 1.0 + (pw - 1.0) * masks
            loss_map = base_loss * weight_map
            loss = (loss_map * valid).sum() / (valid.sum() * loss_map.shape[1] + 1e-6)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        history.append(epoch_loss)
        if epochs <= 10 or (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[TRAIN] device={device_obj.type} epoch={epoch + 1}/{epochs} loss={epoch_loss:.4f}")
    model.eval()
    return model, history, device_obj


def infer_unet(model: UNet, features: np.ndarray, *, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        tensor_input = torch.from_numpy(features).permute(2, 0, 1)[None, :, :, :].float().to(device)
        # Use autocast on CUDA to reduce memory pressure
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(tensor_input)
        else:
            logits = model(tensor_input)
        probs = torch.sigmoid(logits).detach().cpu().squeeze(0).numpy()
    return probs


def infer_unet_tiled(
    model: UNet,
    features: np.ndarray,
    *,
    device: torch.device,
    tile: int = 1024,
    overlap: int = 64,
    use_amp: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """Memory-efficient tiled inference with overlap blending.

    Splits the scene into overlapping tiles to control peak memory usage and
    stitches probability maps by averaging in the overlap regions.
    """
    model.eval()
    H, W, C = features.shape
    C_out = model.outc.out_channels
    out = np.zeros((C_out, H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    step = max(1, tile - overlap)
    # Preconvert numpy once per row chunk if beneficial; here we convert per-tile
    total_tiles = ((H + step - 1) // step) * ((W + step - 1) // step)
    pbar = tqdm(total=total_tiles, desc="Tiled inference", leave=False) if show_progress else None
    try:
        for r in range(0, H, step):
            r_end = min(r + tile, H)
            for c in range(0, W, step):
                c_end = min(c + tile, W)
                patch = features[r:r_end, c:c_end, :]
                tin = torch.from_numpy(patch).permute(2, 0, 1)[None].float().to(device)
                with torch.no_grad():
                    if device.type == "cuda" and use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(tin)
                    else:
                        logits = model(tin)
                    prob = torch.sigmoid(logits).detach().cpu().squeeze(0).numpy()
                out[:, r:r_end, c:c_end] += prob
                weight[r:r_end, c:c_end] += 1.0
                if pbar is not None:
                    pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    weight = np.clip(weight, 1.0, None)
    out /= weight[None, :, :]
    return out


def calibrate_probability_thresholds(
    model: UNet,
    dataset: Dataset,
    *,
    device: Optional[DeviceLike] = None,
    class_names: Optional[Sequence[str]] = None,
    candidate_thresholds: Optional[Sequence[float]] = None,
    metric: str = "f1",
    batch_size: int = 2,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> ThresholdCalibrationResult:
    """Calibrate per-class probability thresholds using a labelled dataset.

    Args:
        model: Trained UNet model to evaluate.
        dataset: Dataset providing (image, mask, valid_mask) samples.
        device: Preferred device for inference; resolved via resolve_device.
        class_names: Optional friendly names for each output channel.
        candidate_thresholds: Iterable of thresholds to evaluate. Defaults to 0.2~0.95.
        metric: Selection metric ('f1', 'precision', 'recall', 'youden', 'balanced_accuracy').
        batch_size: Inference batch size during calibration.
        max_samples: Optional cap on number of dataset samples processed.
        seed: RNG seed when subsampling.

    Returns:
        ThresholdCalibrationResult containing chosen thresholds and per-class metrics.
    """

    if len(dataset) == 0:
        raise ValueError("Calibration dataset is empty.")

    device_obj = resolve_device(device)
    model = model.to(device_obj)
    model.eval()

    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.2, 0.95, 16, dtype=np.float32)
    candidate_array = np.asarray(list(candidate_thresholds), dtype=np.float32)
    candidate_array = np.unique(candidate_array)
    candidate_array = candidate_array[(candidate_array > 0.0) & (candidate_array < 1.0)]
    if candidate_array.size == 0:
        raise ValueError("candidate_thresholds must contain values between 0 and 1.")
    candidate_array = np.sort(candidate_array)

    n_classes = model.outc.out_channels
    if class_names is None:
        class_names = tuple(f"class_{i}" for i in range(n_classes))
    if len(class_names) != n_classes:
        raise ValueError(f"class_names length ({len(class_names)}) does not match model outputs ({n_classes}).")

    subset: Dataset
    if max_samples is not None and max_samples < len(dataset):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_samples, replace=False)
        indices.sort()
        subset = Subset(dataset, indices.tolist())
    else:
        subset = dataset

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    n_thresholds = candidate_array.size
    tp = np.zeros((n_classes, n_thresholds), dtype=np.float64)
    fp = np.zeros_like(tp)
    fn = np.zeros_like(tp)
    pos_total = np.zeros(n_classes, dtype=np.float64)
    valid_total = np.zeros(n_classes, dtype=np.float64)
    prob_pos_sum = np.zeros(n_classes, dtype=np.float64)
    prob_pos_count = np.zeros(n_classes, dtype=np.float64)

    with torch.no_grad():
        for images, masks, valid in loader:
            images = images.to(device_obj)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()

            masks_cpu = masks.cpu()
            if masks_cpu.ndim == 3:
                masks_cpu = masks_cpu.unsqueeze(1)
            valid_cpu = valid.unsqueeze(1).cpu()
            valid_mask = valid_cpu > 0.5
            target_mask = masks_cpu > 0.5
            if target_mask.shape[1] < n_classes:
                pad = n_classes - target_mask.shape[1]
                target_mask = F.pad(target_mask, (0, 0, 0, 0, 0, pad))
            if valid_mask.shape[1] < n_classes:
                pad = n_classes - valid_mask.shape[1]
                valid_mask = F.pad(valid_mask, (0, 0, 0, 0, 0, pad))

            pos_mask = target_mask & valid_mask

            for cls_idx in range(n_classes):
                cls_pos = pos_mask[:, cls_idx]
                cls_valid = valid_mask[:, cls_idx]
                cls_probs = probs[:, cls_idx]

                pos_count = cls_pos.sum().item()
                valid_count = cls_valid.sum().item()
                pos_total[cls_idx] += pos_count
                valid_total[cls_idx] += valid_count

                if pos_count > 0:
                    prob_pos_sum[cls_idx] += cls_probs[cls_pos].sum().item()
                    prob_pos_count[cls_idx] += pos_count

                neg_mask = (~target_mask[:, cls_idx]) & cls_valid

                for thr_idx, thr_val in enumerate(candidate_array):
                    pred_mask = (cls_probs >= thr_val) & cls_valid
                    tp[cls_idx, thr_idx] += (pred_mask & cls_pos).sum().item()
                    fp[cls_idx, thr_idx] += (pred_mask & neg_mask).sum().item()
                    fn[cls_idx, thr_idx] += ((~pred_mask) & cls_pos).sum().item()

    # Convert to floats for metric computation
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)

    neg_total = np.clip(valid_total - pos_total, a_min=0.0, a_max=None)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    specificity = np.divide(
        np.clip(neg_total[:, None] - fp, a_min=0.0, a_max=None),
        neg_total[:, None],
        out=np.zeros_like(tp),
        where=neg_total[:, None] > 0,
    )
    balanced_accuracy = 0.5 * (recall + specificity)
    # Youden's J index = TPR + TNR - 1
    youden = recall + specificity - 1.0
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    metric_key = metric.lower()
    metric_map = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "youden": youden,
        "balanced_accuracy": balanced_accuracy,
    }
    if metric_key not in metric_map:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from {tuple(metric_map.keys())}.")
    metric_values = metric_map[metric_key]

    thresholds: Dict[str, float] = {}
    per_class_metrics: Dict[str, Dict[str, float]] = {}

    for cls_idx, cls_name in enumerate(class_names):
        support = pos_total[cls_idx]
        neg_support = neg_total[cls_idx]
        if support <= 0:
            chosen_idx = candidate_array.size // 2
        else:
            chosen_idx = int(np.argmax(metric_values[cls_idx]))
        chosen_idx = int(np.clip(chosen_idx, 0, candidate_array.size - 1))
        chosen_thr = float(candidate_array[chosen_idx])
        thresholds[cls_name] = chosen_thr

        per_class_metrics[cls_name] = {
            "threshold": chosen_thr,
            "precision": float(precision[cls_idx, chosen_idx]),
            "recall": float(recall[cls_idx, chosen_idx]),
            "f1": float(f1[cls_idx, chosen_idx]),
            "specificity": float(specificity[cls_idx, chosen_idx]),
            "balanced_accuracy": float(balanced_accuracy[cls_idx, chosen_idx]),
            "support": float(support),
            "negative_support": float(neg_support),
            "positive_fraction": float(support / valid_total[cls_idx]) if valid_total[cls_idx] > 0 else 0.0,
            "avg_positive_probability": float(
                prob_pos_sum[cls_idx] / prob_pos_count[cls_idx]
            )
            if prob_pos_count[cls_idx] > 0
            else 0.0,
        }

    return ThresholdCalibrationResult(
        thresholds=thresholds,
        metric_name=metric_key,
        per_class_metrics=per_class_metrics,
        candidate_thresholds=[float(x) for x in candidate_array],
        class_order=list(class_names),
    )


def save_calibration_result(result: ThresholdCalibrationResult, path: Path | str) -> None:
    """Persist calibration results as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)


__all__ = [
    "PlasticModelMetadata",
    "PlasticMLP",
    "build_plastic_feature_matrix",
    "train_plastic_model",
    "predict_plastic",
    "save_plastic_model",
    "load_plastic_model",
    "resolve_device",
    "FloatingDataset",
    "UNet",
    "sample_training_patches",
    "train_unet_from_patches",
    "infer_unet",
    "infer_unet_tiled",
    "ThresholdCalibrationResult",
    "calibrate_probability_thresholds",
    "save_calibration_result",
]


def _weighted_bce_map(
    logits: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    *,
    pos_weight_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Ensure NCHW and align shapes
    n_out = logits.shape[1]
    logits = _ensure_nchw(logits, n_out, name="logits").contiguous()
    masks = _ensure_nchw(masks, n_out, name="masks").contiguous()
    if masks.shape[-2:] != logits.shape[-2:]:
        masks = torch.nn.functional.interpolate(masks, size=logits.shape[-2:], mode="nearest")
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if valid.shape[-2:] != logits.shape[-2:]:
        valid = torch.nn.functional.interpolate(valid, size=logits.shape[-2:], mode="nearest")

    base_criterion = nn.BCEWithLogitsLoss(reduction="none")
    base_loss = base_criterion(logits, masks)
    if pos_weight_tensor is not None:
        pw = pos_weight_tensor.view(1, n_out, 1, 1)
        weight_map = 1.0 + (pw - 1.0) * masks
        loss_map = base_loss * weight_map
    else:
        loss_map = base_loss
    return loss_map, valid


def evaluate_unet_loss(
    model: UNet,
    dataloader: DataLoader,
    *,
    device: torch.device,
    pos_weight_tensor: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.no_grad():
        for images, masks, valid in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            valid = valid.to(device)
            logits = model(images)
            loss_map, valid_mask = _weighted_bce_map(logits, masks, valid, pos_weight_tensor=pos_weight_tensor)
            # Mean over valid pixels and channels
            denom = (valid_mask.sum() * loss_map.shape[1]).clamp_min(1e-6)
            loss = (loss_map * valid_mask).sum() / denom
            total_loss += float(loss.item())
            total_weight += 1.0
    return total_loss / max(total_weight, 1.0)


def train_unet_repeated(
    dataset: Dataset,
    *,
    repeats: int = 3,
    val_split: float = 0.2,
    seed: int = 42,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
    batch_size: int = 1,
    pos_weight: Optional[Union[np.ndarray, torch.Tensor, Sequence[float]]] = None,
    save_best_to: Optional[Path] = None,
) -> Tuple[UNet, List[float], torch.device, float, Path]:
    """Train multiple times and select the best model by validation loss.

    Returns (best_model, best_history, device, best_val_loss).
    Optionally saves the best checkpoint to save_best_to (directory or file path).
    """
    device_obj = resolve_device(device)
    normalized_pos_weight: Optional[np.ndarray]
    if pos_weight is None:
        normalized_pos_weight = None
    elif isinstance(pos_weight, torch.Tensor):
        normalized_pos_weight = pos_weight.detach().cpu().numpy()
    else:
        normalized_pos_weight = np.asarray(pos_weight, dtype=np.float32)
    n = len(dataset)  # type: ignore[arg-type]
    indices = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(n * (1.0 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    if len(val_idx) == 0:
        val_idx = indices[-max(1, n // 5):]
        train_idx = indices[:-len(val_idx)]
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    best_val = float("inf")
    best_model: Optional[UNet] = None
    best_hist: List[float] = []
    best_path: Optional[Path] = None

    # Precompute class balance on the training subset for stable weighting in eval if desired
    # (Keep evaluation unweighted to simplify selection)

    for i in range(repeats):
        torch.manual_seed(seed + i)
        # Train one model
        model_i, hist_i, dev_i = train_unet_from_patches(
            train_ds,
            epochs=epochs,
            lr=lr,
            device=device,
            batch_size=batch_size,
            pos_weight=normalized_pos_weight,
        )
        # Validate
        val_loss = evaluate_unet_loss(model_i, val_loader, device=dev_i)
        if val_loss < best_val:
            best_val = val_loss
            best_model = model_i
            best_hist = hist_i
            # Update best path placeholder (actual save happens after loop)
            best_path = None
        # Free intermediate model weights on CUDA
        if dev_i.type == "cuda":
            del model_i
            torch.cuda.empty_cache()

    assert best_model is not None

    # Save best model if requested; ensure directory semantics
    if save_best_to is not None:
        save_dir = Path(save_best_to)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / "best_unet.pth"
        torch.save(
            {
                "state_dict": best_model.state_dict(),
                "meta": {
                    "repeats": repeats,
                    "val_split": val_split,
                    "epochs": epochs,
                    "best_val_loss": best_val,
                },
            },
            save_file,
        )
        best_path = save_file
    else:
        # If not saving, create a temporary path reference
        best_path = Path("<memory>")

    best_model.eval()
    assert best_path is not None
    return best_model, best_hist, device_obj, best_val, best_path




# ===================== Temporal Conv RNN Cells =====================

class ConvRNNCell(nn.Module):
    """简单 ConvRNN（tanh）"""
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv_x = nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.conv_h.in_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        h = torch.tanh(self.conv_x(x) + self.conv_h(h))
        return h


class ConvGRUCell(nn.Module):
    """ConvGRU"""
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.reset = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.update = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.out = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.update.out_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        z = torch.sigmoid(self.update(torch.cat([x, h], dim=1)))
        r = torch.sigmoid(self.reset(torch.cat([x, h], dim=1)))
        h_tilde = torch.tanh(self.out(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class ConvLSTMCell(nn.Module):
    """ConvLSTM"""
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=bias)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if state is None:
            h = torch.zeros(x.size(0), self.hidden_channels, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, (h_new, c_new)
    
    
    
    
# ===================== Temporal Segmentation Heads =====================

class CNNTemporalHead(nn.Module):
    """卷积编码 + Conv{RNN|GRU|LSTM} 时序融合 + 1x1 分类头"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        cell_type: str = "gru",
        kernel_size: int = 3,
        return_sequence: bool = False,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        cell_type = cell_type.lower()
        if cell_type == "gru":
            self.cell = ConvGRUCell(64, hidden_channels, kernel_size=kernel_size)
            self._is_lstm = False
        elif cell_type == "lstm":
            self.cell = ConvLSTMCell(64, hidden_channels, kernel_size=kernel_size)
            self._is_lstm = True
        elif cell_type == "rnn":
            self.cell = ConvRNNCell(64, hidden_channels, kernel_size=kernel_size)
            self._is_lstm = False
        else:
            raise ValueError("cell_type must be one of {'rnn','gru','lstm'}")
        self.head = nn.Conv2d(hidden_channels, out_channels, 1)
        self.return_sequence = bool(return_sequence)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (N, T, C, H, W) 或 (T, C, H, W)（自动扩 batch）
        返回: (N, out, H, W) 或 (N, T, out, H, W) 取决于 return_sequence
        """
        if x_seq.ndim == 4:
            x_seq = x_seq.unsqueeze(0)  # -> (1,T,C,H,W)
        N, T, C, H, W = x_seq.shape
        h = None
        state = None
        outputs = []

        for t in range(T):
            xt = x_seq[:, t]  # (N,C,H,W)
            ft = self.encoder(xt)
            if self._is_lstm:
                h, state = self.cell(ft, state)
                ht = h
            else:
                h = self.cell(ft, h)
                ht = h
            logits_t = self.head(ht)
            if self.return_sequence:
                outputs.append(logits_t)

        if self.return_sequence:
            # (N,T,out,H,W)
            return torch.stack(outputs, dim=1)
        else:
            # 只返回末帧 (N,out,H,W)
            return self.head(ht)
        
        
# ===================== Temporal Dataset & Training =====================

class TemporalDataset(Dataset):
    """
    单样本：一个时序块 (T,H,W,C)，配 (T 或 1, out, H, W) 的 mask 与 (T 或 1, H, W) 的 valid。
    训练通常用 return_latest_only=True（只监督末帧）；也可全序列监督。
    """
    def __init__(
        self,
        images_seq: Sequence[np.ndarray],  # 每个元素形状 (T,H,W,C)
        masks_seq: Sequence[np.ndarray],   # (T,out,H,W) 或 (out,H,W) 或 (T,H,W) 二值/概率
        valid_seq: Optional[Sequence[np.ndarray]] = None,  # (T,H,W) 或 (H,W)
        return_latest_only: bool = True,
    ):
        assert len(images_seq) == len(masks_seq)
        self.images_seq = images_seq
        self.masks_seq = masks_seq
        self.valid_seq = valid_seq if valid_seq is not None else [np.ones(img.shape[1:3], np.float32) for img in images_seq]
        self.return_latest_only = bool(return_latest_only)

    def __len__(self) -> int:
        return len(self.images_seq)

    def __getitem__(self, idx: int):
        X = self.images_seq[idx]            # (T,H,W,C)
        Y = self.masks_seq[idx]             # (T,out,H,W) / (out,H,W) / (T,H,W)
        V = self.valid_seq[idx]             # (T,H,W) / (H,W)

        assert X.ndim == 4
        T, H, W, C = X.shape
        x_torch = torch.from_numpy(X).permute(0, 3, 1, 2).float()  # (T,C,H,W)

        # 规范化 mask
        if Y.ndim == 3:         # (out,H,W) 或 (T,H,W)
            if Y.shape[0] == T and Y.ndim == 3:
                Y = Y[:, None, :, :]         # (T,1,H,W)
            else:
                Y = Y[None, ...]             # (1,out,H,W)
        elif Y.ndim == 2:       # (H,W) -> (1,1,H,W)
            Y = Y[None, None, ...]
        y_torch = torch.from_numpy(Y).float()  # (T or 1, out, H, W)

        # 规范化 valid
        if V.ndim == 2:
            V = V[None, ...]   # (1,H,W)
        v_torch = torch.from_numpy(V).float()  # (T or 1,H,W)

        return x_torch, y_torch, v_torch


def _temporal_weighted_bce(
    logits: torch.Tensor,  # (N,out,H,W) 或 (N,T,out,H,W)
    targets: torch.Tensor, # (N,T,out,H,W) 或 (N,out,H,W)
    valid: torch.Tensor,   # (N,T,H,W) 或 (N,1,H,W)
    pos_weight: Optional[torch.Tensor] = None,
    latest_only: bool = True,
) -> torch.Tensor:
    # 对齐维度
    if logits.ndim == 4 and targets.ndim == 5:
        # 只监督末帧
        targets = targets[:, -1]
        valid = valid[:, -1] if valid.ndim == 4 else valid[:, 0]
    elif logits.ndim == 5 and targets.ndim == 4:
        # 扩成 T 维，复制监督
        T = logits.size(1)
        targets = targets.unsqueeze(1).repeat(1, T, 1, 1, 1)
        valid = valid if valid.ndim == 4 else valid.unsqueeze(1).repeat(1, T, 1, 1)

    if logits.ndim == 5:
        N, T, C, H, W = logits.shape
        logits = logits.reshape(N * T, C, H, W)
        targets = targets.reshape(N * T, C, H, W)
        valid = valid.reshape(N * T, 1, H, W)

    base = nn.BCEWithLogitsLoss(reduction="none")
    loss_map = base(logits, targets)
    if pos_weight is not None:
        pw = pos_weight.view(1, loss_map.size(1), 1, 1)
        loss_map = 1.0 + (pw - 1.0) * targets * loss_map
    # 有效像元平均
    valid = valid.unsqueeze(1).expand_as(loss_map)
    denom = (valid.sum() * loss_map.shape[1]).clamp_min(1e-6)
    loss = (loss_map * valid).sum() / denom
    return loss


def train_temporal_from_sequences(
    dataset: Dataset,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[str] = None,
    batch_size: int = 1,
    out_channels: Optional[int] = None,
    hidden_channels: int = 64,
    cell_type: str = "gru",
    latest_only: bool = True,
    pos_weight: Optional[Union[np.ndarray, torch.Tensor, Sequence[float]]] = None,
    return_sequence: bool = False,
):
    """训练 CNN-Conv{RNN|GRU|LSTM} 时序分割模型。"""
    device_obj = resolve_device(device)
    x0, y0, _ = dataset[0]
    in_ch = int(x0.shape[1])
    out_ch = int(y0.shape[1] if y0.ndim == 4 else (y0.shape[2] if y0.ndim == 5 else 1))
    if out_channels is not None:
        out_ch = int(out_channels)

    model = CNNTemporalHead(in_channels=in_ch, out_channels=out_ch,
                            hidden_channels=hidden_channels, cell_type=cell_type,
                            return_sequence=return_sequence).to(device_obj)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if pos_weight is None:
        # 估计正负权重（粗略）：随机取一批
        with torch.no_grad():
            _, yb, _ = dataset[0]
            if yb.ndim == 5:
                y_est = yb[-1]     # (out,H,W)
            else:
                y_est = yb
            pos = float((yb > 0.5).sum().item())
            tot = float(yb.numel())
            pw_np = np.ones(out_ch, dtype=np.float32) if pos == 0 else np.full(out_ch, (tot - pos) / (pos + 1e-3), dtype=np.float32)
    else:
        pw_np = np.asarray(pos_weight, dtype=np.float32).reshape(-1)
        if pw_np.size == 1 and out_ch > 1:
            pw_np = np.repeat(pw_np, out_ch)
    pos_weight_t = torch.as_tensor(pw_np, dtype=torch.float32, device=device_obj)

    history: List[float] = []
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for x_seq, y_seq, v_seq in dl:
            # x_seq: (B,T,C,H,W)
            x_seq = x_seq.to(device_obj)
            y_seq = y_seq.to(device_obj)
            v_seq = v_seq.to(device_obj)

            optimizer.zero_grad()
            logits = model(x_seq)   # (B,out,H,W) 或 (B,T,out,H,W)
            loss = _temporal_weighted_bce(logits, y_seq, v_seq, pos_weight=pos_weight_t, latest_only=latest_only)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += float(loss.item())
        history.append(ep_loss)
        if epochs <= 10 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"[TRAIN-TEMP] {cell_type} epoch={ep+1}/{epochs} loss={ep_loss:.4f}")

    model.eval()
    return model, history, device_obj


# ===================== Temporal Inference =====================

def infer_temporal_latest(
    model: CNNTemporalHead,
    features_seq: np.ndarray,   # (T,H,W,C)
    *,
    device: torch.device,
) -> np.ndarray:
    model = model.to(device).eval()
    with torch.no_grad():
        x = torch.from_numpy(features_seq).permute(0,3,1,2).unsqueeze(0).float().to(device)  # (1,T,C,H,W)
        logits = model(x)   # (1,out,H,W)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs


def infer_temporal_sequence(
    model: CNNTemporalHead,
    features_seq: np.ndarray,   # (T,H,W,C)
    *,
    device: torch.device,
) -> np.ndarray:
    """要求 model.return_sequence=True，返回 (T,out,H,W) 概率序列"""
    if not getattr(model, "return_sequence", False):
        raise ValueError("Model was not created with return_sequence=True")
    model = model.to(device).eval()
    with torch.no_grad():
        x = torch.from_numpy(features_seq).permute(0,3,1,2).unsqueeze(0).float().to(device)
        logits = model(x)   # (1,T,out,H,W)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return probs


__all__ += [
    "ConvRNNCell", "ConvGRUCell", "ConvLSTMCell",
    "CNNTemporalHead",
    "TemporalDataset",
    "train_temporal_from_sequences",
    "infer_temporal_latest",
    "infer_temporal_sequence",
]
