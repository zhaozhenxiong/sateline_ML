"""Stacking/Blending utilities for Sentinel-2 debris detection.

Provides:
    - Tabular feature construction (bands + spectral indices + local stats)
    - Base learner specifications (LightGBM, XGBoost, etc.) with graceful fallbacks
    - Binary stacking ensemble with optional calibration
    - High-level presets that match recommended stacking combinations
    - Optional CRF and morphology post-processing hooks
"""

from __future__ import annotations
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Any

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import logging
import math
import time
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import platform
import concurrent.futures
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = TensorDataset = None  # type: ignore


try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    xgb = None

try:
    import catboost as cb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cb = None

try:
    from sklearn.naive_bayes import GaussianNB
except Exception:  # pragma: no cover
    GaussianNB = None  # type: ignore

try:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
except Exception:  # pragma: no cover
    QuadraticDiscriminantAnalysis = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "OpenCV (cv2) is required for stacking feature engineering.") from exc

try:
    import pydensecrf.densecrf as dcrf  # type: ignore
    import pydensecrf.utils as crf_utils  # type: ignore
except Exception:  # pragma: no cover
    dcrf = None  # type: ignore
    crf_utils = None  # type: ignore

try:
    from skimage.feature import greycomatrix, greycoprops  # type: ignore
except Exception:  # pragma: no cover
    greycomatrix = None  # type: ignore
    greycoprops = None  # type: ignore

LOGGER = logging.getLogger(__name__)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    raise ValueError(f"Expected 2D array, got shape {arr.shape}")


def _conv_mean_variance(arr: np.ndarray, ksize: int) -> Tuple[np.ndarray, np.ndarray]:
    kernel = np.ones((ksize, ksize), dtype=np.float32) / float(ksize * ksize)
    mean = cv2.filter2D(arr, -1, kernel, borderType=cv2.BORDER_REFLECT)
    sq = cv2.filter2D(arr * arr, -1, kernel, borderType=cv2.BORDER_REFLECT)
    var = np.clip(sq - mean * mean, a_min=0.0, a_max=None)
    std = np.sqrt(var, dtype=np.float32)
    return mean.astype(np.float32), std


def _glcm_contrast(arr: np.ndarray, distances: Sequence[int], angles: Sequence[float]) -> Optional[np.ndarray]:
    if greycomatrix is None or greycoprops is None:
        return None
    arr_uint8 = np.clip((arr - arr.min()) / (arr.max() -
                        arr.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
    try:
        glcm = greycomatrix(arr_uint8, distances=distances,
                            angles=angles, symmetric=True, normed=True)
        contrast = greycoprops(glcm, prop="contrast")
        # Average across distances/angles
        return np.nanmean(contrast, axis=(0, 1)).astype(np.float32, copy=False)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("GLCM computation failed: %s", exc)
        return None


def build_tabular_feature_stack(
    band_cube: np.ndarray,
    derived_indices: Mapping[str, np.ndarray],
    *,
    band_names: Optional[Sequence[str]] = None,
    water_mask: Optional[np.ndarray] = None,
    kernel_sizes: Sequence[int] = (3, 5),
    include_texture: bool = True,
    extra_features: Optional[Mapping[str, np.ndarray]] = None,
    flat_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Construct per-pixel tabular features (band, index, local stats, optional extras).

    Parameters
    ----------
    band_cube : np.ndarray
        Input reflectance cube with shape (H, W, C).
    derived_indices : Mapping[str, np.ndarray]
        Dictionary of pre-computed index rasters keyed by name.
    water_mask : Optional[np.ndarray]
        Boolean mask indicating valid pixels (default: all pixels valid).
    kernel_sizes : Sequence[int]
        Window sizes for local mean/std texture features.
    include_texture : bool
        Whether to compute local statistics for selected features.
    extra_features : Optional[Mapping[str, np.ndarray]]
        Additional rasters appended to the feature stack.
    flat_indices : Optional[Sequence[int]]
        Optional flattened pixel indices to sample; if provided, only these
        locations are extracted (and any indices outside the mask are dropped).

    Returns
    -------
    Tuple[np.ndarray, List[str], np.ndarray]
        Feature matrix (n_samples, n_features), feature name list, and the
        flattened pixel indices actually included (after mask filtering).
    """

    H, W, C = band_cube.shape
    mask = np.ones(
        (H, W), dtype=bool) if water_mask is None else water_mask.astype(bool)

    if flat_indices is not None:
        candidate_idx = np.asarray(flat_indices, dtype=np.int64).reshape(-1)
        if candidate_idx.size == 0:
            selected_idx = candidate_idx
        else:
            valid_mask = mask.reshape(-1)[candidate_idx]
            selected_idx = candidate_idx[valid_mask]
    else:
        selected_idx = np.flatnonzero(mask.reshape(-1))

    feature_maps: List[np.ndarray] = []
    feature_names: List[str] = []

    # Raw band reflectances (B2..)
    for i in range(C):
        feature_maps.append(band_cube[:, :, i])
        if band_names is not None and i < len(band_names):
            feature_names.append(f"rho_{band_names[i]}")
        else:
            feature_names.append(f"rho_{i}")

    # Derived indices (NDWI etc)
    for name, arr in derived_indices.items():
        feature_maps.append(arr)
        feature_names.append(name)

    # Local statistics for selected maps
    if include_texture:
        target_maps = list(
            feature_maps[-len(derived_indices):]) + feature_maps[:min(C, 4)]
        for arr, original_name in zip(target_maps, feature_names[-len(target_maps):]):
            for k in kernel_sizes:
                mean, std = _conv_mean_variance(arr, k)
                feature_maps.append(mean)
                feature_names.append(f"{original_name}_mean{k}")
                feature_maps.append(std)
                feature_names.append(f"{original_name}_std{k}")
            if k in (3, 5) and greycomatrix is not None:
                # Compute simple global GLCM contrast as a summary (not per-pixel to avoid cost)
                contrast_val = _glcm_contrast(
                    arr, distances=[1, 2], angles=[0, math.pi / 4])
                if contrast_val is not None:
                    feature_maps.append(np.full_like(
                        arr, contrast_val, dtype=np.float32))
                    feature_names.append(f"{original_name}_glcm_contrast")

    if extra_features:
        for name, arr in extra_features.items():
            feature_maps.append(arr)
            feature_names.append(name)

    stacked_cols: List[np.ndarray] = []
    flat_idx_view = selected_idx.astype(np.int64)
    for arr in feature_maps:
        values = np.asarray(arr, dtype=np.float32).reshape(-1)[flat_idx_view]
        stacked_cols.append(values.astype(np.float32, copy=False))

    stacked = np.stack(stacked_cols, axis=-1)

    # Replace NaNs/Infs
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0,
                            neginf=0.0).astype(np.float32)
    return stacked, feature_names, flat_idx_view


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[], ClassifierMixin]
    supports_proba: bool = True


# 鏀惧湪 imports 涔嬪悗


def _detect_gpu_backend() -> str:
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


GPU_BACKEND = _detect_gpu_backend()
LOGGER.info("Compute backend: %s", GPU_BACKEND)


def _make_lightgbm(**params) -> ModelSpec:
    if lgb is None:
        raise RuntimeError("LightGBM is not installed.")
    default = dict(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        verbose=-1,
        n_jobs=1,
    )
    if GPU_BACKEND in ("cuda", "mps"):  # LightGBM 4.3+ 鏀寔 mps
        default.update(device_type="gpu")
    default.update(params)
    return ModelSpec(name="lightgbm", factory=lambda: lgb.LGBMClassifier(**default))


def _make_catboost(**params) -> ModelSpec:
    if cb is None:
        raise RuntimeError("CatBoost is not installed.")
    default = dict(
        loss_function="Logloss",
        depth=6,
        learning_rate=0.05,
        verbose=False,
        allow_writing_files=False,
    )
    if GPU_BACKEND in ("cuda", "mps"):
        default.update(task_type="GPU", devices="0")
    default.update(params)
    return ModelSpec(name="catboost", factory=lambda: cb.CatBoostClassifier(**default))


def _make_xgboost_fixed(**params) -> ModelSpec:
    if xgb is None:
        raise RuntimeError("XGBoost is not installed.")
    # drop unsupported args such as predictor from user overrides
    if "predictor" in params:
        params = {k: v for k, v in params.items() if k != "predictor"}
    default = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        tree_method="hist",
        n_jobs=1,
    )
    # Use CPU path to avoid device mismatch warnings with numpy inputs.
    default.update(device="cpu")
    default.update(params)
    return ModelSpec(name="xgboost", factory=lambda: xgb.XGBClassifier(**default))


class TorchTabularCNN:
    """Lightweight 1D CNN for tabular features; runs on GPU when available."""

    def __init__(
        self,
        *,
        hidden_channels: int = 32,
        epochs: int = 6,
        batch_size: int = 2048,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        if torch is None or nn is None or DataLoader is None or TensorDataset is None:
            raise ImportError("Torch is required for TorchTabularCNN.")
        self.hidden_channels = int(hidden_channels)
        self.epochs = int(max(1, epochs))
        self.batch_size = int(max(1, batch_size))
        self.lr = float(lr)
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None

    def _build(self, in_features: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(1, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_channels, self.hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(self.hidden_channels, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchTabularCNN":
        X_f = np.asarray(X, dtype=np.float32)
        y_f = np.asarray(y, dtype=np.float32).reshape(-1)
        if X_f.shape[0] == 0:
            raise ValueError("TorchTabularCNN received empty training data.")
        self.input_dim = X_f.shape[1]
        model = self._build(self.input_dim).to(self.device)
        opt = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        ds = TensorDataset(torch.from_numpy(X_f), torch.from_numpy(y_f))
        loader = DataLoader(ds, batch_size=min(
            self.batch_size, max(8, len(ds))), shuffle=True, drop_last=False)
        model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.unsqueeze(1).to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                logits = model(xb).squeeze(1)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        self.model = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.input_dim is None:
            raise RuntimeError(
                "TorchTabularCNN must be fitted before prediction.")
        X_f = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X_f))
        loader = DataLoader(ds, batch_size=min(
            self.batch_size * 2, max(64, len(ds))), shuffle=False)
        preds: List[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.unsqueeze(1).to(self.device, non_blocking=True)
                prob = torch.sigmoid(self.model(xb).squeeze(1))
                preds.append(prob.cpu())
        return torch.cat(preds, dim=0).numpy().astype(np.float32, copy=False)


def _model_spec_registry() -> Dict[str, Callable[[], ModelSpec]]:
    registry: Dict[str, Callable[[], ModelSpec]] = {
        "lightgbm": lambda: _make_lightgbm(num_leaves=15, feature_fraction=0.7),
        "lightgbm_small": lambda: _make_lightgbm(num_leaves=7, max_depth=-1, feature_fraction=0.6),
        "xgboost": lambda: _make_xgboost_fixed(max_depth=5, colsample_bytree=0.7),
        "catboost": lambda: _make_catboost(),
        "cnn_tabular": lambda: ModelSpec(
            name="cnn_tabular",
            factory=lambda: TorchTabularCNN(
                epochs=8 if GPU_BACKEND == "cuda" else 5, device=GPU_BACKEND),
        ),
        "extratrees": lambda: ModelSpec(
            name="extratrees",
            factory=lambda: ExtraTreesClassifier(
                n_estimators=200, max_depth=None, n_jobs=-1, min_samples_leaf=2, bootstrap=False
            ),
        ),
        "randomforest": lambda: ModelSpec(
            name="randomforest",
            factory=lambda: RandomForestClassifier(
                n_estimators=300, n_jobs=-1, min_samples_leaf=2, bootstrap=True),
        ),
        "linear_svm": lambda: ModelSpec(
            name="linear_svm",
            factory=lambda: LinearSVC(
                C=1.0, class_weight="balanced", max_iter=5000),
            supports_proba=False,
        ),
        "rbf_svm": lambda: ModelSpec(
            name="rbf_svm",
            factory=lambda: SVC(C=2.0, gamma="scale",
                                probability=True, class_weight="balanced"),
        ),
        "logreg": lambda: ModelSpec(
            name="logreg",
            factory=lambda: LogisticRegression(
                penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=1000, tol=1e-4
            ),
        ),
        "ridge": lambda: ModelSpec(
            name="ridge",
            factory=lambda: RidgeClassifier(
                alpha=1.0, class_weight="balanced"),
            supports_proba=False,
        ),
        "knn": lambda: ModelSpec(
            name="knn",
            factory=lambda: KNeighborsClassifier(
                n_neighbors=9, weights="distance", n_jobs=-1),
        ),
    }
    if GaussianNB is not None:
        registry["gaussian_nb"] = lambda: ModelSpec(
            name="gaussian_nb", factory=lambda: GaussianNB())
    if QuadraticDiscriminantAnalysis is not None:
        registry["qda"] = lambda: ModelSpec(
            name="qda", factory=lambda: QuadraticDiscriminantAnalysis(reg_param=1e-2)
        )
    return registry


MODEL_SPECS = _model_spec_registry()
# ======== NEW: hyperparameter tuning helpers ========

# 1) 鍙傛暟绌洪棿锛堢鏁ｅ€欓€夛紝绋冲畾鍙潬锛涘闇€鏇寸粏鍙嚜琛屾墿灞曪級
PARAM_SPACE_BASE: Dict[str, Dict[str, List[Any]]] = {
    "lightgbm": {
        "num_leaves": [7, 15, 31, 63],
        "feature_fraction": [0.6, 0.7, 0.8, 0.9],
        "min_child_samples": [5, 10, 20, 30],
        "subsample": [0.7, 0.85, 1.0],
        "n_estimators": [200, 400, 800]
    },
    "xgboost": {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [1.0, 5.0, 10.0],
        "n_estimators": [300, 600, 1000]
    },
    "catboost": {
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5],
        "iterations": [400, 800, 1200]
    },
    "extratrees": {
        "n_estimators": [200, 300, 500],
        "min_samples_leaf": [1, 2, 4],
        "max_depth": [None, 20, 40]
    },
    "randomforest": {
        "n_estimators": [300, 600, 1000],
        "min_samples_leaf": [1, 2, 4],
        "max_depth": [None, 20, 40]
    },
    "rbf_svm": {
        "C": [0.5, 1.0, 2.0, 4.0],
        "gamma": ["scale", 0.1, 0.01]
    },
    "linear_svm": {"C": [0.5, 1.0, 2.0, 5.0]},
    "logreg": {"C": [0.1, 0.5, 1.0, 2.0, 5.0]},
    "ridge": {"alpha": [0.1, 1.0, 5.0]},
    "knn": {"n_neighbors": [3, 5, 9, 15]},
}

PARAM_SPACE_META: Dict[str, Dict[str, List[Any]]] = {
    "lightgbm": {"num_leaves": [7, 15, 31], "feature_fraction": [0.6, 0.7, 0.8], "n_estimators": [200, 400, 800]},
    "lightgbm_small": {"num_leaves": [7, 15], "feature_fraction": [0.6, 0.7, 0.8], "n_estimators": [200, 400, 800]},
    "xgboost": {"max_depth": [3, 4, 5], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [300, 600]},
    "logreg": {"C": [0.1, 0.5, 1.0, 2.0]},
    "ridge": {"alpha": [0.1, 1.0, 5.0]},
}


def _pick_scoring(name: str):
    name = (name or "").lower()
    if name in ("ap", "average_precision", "pr_auc"):
        return "average_precision"
    if name in ("roc", "roc_auc", "auc"):
        return "roc_auc"
    return "average_precision"


def _score_vec(y_true: np.ndarray, y_prob: np.ndarray, scoring: str) -> float:
    if scoring == "roc_auc":
        return float(roc_auc_score(y_true, y_prob))
    return float(average_precision_score(y_true, y_prob))


def _cv_mean_score(
    spec: "ModelSpec",
    params: Optional[Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Sequence[str]],
    n_splits: int,
    scoring: str,
    random_state: int,
) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    scores: List[float] = []
    for tr, va in skf.split(X, y):
        learner = BaseLearner(spec, param_overrides=params)
        learner.fit(X[tr], y[tr], feature_names=feature_names)
        p = learner.predict_proba(X[va], feature_names=feature_names)
        scores.append(_score_vec(y[va], p, scoring))
    return float(np.mean(scores)) if scores else float("-inf")


def _random_search_params(
    spec_name: str,
    is_meta: bool,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[Sequence[str]],
    *,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "average_precision",
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    # 鎷垮埌 spec & 绌烘ā鍨嬶紙鐢ㄦ潵鎷块粯璁よ秴鍙傦級
    spec = MODEL_SPECS[spec_name](
    ) if not is_meta else MODEL_SPECS[spec_name]()
    space = (PARAM_SPACE_META if is_meta else PARAM_SPACE_BASE).get(
        spec_name, {})
    if not space:
        if logger:
            logger.info(f"[tune] no param space for {spec_name}, skip tuning.")
        return {}
    best_params: Dict[str, Any] = {}
    best_score = float("-inf")
    keys = list(space.keys())

    for t in range(int(n_iter)):
        trial = {k: rng.choice(space[k]) for k in keys}
        s = _cv_mean_score(spec, trial, X, y, feature_names, n_splits=cv,
                           scoring=scoring, random_state=int(rng.integers(1e9)))
        if logger:
            logger.info(
                f"[tune] {spec_name} trial {t+1}/{n_iter} score={s:.4f} params={trial}")
        if s > best_score:
            best_score = s
            best_params = trial
    if logger:
        logger.info(
            f"[tune] {spec_name} best score={best_score:.4f} params={best_params}")
    return best_params
# ======== /NEW ========


class TemperatureCalibrator:
    """Simple temperature scaling calibrator for logit inputs."""

    def __init__(self) -> None:
        self.temperature_: float = 1.0

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        logits = self._logit(probs)

        def objective(temp: float) -> float:
            scaled = 1.0 / (1.0 + np.exp(-logits / temp))
            return log_loss(labels, scaled, labels=[0, 1])

        best_temp = 1.0
        best_loss = objective(best_temp)
        for candidate in np.linspace(0.5, 3.0, 20):
            loss = objective(candidate)
            if loss < best_loss:
                best_loss = loss
                best_temp = float(candidate)
        self.temperature_ = best_temp

    def transform(self, probs: np.ndarray) -> np.ndarray:
        logits = self._logit(probs)
        scaled = 1.0 / (1.0 + np.exp(-logits / self.temperature_))
        return np.clip(scaled, 0.0, 1.0)


class BaseLearner:
    def __init__(self, spec: ModelSpec, param_overrides: Optional[Dict[str, Any]] = None) -> None:
        self.spec = spec
        self.model = spec.factory()
        # 搴旂敤瓒呭弬瑕嗙洊
        if param_overrides and hasattr(self.model, "set_params"):
            try:
                self.model.set_params(**param_overrides)
            except Exception as e:
                LOGGER.warning("Failed to set params for %s: %s", spec.name, e)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.is_fitted = False
        self.feature_names_: Optional[List[str]] = None
        self.requires_feature_names = lgb is not None and isinstance(
            self.model, lgb.LGBMClassifier)

    @staticmethod
    def _default_names(n_features: int) -> List[str]:
        return [f"f{i}" for i in range(n_features)]

    def fit(self, X: np.ndarray, y: np.ndarray, *, feature_names: Optional[Sequence[str]] = None) -> None:
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.fit_transform(X)
        model = self.model
        if self.requires_feature_names:
            names = list(feature_names) if feature_names is not None else self._default_names(
                Xs.shape[1])
            self.feature_names_ = names
            X_input = pd.DataFrame(Xs, columns=names)
            model.fit(X_input, y)
        else:
            model.fit(Xs, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray, *, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("BaseLearner must be fitted before prediction.")
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler.transform(X)
        model = self.model
        if self.requires_feature_names:
            names = self.feature_names_ or self._default_names(Xs.shape[1])
            X_input = pd.DataFrame(Xs, columns=names)
            raw = model.predict_proba(X_input)
        else:
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(Xs)
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(Xs)
                preds = 1.0 / (1.0 + np.exp(-scores))
                return preds.astype(np.float32, copy=False)
            else:
                raise AttributeError(
                    f"Model {self.spec.name} lacks predict_proba/decision_function.")
        # accommodate models returning 1D probs (e.g., torch CNN)
        raw_arr = np.asarray(raw)
        if raw_arr.ndim == 1:
            preds = raw_arr
        else:
            preds = raw_arr[:, 1]
        return preds.astype(np.float32, copy=False)


class BinaryStackingEnsemble:
    def __init__(
        self,
        base_specs: Sequence[ModelSpec],
        meta_spec: ModelSpec,
        *,
        calibrator: Optional[str] = None,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.base_specs = base_specs
        self.meta_spec = meta_spec
        self.calibrator_type = calibrator
        self.n_folds = n_folds
        self.random_state = random_state

        self.base_models: List[BaseLearner] = []
        self.meta_model: Optional[ClassifierMixin] = None
        self.calibrator: Optional[object] = None

        self.feature_names: Optional[List[str]] = None
        self.precomputed_names: Optional[List[str]] = None
        self.base_output_names: Optional[List[str]] = None
        self.meta_feature_names: Optional[List[str]] = None
        self.meta_requires_feature_names = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        precomputed_features: Optional[np.ndarray] = None,
        *,
        feature_names: Optional[Sequence[str]] = None,
        precomputed_names: Optional[Sequence[str]] = None,
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("Stacking fit received zero samples.")

        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        self.precomputed_names = list(
            precomputed_names) if precomputed_names is not None else None
        self.base_output_names = [spec.name for spec in self.base_specs]

        class_counts = np.bincount(y, minlength=2)
        nonzero_counts = class_counts[class_counts > 0]
        effective_folds = int(
            min(self.n_folds, nonzero_counts.min())) if nonzero_counts.size else 0
        if effective_folds < 2:
            if nonzero_counts.size == 0:
                raise ValueError(
                    "Stacking fit received labels with no positive/negative samples.")
            LOGGER.warning(
                "Insufficient samples for stratified CV (counts=%s); training base learners without folds.",
                class_counts.tolist(),
            )
        elif effective_folds < self.n_folds:
            LOGGER.info(
                "Reducing stratified CV folds from %d to %d to respect least-populated class (counts=%s).",
                self.n_folds,
                effective_folds,
                class_counts.tolist(),
            )

        def _train_single_base(spec_idx, spec):
            oof_col = np.zeros(n_samples, dtype=np.float32)
            if effective_folds >= 2:
                skf = StratifiedKFold(
                    n_splits=effective_folds, shuffle=True, random_state=self.random_state)
                for train_idx, valid_idx in skf.split(X, y):
                    learner = BaseLearner(spec)
                    learner.fit(X[train_idx], y[train_idx],
                                feature_names=self.feature_names)
                    oof_col[valid_idx] = learner.predict_proba(
                        X[valid_idx], feature_names=self.feature_names)

        def _fit_base_oof(spec: ModelSpec, X, y, feature_names, effective_folds, random_state):
            n_samples = X.shape[0]
            oof_col = np.zeros(n_samples, dtype=np.float32)
            if effective_folds >= 2:
                skf = StratifiedKFold(
                    n_splits=effective_folds, shuffle=True, random_state=random_state)
                for train_idx, valid_idx in skf.split(X, y):
                    learner = BaseLearner(spec)
                    learner.fit(X[train_idx], y[train_idx],
                                feature_names=feature_names)
                    oof_col[valid_idx] = learner.predict_proba(
                        X[valid_idx], feature_names=feature_names)
            else:
                learner = BaseLearner(spec)
                learner.fit(X, y, feature_names=feature_names)
                oof_col[:] = learner.predict_proba(
                    X, feature_names=feature_names)
            # 鏈€缁堟暣鏁版嵁鍐嶈涓€涓叏閲忔ā鍨?            learner_final = BaseLearner(spec)
            learner_final.fit(X, y, feature_names=feature_names)
            return oof_col, learner_final

        oof_matrix = np.zeros(
            (n_samples, len(self.base_specs)), dtype=np.float32)
        fitted_models = [None] * len(self.base_specs)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(2, effective_folds), len(self.base_specs))
        ) as ex:
            futures = {
                ex.submit(_fit_base_oof, spec, X, y, self.feature_names, effective_folds, self.random_state): i
                for i, spec in enumerate(self.base_specs)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                oof_col, learner_final = fut.result()
                oof_matrix[:, idx] = oof_col
                fitted_models[idx] = learner_final

        self.base_models = fitted_models  # type: ignore

        meta_inputs = oof_matrix
        meta_feature_names: List[str] = []
        if precomputed_features is not None:
            meta_inputs = np.hstack([precomputed_features, meta_inputs])
            if self.precomputed_names is not None:
                meta_feature_names.extend(self.precomputed_names)
            else:
                meta_feature_names.extend(
                    [f"meta_{i}" for i in range(precomputed_features.shape[1])])
        meta_feature_names.extend(self.base_output_names or [
                                  f"base_{i}" for i in range(len(self.base_specs))])
        self.meta_feature_names = meta_feature_names

        meta_model = self.meta_spec.factory()
        self.meta_requires_feature_names = lgb is not None and isinstance(
            meta_model, lgb.LGBMClassifier)
        if self.meta_requires_feature_names:
            X_meta = pd.DataFrame(meta_inputs, columns=self.meta_feature_names)
            meta_model.fit(X_meta, y)
        else:
            meta_model.fit(meta_inputs, y)
        self.meta_model = meta_model

        if self.calibrator_type is not None:
            proba = self._meta_predict(meta_inputs)
            if self.calibrator_type == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(proba, y)
                self.calibrator = iso
            elif self.calibrator_type == "temperature":
                temp = TemperatureCalibrator()
                temp.fit(proba, y)
                self.calibrator = temp
            elif self.calibrator_type == "platt":
                calib = LogisticRegression(
                    penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=500, tol=1e-4
                )
                calib.fit(proba.reshape(-1, 1), y)
                self.calibrator = calib
            else:
                LOGGER.warning(
                    "Unsupported calibrator '%s'; skipping calibration.", self.calibrator_type)
                self.calibrator = None

    def _meta_predict(self, meta_inputs: np.ndarray) -> np.ndarray:
        if self.meta_model is None:
            raise RuntimeError("Meta model not fitted.")
        model = self.meta_model
        if self.meta_requires_feature_names:
            X_meta = pd.DataFrame(meta_inputs, columns=self.meta_feature_names)
            return model.predict_proba(X_meta)[:, 1]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(meta_inputs)[:, 1]
        if hasattr(model, "decision_function"):
            scores = model.decision_function(meta_inputs)
            return 1.0 / (1.0 + np.exp(-scores))
        raise AttributeError(
            f"Meta model {type(model).__name__} lacks predict_proba/decision_function.")

    def predict_proba(
        self,
        X: np.ndarray,
        *,
        precomputed_features: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        precomputed_names: Optional[Sequence[str]] = None,
    ) -> np.ndarray:

        if not self.base_models:
            raise RuntimeError("Stacking ensemble not fitted.")

        feature_names_list = list(
            feature_names) if feature_names is not None else self.feature_names
        base_preds = []
        for learner in self.base_models:
            base_preds.append(learner.predict_proba(
                X, feature_names=feature_names_list))
        base_matrix = np.vstack(base_preds).T

        if precomputed_features is not None:
            meta_inputs = np.hstack([precomputed_features, base_matrix])
            if precomputed_names is not None:
                meta_feature_names = list(
                    precomputed_names) + (self.base_output_names or [])
            elif self.precomputed_names is not None:
                meta_feature_names = list(
                    self.precomputed_names) + (self.base_output_names or [])
            else:
                meta_feature_names = [f"meta_{i}" for i in range(precomputed_features.shape[1])] + (
                    self.base_output_names or []
                )
        else:
            meta_inputs = base_matrix
            meta_feature_names = self.base_output_names or [
                f"base_{i}" for i in range(len(self.base_specs))]

        if self.meta_requires_feature_names:
            X_meta = pd.DataFrame(meta_inputs, columns=meta_feature_names)
            probs = self.meta_model.predict_proba(X_meta)[:, 1]
        else:
            probs = self._meta_predict(meta_inputs)

        if self.calibrator is not None:
            if isinstance(self.calibrator, IsotonicRegression):
                probs = self.calibrator.transform(probs)
            elif isinstance(self.calibrator, TemperatureCalibrator):
                probs = self.calibrator.transform(probs)
            elif isinstance(self.calibrator, LogisticRegression):
                probs = self.calibrator.predict_proba(
                    probs.reshape(-1, 1))[:, 1]
        return probs.astype(np.float32)


def _resolve_specs(names: Sequence[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for name in names:
        key = name.lower()
        if key not in MODEL_SPECS:
            raise KeyError(f"Unknown model spec '{name}'")
        specs.append(MODEL_SPECS[key]())
    return specs


STACKING_PRESETS: Dict[str, Dict[str, object]] = {
    "fast_tabular": {
        "base": ["lightgbm", "xgboost",],
        "meta": "lightgbm_small",
        "calibrator": "isotonic",
        "postprocess": None,
    },
    "fast_linear": {
        "base": ["logreg"],
        "meta": "logreg",
        "calibrator": None,
        "postprocess": None,
    },
    "high_recall_plastic": {
        "base": ["xgboost", "cnn_tabular", "lightgbm"],
        "meta": "lightgbm_small",
        "calibrator": "isotonic",
        "postprocess": None,
    },
    "hybrid_seg_tab": {
        "base": ["lightgbm", "logreg"],
        "meta": "logreg",
        "calibrator": None,
        "postprocess": {"crf": {"gaussian_sxy": 3, "bilateral_sxy": 49, "compat": 5.0}},
    },
    "cross_domain": {
        "base": ["catboost", "ridge", "linear_svm"],
        "meta": "logreg",
        "calibrator": "temperature",
        "postprocess": None,
    },
}


def run_stacking_pipeline(
    preset: str,
    *,
    tabular_X: np.ndarray,
    labels: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    precomputed_meta: Optional[Mapping[str, np.ndarray]] = None,
    mask_flat: Optional[Sequence[int]] = None,
    calibrator_override: Optional[str] = None,
    random_state: int = 42,
    show_progress: bool = True,
    log: Optional[logging.Logger] = None,
    progress_desc: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    cv_folds: int = 5,
    # ===== NEW: tuning config =====
    # e.g., {"enable": True, "base_iter": 25, "meta_iter": 30, "cv": 3, "scoring": "average_precision"}
    tune_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Run stacking for each target class and return fused probabilities, with optional hyperparameter tuning."""
    logger = log or logging.getLogger(__name__)

    def _info(msg: str):
        try:
            logger.info(msg)
        except Exception:
            print(msg, flush=True)

    # 杩涘害鏉?    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm as _tqdm
            total_steps = 4 + 3 * max(1, len(labels))  # +1 姝ョ粰璋冨弬
            pbar = _tqdm(total=total_steps,
                         desc=progress_desc or f"Stacking[{preset}]")
        except Exception:
            pbar = None

    def _step(msg: str, inc: int = 1):
        _info(msg)
        if pbar is not None:
            try:
                pbar.set_postfix_str(msg[:60])
                pbar.update(inc)
            except Exception:
                pass

    # 瑙ｆ瀽棰勮
    preset_key = preset.lower()
    if preset_key not in STACKING_PRESETS:
        raise KeyError(f"Unknown stacking preset '{preset}'.")
    cfg = STACKING_PRESETS[preset_key]
    base_specs = _resolve_specs(cfg["base"])  # type: ignore[arg-type]
    meta_name: str = cfg["meta"]  # type: ignore[assignment]
    meta_spec = MODEL_SPECS[meta_name]()
    calibrator = calibrator_override if calibrator_override is not None else cfg.get(
        "calibrator")

    _step(
        f"Loaded preset '{preset_key}': base={cfg['base']}, meta={meta_name}, calibrator={calibrator}")

    # 预计算特征＋输出容器
    outputs: Dict[str, np.ndarray] = {}
    precomputed = None
    precomputed_names = None
    if precomputed_meta:
        sorted_keys = sorted(precomputed_meta)
        pre_list = [precomputed_meta[name].astype(
            np.float32).reshape(-1) for name in sorted_keys]
        precomputed = np.vstack(pre_list).T
        precomputed_names = [f"meta_{name}" for name in sorted_keys]
    _step("Prepared precomputed meta features" +
          (f" (shape={None if precomputed is None else precomputed.shape})"), inc=1)

    # 璁粌绱㈠紩
    if mask_flat is None:
        mask_idx = np.arange(tabular_X.shape[0])
    else:
        mask_idx = np.asarray(mask_flat, dtype=np.int64)
    _step(f"Prepared train mask (n={mask_idx.size})", inc=1)

    X_train = tabular_X[mask_idx]
    precomputed_train = precomputed[mask_idx] if precomputed is not None else None

    # ===== NEW: 瓒呭弬璋冧紭璁剧疆 =====
    tune_enable = bool(tune_config and tune_config.get("enable", False))
    base_iter = int(tune_config.get("base_iter", 20)) if tune_config else 0
    meta_iter = int(tune_config.get("meta_iter", 20)) if tune_config else 0
    tune_cv = int(tune_config.get("cv", 3)) if tune_config else 3
    scoring = _pick_scoring(tune_config.get(
        "scoring", "average_precision") if tune_config else "average_precision")
    _step(
        f"Tuning: enable={tune_enable}, base_iter={base_iter}, meta_iter={meta_iter}, cv={tune_cv}, scoring={scoring}", inc=1)

    # ===== NEW: 瀵规瘡涓熀瀛︿範鍣ㄥ厛鍦ㄢ€滄暣浣撹缁冮泦鈥濅笂鍋氬皬 CV 闅忔満鎼滅储锛岄€夊嚭瓒呭弬 =====
    tuned_params_per_base: List[Optional[Dict[str, Any]]] = [
        None] * len(base_specs)
    if tune_enable:
        for i, spec in enumerate(base_specs):
            name = spec.name
            try:
                params = _random_search_params(
                    name, False, X_train, labels[next(
                        iter(labels))][mask_idx].astype(np.int8),
                    feature_names=feature_names, n_iter=base_iter, cv=tune_cv,
                    scoring=scoring, random_state=random_state, logger=logger
                )
                tuned_params_per_base[i] = params or None
            except Exception as e:
                _info(f"[tune] skip {name} due to error: {e}")
        _step("Base learners tuned.", inc=1)

    # ===== 璁粌姣忎釜绫诲埆 =====
    n_classes = len(labels)
    for i, (class_name, y) in enumerate(labels.items(), start=1):
        y = y.astype(np.int8)
        y_train = y[mask_idx]
        base_probs_input = probabilities.get(
            class_name) if probabilities else None
        fallback_probs: Optional[np.ndarray] = None
        if base_probs_input is not None:
            try:
                base_arr = np.asarray(
                    base_probs_input, dtype=np.float32).reshape(-1)
                if base_arr.shape[0] == tabular_X.shape[0]:
                    fallback_probs = base_arr
                else:
                    _step(
                        f"[{i}/{n_classes}] Ignoring fallback probabilities for '{class_name}' "
                        f"due to shape mismatch (got {base_arr.shape[0]}, expected {tabular_X.shape[0]})."
                    )
            except Exception as prob_exc:
                _step(
                    f"[{i}/{n_classes}] Failed to parse fallback probabilities for '{class_name}': {prob_exc}")

        _step(f"[{i}/{n_classes}] Fitting ensemble for '{class_name}' "
              f"(pos={int(y_train.sum())}, neg={int((y_train==0).sum())})")
        t0 = time.perf_counter()

        pos_count = int(np.count_nonzero(y_train == 1))
        neg_count = int(np.count_nonzero(y_train == 0))
        if pos_count == 0 or neg_count == 0:
            _step(
                f"[{i}/{n_classes}] Skipping ensemble: insufficient class diversity (pos={pos_count}, neg={neg_count})."
            )
            if fallback_probs is not None:
                outputs[class_name] = fallback_probs.copy()
                _step(
                    f"[{i}/{n_classes}] Stored fallback probabilities for '{class_name}'", inc=1)
            else:
                outputs[class_name] = np.zeros(
                    tabular_X.shape[0], dtype=np.float32)
                _step(
                    f"[{i}/{n_classes}] Stored zero fallback probabilities for '{class_name}' (no base provided).",
                    inc=1,
                )
            continue

        # 内部函数：带最优超参训练 OOF + 全量模型
        def _fit_base_oof_with_params(spec: ModelSpec, param_overrides: Optional[Dict[str, Any]]):
            # 自适应折数（保持你原逻辑）
            class_counts = np.bincount(y_train, minlength=2)
            nonzero = class_counts[class_counts > 0]
            effective_folds = int(
                min(cv_folds, nonzero.min())) if nonzero.size else 0
            if effective_folds < 2:
                learner_all = BaseLearner(
                    spec, param_overrides=param_overrides)
                learner_all.fit(X_train, y_train, feature_names=feature_names)
                oof_col = learner_all.predict_proba(
                    X_train, feature_names=feature_names)
                return oof_col, learner_all

            oof_col = np.zeros(X_train.shape[0], dtype=np.float32)
            skf = StratifiedKFold(n_splits=effective_folds,
                                  shuffle=True, random_state=random_state)
            learners_fold: List[BaseLearner] = []
            for tr, va in skf.split(X_train, y_train):
                learner = BaseLearner(spec, param_overrides=param_overrides)
                learner.fit(X_train[tr], y_train[tr],
                            feature_names=feature_names)
                oof_col[va] = learner.predict_proba(
                    X_train[va], feature_names=feature_names)
                learners_fold.append(learner)
            # 鐢ㄥ叏閲忓啀璁竴涓渶缁堟ā鍨嬩緵鎺ㄦ柇
            learner_final = BaseLearner(spec, param_overrides=param_overrides)
            learner_final.fit(X_train, y_train, feature_names=feature_names)
            return oof_col, learner_final

        # 鈥斺€?閫愪釜鍩哄涔犲櫒璁粌锛堝甫璋冨弬锛?鈥斺€?#
        oof_matrix = np.zeros(
            (X_train.shape[0], len(base_specs)), dtype=np.float32)
        fitted_models: List[Optional[BaseLearner]] = [None] * len(base_specs)
        for j, spec in enumerate(base_specs):
            oof_col, learner_final = _fit_base_oof_with_params(
                spec, tuned_params_per_base[j])
            oof_matrix[:, j] = oof_col
            fitted_models[j] = learner_final
        # 瀛樿捣鏉ヤ緵鍚庣画鍏ㄥ浘棰勬祴
        base_models = fitted_models  # type: ignore

        _step(
            f"[{i}/{n_classes}] Base learners done in {time.perf_counter()-t0:.2f}s")

        # 鈥斺€?鍏冨涔犲櫒锛堝彲閫夎皟鍙傦級 鈥斺€?#
        meta_inputs = oof_matrix if precomputed_train is None else np.hstack(
            [precomputed_train, oof_matrix])
        meta_feature_names: List[str] = []
        if precomputed_train is not None:
            meta_feature_names.extend(precomputed_names or [
                                      f"meta_{k}" for k in range(precomputed_train.shape[1])])
        meta_feature_names.extend([s.name for s in base_specs])

        meta_model = MODEL_SPECS[meta_name]().factory()
        meta_requires_feature_names = lgb is not None and isinstance(
            meta_model, lgb.LGBMClassifier)

        meta_params = {}
        if tune_enable and meta_iter > 0:
            try:
                meta_params = _random_search_params(
                    meta_name, True, meta_inputs, y_train, meta_feature_names,
                    n_iter=meta_iter, cv=tune_cv, scoring=scoring, random_state=random_state, logger=logger
                )
            except Exception as e:
                _info(f"[tune] meta search error: {e}")
        if meta_params and hasattr(meta_model, "set_params"):
            try:
                meta_model.set_params(**meta_params)
            except Exception as e:
                _info(f"[tune] meta set_params failed: {e}")

        if meta_requires_feature_names:
            X_meta = pd.DataFrame(meta_inputs, columns=meta_feature_names)
            meta_model.fit(X_meta, y_train)
        else:
            meta_model.fit(meta_inputs, y_train)

        # 鈥斺€?鏍″噯 鈥斺€?#
        calibrator_obj = None
        if calibrator is not None:
            # 用训练集 meta_inputs 取得未校准概率
            def _meta_predict_local(M, X_):
                if meta_requires_feature_names:
                    return M.predict_proba(pd.DataFrame(X_, columns=meta_feature_names))[:, 1]
                if hasattr(M, "predict_proba"):
                    return M.predict_proba(X_)[:, 1]
                if hasattr(M, "decision_function"):
                    s = M.decision_function(X_)
                    return 1.0 / (1.0 + np.exp(-s))
                raise AttributeError(
                    "Meta model lacks predict_proba/decision_function.")

            proba_train = _meta_predict_local(meta_model, meta_inputs)
            if calibrator == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(proba_train, y_train)
                calibrator_obj = iso
            elif calibrator == "temperature":
                temp = TemperatureCalibrator()
                temp.fit(proba_train, y_train)
                calibrator_obj = temp
            elif calibrator == "platt":
                pl = LogisticRegression(
                    penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=500, tol=1e-4)
                pl.fit(proba_train.reshape(-1, 1), y_train)
                calibrator_obj = pl
        # 全图推断
        # 先算基学习器的全图输出
        base_preds = []
        for learner in base_models:
            base_preds.append(learner.predict_proba(
                tabular_X, feature_names=feature_names))
        base_matrix_full = np.vstack(base_preds).T
        meta_inputs_full = base_matrix_full if precomputed is None else np.hstack(
            [precomputed, base_matrix_full])

        if meta_requires_feature_names:
            X_meta_full = pd.DataFrame(
                meta_inputs_full, columns=meta_feature_names)
            probs = meta_model.predict_proba(X_meta_full)[:, 1]
        else:
            if hasattr(meta_model, "predict_proba"):
                probs = meta_model.predict_proba(meta_inputs_full)[:, 1]
            elif hasattr(meta_model, "decision_function"):
                s = meta_model.decision_function(meta_inputs_full)
                probs = 1.0 / (1.0 + np.exp(-s))
            else:
                raise AttributeError(
                    f"Meta model {type(meta_model).__name__} lacks predict_proba/decision_function.")

        if calibrator_obj is not None:
            if isinstance(calibrator_obj, IsotonicRegression):
                probs = calibrator_obj.transform(probs)
            elif isinstance(calibrator_obj, TemperatureCalibrator):
                probs = calibrator_obj.transform(probs)
            elif isinstance(calibrator_obj, LogisticRegression):
                probs = calibrator_obj.predict_proba(
                    probs.reshape(-1, 1))[:, 1]
        _step(f"[{i}/{n_classes}] Finished, output shape={probs.shape}", inc=1)

    if pbar is not None:
        try:
            pbar.close()
        except Exception:
            pass

    return outputs


def apply_dense_crf(
    rgb_image: np.ndarray,
    prob_map: np.ndarray,
    *,
    gaussian_sxy: int = 3,
    bilateral_sxy: int = 49,
    compat: float = 5.0,
) -> np.ndarray:
    """Apply DenseCRF post-processing to a single-channel probability map."""

    if dcrf is None or crf_utils is None:
        LOGGER.warning(
            "pydensecrf not available; skipping CRF post-processing.")
        return prob_map

    H, W = prob_map.shape
    unary = np.stack([1 - prob_map, prob_map], axis=0)
    unary = -np.log(np.clip(unary, 1e-6, 1 - 1e-6))
    unary = unary.reshape(2, -1)

    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary.astype(np.float32))
    rgb_u8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)
    d.addPairwiseGaussian(sxy=gaussian_sxy, compat=compat)
    d.addPairwiseBilateral(
        sxy=bilateral_sxy,
        srgb=5,
        rgbim=rgb_u8,
        compat=compat,
    )
    Q = d.inference(5)
    refined = np.array(Q[1]).reshape(H, W).astype(np.float32)
    return refined


def postprocess_probabilities(
    fused_maps: Mapping[str, np.ndarray],
    *,
    rgb: Optional[np.ndarray] = None,
    postprocess_cfg: Optional[Mapping[str, object]] = None,
) -> Dict[str, np.ndarray]:
    """Apply optional CRF or morphology operations on fused probability maps."""

    if not postprocess_cfg:
        return dict(fused_maps)

    output = {k: v.copy() for k, v in fused_maps.items()}
    if "crf" in postprocess_cfg and rgb is not None:
        crf_cfg = postprocess_cfg["crf"] or {}
        for name, prob in output.items():
            output[name] = apply_dense_crf(
                rgb,
                prob,
                gaussian_sxy=int(crf_cfg.get("gaussian_sxy", 3)),
                bilateral_sxy=int(crf_cfg.get("bilateral_sxy", 49)),
                compat=float(crf_cfg.get("compat", 5.0)),
            )
    return output


__all__ = [
    "build_tabular_feature_stack",
    "run_stacking_pipeline",
    "postprocess_probabilities",
    "STACKING_PRESETS",
]
