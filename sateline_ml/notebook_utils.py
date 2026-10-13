"""Utilities originally developed in notebooks for organising GEE outputs and building spectral libraries."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# File-system helpers for organising downloaded tiles
# ---------------------------------------------------------------------------

def is_garbage_file(name: str) -> bool:
    """Return True for AppleDouble, .DS_Store, Thumbs.db and hidden files."""
    low = name.lower()
    if name.startswith("._") or low in {".ds_store", "thumbs.db"}:
        return True
    if "__macosx" in low:
        return True
    if name.startswith("."):
        return True
    return False


def safe_rename(src: Path, new_basename: str, *, dry_run: bool = False) -> Tuple[Path, bool]:
    """Rename *src* to *new_basename* in the same directory, appending _N if needed."""
    src = Path(src)
    dst = src.with_name(new_basename)
    if src.resolve() == dst.resolve():
        return src, False

    stem, ext = dst.stem, dst.suffix
    k = 1
    while dst.exists():
        dst = dst.with_name(f"{stem}_{k}{ext}")
        k += 1

    if not dry_run:
        if not src.is_file():
            raise FileNotFoundError(f"source missing (rename): {src}")
        os.replace(src, dst)
    return dst, True


def safe_move(src: Path, dst_dir: Path, *, dry_run: bool = False) -> Path:
    """Move *src* into *dst_dir*, appending _N if the name already exists."""
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    stem, ext = dst.stem, dst.suffix
    k = 1
    while dst.exists():
        dst = dst_dir / f"{stem}_{k}{ext}"
        k += 1

    if not dry_run:
        if not src.is_file():
            raise FileNotFoundError(f"source missing (move): {src}")
        shutil.move(str(src), str(dst))
    return dst


def organize_gee_outputs(
    root_dir: str | Path,
    target_dir: str | Path,
    *,
    require_key: str = "best",
    size_limit: int | None = None,
    extensions: Iterable[str] = (".tif", ".tiff", ".img", ".jp2"),
    dry_run: bool = False,
) -> dict[str, List[Tuple[str, str]]]:
    """
    Recursively rename GEE output tiles under *root_dir* and move large ones to *target_dir*.

    Expected directory structure: {lake_id}/{month}/{stage}/{filename}.
    Renamed files follow the pattern {lake_id}_{month}_{stage}.tif.
    """
    root_dir = Path(root_dir)
    target_dir = Path(target_dir)
    extensions = tuple(ext.lower() for ext in extensions)

    renamed: List[Tuple[str, str]] = []
    unchanged: List[str] = []

    for src in root_dir.rglob("*"):
        if not src.is_file():
            continue
        if is_garbage_file(src.name):
            continue
        if src.suffix.lower() not in extensions:
            continue
        if require_key and require_key not in src.name:
            continue

        rel = src.relative_to(root_dir).as_posix()
        parts = rel.split("/")
        if len(parts) < 4:
            unchanged.append(str(src))
            continue

        lake_id, month, stage = parts[0], parts[1], parts[2]
        new_name = f"{lake_id}_{month}_{stage}.tif"
        new_path, did = safe_rename(src, new_name, dry_run=dry_run)
        if did:
            renamed.append((str(src), str(new_path)))
        else:
            unchanged.append(str(src))

    moved: List[Tuple[str, str, int]] = []
    for src in root_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in extensions:
            continue
        if require_key and require_key.lower() not in src.name.lower():
            continue
        if src.name.count("_") < 2:
            continue

        try:
            size = src.stat().st_size
        except OSError as e:
            print(f"[WARN] skipping {src}: {e}")
            continue

        if size_limit is not None and size <= size_limit:
            continue

        dst = safe_move(src, target_dir, dry_run=dry_run)
        moved.append((str(src), str(dst), size))

    return {"renamed": renamed, "unchanged": unchanged, "moved": moved}


# ---------------------------------------------------------------------------
# Spectral-library helpers
# ---------------------------------------------------------------------------

def build_spectral_library_from_csvs(
    csv_dir: str | Path,
    out_csv: str | Path,
    *,
    label_pattern: re.Pattern | str | None = None,
) -> pd.DataFrame:
    """
    Average all CSV spectra in *csv_dir* per file and merge into a single library.

    Each input CSV is expected to have a 'wavelength_nm' column followed by curve columns.
    The output column name for each file is derived from the filename unless a regex
    *label_pattern* is supplied.
    """
    csv_dir = Path(csv_dir)
    out_csv = Path(out_csv)
    label_re = re.compile(label_pattern) if isinstance(label_pattern, str) else label_pattern

    files = sorted(
        f for f in csv_dir.iterdir() if f.suffix.lower() == ".csv" and not f.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    df_merged = pd.DataFrame()
    for fa in files:
        df = pd.read_csv(fa)
        if "wavelength_nm" not in df.columns:
            raise ValueError(f"{fa} is missing 'wavelength_nm' column")

        label = fa.stem
        if label_re:
            m = label_re.search(fa.name)
            if m:
                label = m.group(1)

        df.columns = ["wavelength_nm"] + [f"{label}_curve_{i+1}" for i in range(df.shape[1] - 1)]
        df.set_index("wavelength_nm", inplace=True)
        series = df.mean(axis=1).to_frame(name=label)
        series = series.interpolate(method="linear").reset_index()

        if df_merged.empty:
            df_merged = series
        else:
            df_merged = pd.merge(df_merged, series, on="wavelength_nm", how="outer")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_csv, index=False)
    print(f"[OK] Spectral library -> {out_csv} ({len(df_merged)} rows)")
    return df_merged


def filter_and_move_tifs(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    pattern: re.Pattern | str,
    size_limit: int | None = None,
    dry_run: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Move TIFF files matching *pattern* from *source_dir* to *target_dir*."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    pat = re.compile(pattern) if isinstance(pattern, str) else pattern

    selected: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []

    for src in source_dir.rglob("*"):
        if not src.is_file():
            continue
        if is_garbage_file(src.name):
            continue
        if src.suffix.lower() not in {".tif", ".tiff"}:
            skipped.append((str(src), "not a TIFF"))
            continue

        if size_limit is not None:
            try:
                if src.stat().st_size < size_limit:
                    skipped.append((str(src), "below size limit"))
                    continue
            except OSError:
                skipped.append((str(src), "stat failed"))
                continue

        if not pat.search(src.name):
            skipped.append((str(src), "pattern mismatch"))
            continue

        dst = safe_move(src, target_dir, dry_run=dry_run)
        selected.append((str(src), str(dst)))

    return selected, skipped
