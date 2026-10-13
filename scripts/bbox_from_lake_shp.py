"""Compute lake bounding boxes from a shapefile and optionally export per-lake shapefiles.

Usage:
    python scripts/bbox_from_lake_shp.py /path/to/lakes.shp out.csv
    python scripts/bbox_from_lake_shp.py /path/to/lakes.shp out.csv \
        --out-shp-dir shplist/shp --group-field Lake_name --id-field Hylak_id
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Allow running this script directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


NAME_CANDIDATES = [
    "Hylak_id",
    "Lake_name",
    "name",
    "Name",
    "lake",
    "id",
    "ID",
    "Country",
    "Continent",
    "Grand_id",
]


def sanitize_fname(s: str | None, maxlen: int = 60) -> str:
    """Clean a string so it is safe to use as a file name."""
    if s is None or str(s).strip() == "":
        return "lake"
    s = re.sub(r"[^\w\-\.]+", "_", str(s)).strip("._")
    if not s:
        s = "lake"
    return s[:maxlen]


def to_wgs84_fix(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to WGS84 and repair invalid geometries."""
    if df.crs is None:
        raise ValueError("Input shapefile is missing a CRS.")
    if df.crs.to_epsg() != 4326:
        df = df.to_crs(4326)
    df = df[df.geometry.notnull()].copy()
    try:
        df["geometry"] = df.geometry.buffer(0)
    except Exception:
        pass
    df = df[df.is_valid].copy()
    return df


def compute_bbox(df: gpd.GeoDataFrame, group_field: str | None = None) -> pd.DataFrame:
    """Return a DataFrame of bounding boxes, optionally grouped by a field."""
    if group_field:
        g = df.dissolve(by=group_field, as_index=False, dropna=False)
        g["min_lon"] = g.bounds.minx
        g["max_lon"] = g.bounds.maxx
        g["min_lat"] = g.bounds.miny
        g["max_lat"] = g.bounds.maxy
        cols = [group_field, "min_lon", "max_lon", "min_lat", "max_lat"]
        for cand in NAME_CANDIDATES:
            if cand in g.columns and cand != group_field:
                cols.insert(1, cand)
                break
        return g[cols].sort_values(by=cols[0]).reset_index(drop=True)

    b = df.bounds
    out = pd.DataFrame(
        {
            "feature_id": df.index.astype(str),
            "min_lon": b["minx"].values,
            "max_lon": b["maxx"].values,
            "min_lat": b["miny"].values,
            "max_lat": b["maxy"].values,
        }
    )
    for cand in NAME_CANDIDATES:
        if cand in df.columns:
            out.insert(1, cand, df[cand].values)
            break
    return out


def export_shapefiles(
    df: gpd.GeoDataFrame,
    out_dir: Path,
    *,
    id_field: str | None = None,
    group_field: str | None = None,
) -> int:
    """Export one shapefile per lake/feature into *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def choose_id(row: pd.Series, idx: int) -> str:  # type: ignore[type-arg]
        if id_field and id_field in row and pd.notna(row[id_field]):
            return sanitize_fname(str(row[id_field]))
        if group_field and group_field in row and pd.notna(row[group_field]):
            return sanitize_fname(str(row[group_field]))
        for cand in NAME_CANDIDATES:
            if cand in row and pd.notna(row[cand]):
                return sanitize_fname(str(row[cand]))
        return f"feature_{idx}"

    exported = 0
    rows = df.iterrows() if group_field is None else df.dissolve(
        by=group_field, as_index=False, dropna=False
    ).iterrows()
    for i, row in rows:
        fid = choose_id(row, i)
        shp_path = out_dir / f"{fid}.shp"
        feature_df = gpd.GeoDataFrame(
            pd.DataFrame([row.drop(labels="geometry", errors="ignore")]),
            geometry=[row.geometry],
            crs="EPSG:4326",
        )
        feature_df.to_file(shp_path, driver="ESRI Shapefile")
        exported += 1
    return exported


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compute lake bounding boxes and export per-lake shapefiles."
    )
    ap.add_argument("input_shp", help="Input lake shapefile path")
    ap.add_argument("out_csv", help="Output bbox CSV path")
    ap.add_argument(
        "--group-field", default=None, help="Merge features by this field before exporting"
    )
    ap.add_argument(
        "--id-field", default=None, help="Field to use for output shapefile names"
    )
    ap.add_argument(
        "--out-shp-dir", default="shplist/shp", help="Directory for per-lake shapefiles"
    )
    args = ap.parse_args(argv)

    df = gpd.read_file(args.input_shp)
    df = to_wgs84_fix(df)

    bbox_df = compute_bbox(df, group_field=args.group_field)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    bbox_df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] BBox CSV -> {args.out_csv} ({len(bbox_df)} rows)")

    out_dir = Path(args.out_shp_dir)
    n = export_shapefiles(df, out_dir, id_field=args.id_field, group_field=args.group_field)
    print(f"[OK] Exported shapefiles -> {out_dir} ({n} files)")


if __name__ == "__main__":
    main()
