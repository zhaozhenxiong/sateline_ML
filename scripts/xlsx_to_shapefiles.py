"""Parse an Excel file containing lake geometries and export a combined + per-lake shapefiles.

The input Excel is expected to have at least:
    - Hylak_id
    - geometry (WKT, GeoJSON, or WKB hex)

Usage:
    python scripts/xlsx_to_shapefiles.py target_lake_info.xlsx shplist/shp
"""

from __future__ import annotations

import argparse
import binascii
import json
import os
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import from_wkb, from_wkt
from shapely.geometry import shape
from shapely.validation import explain_validity

# Allow running this script directly from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CRS_WGS84 = "EPSG:4326"


def _looks_like_wkb_hex(s: str) -> bool:
    s2 = s.strip().lower()
    return (
        len(s2) >= 4
        and all(c in "0123456789abcdef" for c in s2)
        and len(s2) % 2 == 0
    )


def _clean_text(s: str) -> str:
    s = str(s).strip().strip("'\"").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s)


def parse_geom(cell) -> tuple:
    """Try WKT, GeoJSON, WKB(hex) and return (geom, error_msg)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None, "empty"
    if not isinstance(cell, str):
        try:
            return from_wkt(cell.wkt, on_invalid="ignore"), ""
        except Exception:
            pass

    s = _clean_text(str(cell))
    if not s:
        return None, "empty"

    try:
        geom = from_wkt(s, on_invalid="ignore")
        if geom is not None:
            return geom, ""
    except Exception as e:
        wkt_err = f"WKT parse fail: {e}"
    else:
        wkt_err = "WKT invalid (None)"

    try:
        if s.startswith("{") and s.endswith("}"):
            obj = json.loads(s)
            if obj.get("type") == "Feature":
                geom = shape(obj["geometry"])
            elif obj.get("type") == "FeatureCollection" and obj["features"]:
                geom = shape(obj["features"][0]["geometry"])
            else:
                geom = shape(obj)
            if geom is not None:
                return geom, ""
    except Exception as e:
        gj_err = f"GeoJSON parse fail: {e}"
    else:
        gj_err = "GeoJSON invalid (None)"

    try:
        if _looks_like_wkb_hex(s):
            wkb = binascii.unhexlify(s)
            geom = from_wkb(wkb)
            if geom is not None:
                return geom, ""
    except Exception as e:
        wkb_err = f"WKB parse fail: {e}"
    else:
        wkb_err = "WKB invalid (None)"

    return None, f"{wkt_err}; {gj_err}; {wkb_err}"


def xlsx_to_shapefiles(in_xlsx: str | Path, out_dir: str | Path) -> gpd.GeoDataFrame:
    """Read the Excel and write combined + per-lake shapefiles."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(in_xlsx, dtype={"Hylak_id": str})
    if "geometry" not in df.columns:
        raise ValueError("Excel is missing a 'geometry' column (WKT / GeoJSON / WKB hex).")

    ok_rows, bad_rows = [], []
    for idx, row in df.iterrows():
        geom, err = parse_geom(row.get("geometry"))
        if geom is None:
            bad_rows.append(
                {
                    "row_index": idx,
                    "Hylak_id": row.get("Hylak_id"),
                    "reason": err,
                    "raw": str(row.get("geometry")),
                }
            )
            continue
        try:
            fixed = geom.buffer(0)
            if not fixed.is_valid:
                msg = explain_validity(fixed)
                bad_rows.append(
                    {
                        "row_index": idx,
                        "Hylak_id": row.get("Hylak_id"),
                        "reason": f"invalid_after_buffer0: {msg}",
                        "raw": "<ok>",
                    }
                )
            geom = fixed
        except Exception:
            pass
        ok_rows.append({"Hylak_id": str(row.get("Hylak_id")), "geometry": geom})

    if bad_rows:
        bad_csv = out_dir / "bad_rows.csv"
        pd.DataFrame(bad_rows).to_csv(bad_csv, index=False, encoding="utf-8-sig")
        print(f"[WARN] {len(bad_rows)} bad rows -> {bad_csv}")

    if not ok_rows:
        raise RuntimeError("No usable geometries; check bad_rows.csv.")

    gdf = gpd.GeoDataFrame(ok_rows, geometry="geometry", crs=CRS_WGS84)

    out_all = out_dir / "target_lakes.shp"
    gdf.to_file(out_all, driver="ESRI Shapefile", encoding="utf-8")
    print(f"[OK] Combined shapefile -> {out_all} ({len(gdf)} features)")

    for hid, sub in gdf.groupby("Hylak_id", dropna=False):
        if pd.isna(hid):
            continue
        sub_diss = sub.dissolve(by="Hylak_id", as_index=False)
        out_path = out_dir / f"{hid}.shp"
        sub_diss.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
        print(f"[OK] {hid} -> {out_path}")

    return gdf


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Parse lake geometries from an Excel file and export shapefiles."
    )
    ap.add_argument("in_xlsx", help="Input Excel file")
    ap.add_argument("out_dir", default="shplist/shp", nargs="?", help="Output directory")
    args = ap.parse_args(argv)
    xlsx_to_shapefiles(args.in_xlsx, args.out_dir)


if __name__ == "__main__":
    main()
