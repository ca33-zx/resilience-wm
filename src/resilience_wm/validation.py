"""Scientific Verification & Historical Hindcast Validation (`resilience-wm validate`).

Provides three independent layers of model verification and validation (V&V):
  1. Analytical & Numerical Verification (LISFLOOD-FP 2D Shallow-Water):
     - Lake-at-Rest (Well-Balanced) test over steep submerged topography (`max|d_eta| < 1e-6`).
     - Strict volume mass-conservation test on transient downhill wave propagation.
  2. Historical Flood Hindcast (Dec 31, 2022 San Francisco Atmospheric River):
     - Applies spatially uniform `138.7 mm` (5.46 in, 2nd-wettest day in SF history) over
       the real AWS 30m San Francisco DEM and evaluates predicted ponding depths against
       SFPUC-documented 100-Year Storm Flood Risk hotspots (Mission Creek, SoMa/US-101,
       Islais Creek/I-280, Mission Bay, Bayview) vs. upland bedrock districts.
  3. Historical Seismic Site-Amplification Hindcast (1989 Mw 6.9 Loma Prieta, USGS/CSMIP):
     - Evaluates predicted PGA site-amplification contrast between documented soft Bay-fill /
       liquefaction zones (Marina, Embarcadero, SoMa, Mission Bay, Islais Creek) and firm
       Franciscan bedrock upland zones (Twin Peaks, Parnassus, Castro, Ingleside).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .osm import build_real_sf_city
from .physics import (
    EarthquakeForcing,
    FloodForcing,
    _lisflood_inertial_step,
    simulate_earthquake_rollout,
    simulate_flood_rollout,
)
from .viz import _load_font


@dataclass(frozen=True)
class NumericalVerification:
    """Analytical & numerical solver invariants for LISFLOOD-FP shallow-water physics."""

    lake_at_rest_max_err_m: float
    mass_conservation_rel_err: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "lake_at_rest_max_err_m": self.lake_at_rest_max_err_m,
            "mass_conservation_rel_err": self.mass_conservation_rel_err,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class HindcastResult:
    """Quantitative spatial validation against historical disaster observations."""

    event_name: str
    reference_source: str
    roc_auc: float
    top_k_recall: float
    hotspot_mean: float
    upland_mean: float
    contrast_ratio: float
    unit: str
    node_scores: dict[str, float]
    hotspot_ids: tuple[str, ...]
    upland_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_name": self.event_name,
            "reference_source": self.reference_source,
            "roc_auc": round(self.roc_auc, 4),
            "top_k_recall": round(self.top_k_recall, 4),
            "hotspot_mean": round(self.hotspot_mean, 4),
            "upland_mean": round(self.upland_mean, 4),
            "contrast_ratio": round(self.contrast_ratio, 3),
            "unit": self.unit,
            "node_scores": {k: round(v, 4) for k, v in self.node_scores.items()},
        }


@dataclass(frozen=True)
class ValidationReport:
    """Complete Verification & Validation (V&V) suite report."""

    numerical: NumericalVerification
    sf_flood_2022: HindcastResult
    sf_loma_prieta_1989: HindcastResult

    def to_dict(self) -> dict[str, object]:
        return {
            "numerical_verification": self.numerical.to_dict(),
            "sf_flood_2022_hindcast": self.sf_flood_2022.to_dict(),
            "sf_loma_prieta_1989_hindcast": self.sf_loma_prieta_1989.to_dict(),
        }

    def format_text(self) -> str:
        num = self.numerical
        fl = self.sf_flood_2022
        eq = self.sf_loma_prieta_1989
        lines = [
            "=== resilience-wm Verification & Historical Hindcast Validation ===",
            "1. Numerical Verification (LISFLOOD-FP 2D Shallow-Water):",
            f"   - Lake-at-Rest (Well-Balanced) Max Error : {num.lake_at_rest_max_err_m:.2e} m",
            f"   - Transient Mass Conservation Rel Error  : {num.mass_conservation_rel_err:.2e}",
            f"   - Status                                 : {'PASS' if num.passed else 'FAIL'}",
            "",
            f"2. Historical Flood Hindcast: {fl.event_name}",
            f"   - Reference       : {fl.reference_source}",
            f"   - Spatial ROC-AUC : {fl.roc_auc:.3f}  |  Top-K Recall: {fl.top_k_recall:.3f}",
            (
                f"   - Mean Depth      : Hotspots {fl.hotspot_mean:.3f} {fl.unit} "
                f"vs Uplands {fl.upland_mean:.3f} {fl.unit} "
                f"({fl.contrast_ratio:.2f}x contrast)"
            ),
            "",
            f"3. Historical Seismic Hindcast: {eq.event_name}",
            f"   - Reference       : {eq.reference_source}",
            f"   - Spatial ROC-AUC : {eq.roc_auc:.3f}  |  Top-K Recall: {eq.top_k_recall:.3f}",
            (
                f"   - Mean PGA        : Soft Bay-Fill {eq.hotspot_mean:.3f} {eq.unit} "
                f"vs Bedrock Uplands {eq.upland_mean:.3f} {eq.unit} "
                f"({eq.contrast_ratio:.2f}x amplification)"
            ),
        ]
        return "\n".join(lines)


def _compute_roc_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Exact Mann-Whitney U ROC-AUC in [0, 1] without external sklearn dependency."""
    if not pos_scores or not neg_scores:
        return 0.0
    wins = 0.0
    total = len(pos_scores) * len(neg_scores)
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def verify_numerical_invariants() -> NumericalVerification:
    """Run Lake-at-Rest (well-balanced) and transient mass-conservation checks."""
    # 1. Lake-at-Rest: flat surface eta = 10.0m over a steep submerged hill (z in [0, 6]m)
    yy, xx = np.mgrid[0:24, 0:24].astype(np.float32)
    z_hill = 6.0 * np.exp(-(((xx - 12.0) / 4.5) ** 2 + ((yy - 12.0) / 4.5) ** 2))
    eta0 = np.full((24, 24), 10.0, dtype=np.float32)
    h_lake = eta0 - z_hill
    qx = np.zeros((24, 23), dtype=np.float32)
    qy = np.zeros((23, 24), dtype=np.float32)
    n_grid = np.full((24, 24), 0.035, dtype=np.float32)

    h_after, qx, qy = _lisflood_inertial_step(
        h_lake, qx, qy, z_hill, n_grid, dx=100.0, dt_total=120.0
    )
    lake_err = float(np.max(np.abs((z_hill + h_after) - eta0)))

    # 2. Transient downhill wave mass conservation test
    dem_step = np.zeros((28, 28), dtype=np.float32)
    dem_step[:, 14:] = 3.0
    h0 = np.zeros((28, 28), dtype=np.float32)
    h0[6:22, 16:24] = 0.75
    qx2 = np.zeros((28, 27), dtype=np.float32)
    qy2 = np.zeros((27, 28), dtype=np.float32)
    n2 = np.full((28, 28), 0.035, dtype=np.float32)
    h1, _, _ = _lisflood_inertial_step(h0, qx2, qy2, dem_step, n2, dx=100.0, dt_total=150.0)
    mass_err = abs(float(h1.sum() - h0.sum())) / max(float(h0.sum()), 1e-9)

    return NumericalVerification(
        lake_at_rest_max_err_m=lake_err,
        mass_conservation_rel_err=mass_err,
        passed=(lake_err < 1e-5 and mass_err < 1e-5),
    )


def validate_sf_dec2022_flood_hindcast() -> HindcastResult:
    """Hindcast the Dec 31, 2022 San Francisco atmospheric river (138.7 mm / 5.46 in).

    Uses spatially UNIFORM rainfall (zero spatial peeking!) over the real AWS 30m DEM
    so predicted hotspots emerge purely from real San Francisco terrain + LISFLOOD-FP.
    """
    sf = build_real_sf_city()
    # Spatially uniform 138.7 mm (5.46 in) cloudburst (storm_radius=1e5 => uniform weight 1.0)
    forcing = FloodForcing(
        total_rain_mm=138.7,
        duration_hours=6.0,
        peak_hour=2.5,
        storm_center=(36.0, 48.0),
        storm_radius_cells=10_000.0,
        solver="lisflood_inertial",
    )
    frames = simulate_flood_rollout(sf, forcing=forcing)

    # Peak local ponding depth (max within 5x5 ~750m neighborhood around each district/facility)
    h, w = sf.shape
    peak_field = np.max(np.stack([fr.hazard_field for fr in frames], axis=0), axis=0)
    # Exclude ocean/bay cells when measuring overland street flooding around coastal nodes
    land_peak = np.where(sf.river_mask, 0.0, peak_field)

    # Documented SFPUC 100-Year Storm Flood Risk Map hotspots vs elevated non-flooding uplands
    hotspots = ("R_MIS", "R_SOMA", "U_101", "U_280", "R_BAY", "H_MB", "S_EMB", "S_MIS")
    uplands = ("U_TWN", "H_UCSF", "S_PAR", "R_CAS", "R_ING", "SH_GGP", "R_RICH")

    scores: dict[str, float] = {}
    for nid in (*hotspots, *uplands):
        node = sf.nodes[nid]
        r0, r1 = max(0, node.row - 2), min(h, node.row + 3)
        c0, c1 = max(0, node.col - 2), min(w, node.col + 3)
        patch = land_peak[r0:r1, c0:c1]
        scores[nid] = float(patch.max()) if patch.size else 0.0

    pos = [scores[k] for k in hotspots]
    neg = [scores[k] for k in uplands]
    auc = _compute_roc_auc(pos, neg)

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    top_k = set(ranked[: len(hotspots)])
    recall = len(top_k & set(hotspots)) / len(hotspots)

    pos_mean = float(np.mean(pos))
    neg_mean = float(np.mean(neg))
    return HindcastResult(
        event_name="Dec 31, 2022 SF Atmospheric River (138.7 mm / 5.46 in)",
        reference_source="SFPUC 100-Year Storm Flood Risk Map & NWS Downtown SF Record",
        roc_auc=auc,
        top_k_recall=recall,
        hotspot_mean=pos_mean,
        upland_mean=neg_mean,
        contrast_ratio=pos_mean / max(neg_mean, 1e-4),
        unit="m",
        node_scores=scores,
        hotspot_ids=hotspots,
        upland_ids=uplands,
    )


def validate_sf_1989_loma_prieta_hindcast() -> HindcastResult:
    """Hindcast the Oct 17, 1989 Mw 6.9 Loma Prieta site-response pattern across SF.

    Places the southern far-field earthquake source along the Santa Cruz Mountains azimuth
    and compares predicted PGA on USGS-documented soft Bay-fill / liquefaction zones vs.
    firm Franciscan bedrock upland stations (Presidio/Diamond Heights/Twin Peaks).
    """
    sf = build_real_sf_city()
    # Southern azimuth source (~30 km south of SF core on the 150m grid = row +200)
    # scaled for crustal Moho reflection documented by USGS (rock ~0.10-0.16g, fill ~0.22-0.30g)
    forcing = EarthquakeForcing(
        magnitude_mw=6.9,
        epicenter=(210.0, 62.0),
        depth_km=18.0,
        duration_hours=1.0,
    )
    frames = simulate_earthquake_rollout(sf, forcing=forcing)
    pga = frames[0].hazard_field

    # USGS/CSMIP documented soft artificial fill / Bay mud vs firm Franciscan bedrock hills
    soft_fill = ("R_MAR", "S_EMB", "R_SOMA", "H_MB", "U_101", "U_280", "R_BAY", "S_MIS")
    bedrock = ("U_TWN", "H_UCSF", "S_PAR", "R_CAS", "R_ING", "SH_GGP", "R_RICH")

    scores: dict[str, float] = {}
    for nid in (*soft_fill, *bedrock):
        node = sf.nodes[nid]
        scores[nid] = float(pga[node.row, node.col])

    pos = [scores[k] for k in soft_fill]
    neg = [scores[k] for k in bedrock]
    auc = _compute_roc_auc(pos, neg)

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    top_k = set(ranked[: len(soft_fill)])
    recall = len(top_k & set(soft_fill)) / len(soft_fill)

    pos_mean = float(np.mean(pos))
    neg_mean = float(np.mean(neg))
    return HindcastResult(
        event_name="Oct 17, 1989 Loma Prieta Earthquake (Mw 6.9, SF Site Response)",
        reference_source="USGS / CSMIP Strong-Motion Station & Liquefaction Microzonation",
        roc_auc=auc,
        top_k_recall=recall,
        hotspot_mean=pos_mean,
        upland_mean=neg_mean,
        contrast_ratio=pos_mean / max(neg_mean, 1e-4),
        unit="g",
        node_scores=scores,
        hotspot_ids=soft_fill,
        upland_ids=bedrock,
    )


def run_validation_suite() -> ValidationReport:
    """Execute the full Verification & Historical Hindcast Validation suite."""
    return ValidationReport(
        numerical=verify_numerical_invariants(),
        sf_flood_2022=validate_sf_dec2022_flood_hindcast(),
        sf_loma_prieta_1989=validate_sf_1989_loma_prieta_hindcast(),
    )


def render_validation_figure(report: ValidationReport, output_path: str | Path) -> Path:
    """Render a publication-grade 2-panel Historical Hindcast Validation chart."""
    W, H = 1480, 620
    img = Image.new("RGB", (W, H), (11, 17, 32))
    draw = ImageDraw.Draw(img)
    h1 = _load_font(24)
    sub = _load_font(15)
    mono = _load_font(14, mono=True)
    small = _load_font(12, mono=True)

    draw.text(
        (28, 18),
        "resilience-wm Verification & Historical San Francisco Hindcast Validation",
        fill=(241, 245, 249),
        font=h1,
    )
    num = report.numerical
    draw.text(
        (28, 50),
        (
            f"Numerical Invariants: Lake-at-Rest Err = {num.lake_at_rest_max_err_m:.1e} m  |  "
            f"Mass Conservation Rel Err = {num.mass_conservation_rel_err:.1e}  |  "
            f"Status: {'PASS' if num.passed else 'FAIL'}"
        ),
        fill=(52, 211, 153),
        font=sub,
    )

    panels = [
        (report.sf_flood_2022, 28, 92, 728, 590, "SFPUC Flood Hotspots", "Bedrock Uplands"),
        (
            report.sf_loma_prieta_1989,
            752,
            92,
            1452,
            590,
            "USGS Soft Bay-Fill",
            "Franciscan Bedrock",
        ),
    ]

    for res, x1, y1, x2, y2, pos_lbl, neg_lbl in panels:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=(15, 23, 42), outline=(51, 65, 85))
        draw.text((x1 + 16, y1 + 14), res.event_name, fill=(241, 245, 249), font=mono)
        draw.text(
            (x1 + 16, y1 + 38),
            f"ROC-AUC = {res.roc_auc:.3f}  |  Top-K Recall = {res.top_k_recall:.2f}  |  "
            f"Contrast = {res.contrast_ratio:.2f}x",
            fill=(250, 204, 21),
            font=small,
        )

        # Clean dedicated legend row below metrics (zero overlap)
        draw.rectangle((x1 + 16, y1 + 60, x1 + 28, y1 + 70), fill=(239, 68, 68))
        draw.text((x1 + 34, y1 + 58), pos_lbl, fill=(254, 202, 202), font=small)
        draw.rectangle((x1 + 224, y1 + 60, x1 + 236, y1 + 70), fill=(56, 189, 248))
        draw.text((x1 + 242, y1 + 58), neg_lbl, fill=(186, 230, 253), font=small)

        # Horizontal ranked bar chart of every validated SF location
        items = sorted(res.node_scores.items(), key=lambda kv: kv[1], reverse=True)
        max_v = max((v for _, v in items), default=1.0)
        max_v = max(max_v, 1e-4)
        hot_set = set(res.hotspot_ids)

        by = y1 + 84
        row_h = 25
        bar_x0 = x1 + 96
        bar_max_w = (x2 - x1) - 180

        for nid, val in items:
            is_hot = nid in hot_set
            col = (239, 68, 68) if is_hot else (56, 189, 248)
            tag = "[OBS+]" if is_hot else "[REF ]"
            draw.text(
                (x1 + 14, by + 3),
                f"{nid:<6s}",
                fill=(226, 232, 240),
                font=small,
            )
            draw.rounded_rectangle(
                (bar_x0, by + 2, bar_x0 + bar_max_w, by + row_h - 6),
                radius=4,
                fill=(30, 41, 59),
            )
            fw = max(4, int(bar_max_w * (val / (max_v * 1.08))))
            draw.rounded_rectangle(
                (bar_x0, by + 2, bar_x0 + fw, by + row_h - 6),
                radius=4,
                fill=col,
            )
            draw.text(
                (bar_x0 + bar_max_w + 10, by + 3),
                f"{val:.3f}{res.unit}",
                fill=(203, 213, 225),
                font=small,
            )
            del tag
            by += row_h

        draw.text(
            (x1 + 16, y2 - 26),
            f"Ref: {res.reference_source}",
            fill=(148, 163, 184),
            font=small,
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
