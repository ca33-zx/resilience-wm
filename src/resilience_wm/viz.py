"""Visual renderers: side-by-side maps, animated rollout GIFs, and interactive HTML sandbox."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .city import UrbanTwin
from .physics import HazardType, InterventionAction, WorldFrame
from .planner import PlanComparison
from .resilience import FrameMetrics


def _load_font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cands = (
        (
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
        if mono
        else (
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    for path in cands:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _render_terrain_rgb(
    city: UrbanTwin,
    frame: WorldFrame,
    *,
    hazard: HazardType,
    out_w: int,
    out_h: int,
) -> Image.Image:
    """Render vectorized dark-mode topographic base + flood/seismic heatmap into an RGB image."""
    dem = city.dem
    dem_norm = np.clip((dem - float(dem.min())) / max(float(dem.max() - dem.min()), 1e-3), 0.0, 1.0)

    # Smooth topographic hillshade (GIS-grade dark slate base)
    gy, gx = np.gradient(dem)
    shade = np.clip((-0.65 * gx - 0.65 * gy) * 1.4, -0.35, 0.35)
    r = 16.0 + 30.0 * dem_norm + 18.0 * shade
    g = 24.0 + 38.0 * dem_norm + 24.0 * shade
    b = 38.0 + 48.0 * dem_norm + 30.0 * shade

    # Winding river channel
    r = np.where(city.river_mask, 14.0, r)
    g = np.where(city.river_mask, 56.0, g)
    b = np.where(city.river_mask, 108.0, b)

    field = frame.hazard_field
    if hazard == "flood":
        # Glowing cyan -> deep royal blue -> magenta-cyan flood inundation overlay
        alpha = np.clip((field - 0.06) / 0.72, 0.0, 0.88)
        depth_t = np.clip(field / 0.85, 0.0, 1.0)
        fr = 15.0 + 95.0 * (depth_t**2)
        fg = 175.0 - 70.0 * depth_t
        fb = 255.0
        r = r * (1.0 - alpha) + fr * alpha
        g = g * (1.0 - alpha) + fg * alpha
        b = b * (1.0 - alpha) + fb * alpha
    else:
        # Seismic PGA heatmap: amber -> fiery orange -> crimson epicenter
        alpha = np.clip((field - 0.14) / 0.55, 0.0, 0.78)
        t = np.clip((field - 0.14) / 0.55, 0.0, 1.0)
        fr = 245.0
        fg = 185.0 * (1.0 - t) + 35.0 * t
        fb = 45.0 * (1.0 - t) + 55.0 * t
        r = r * (1.0 - alpha) + fr * alpha
        g = g * (1.0 - alpha) + fg * alpha
        b = b * (1.0 - alpha) + fb * alpha

    rgb = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(
        np.uint8
    )
    return Image.fromarray(rgb, mode="RGB").resize((out_w, out_h), Image.Resampling.BILINEAR)


def _draw_city_panel(
    city: UrbanTwin,
    frame: WorldFrame,
    metrics: FrameMetrics,
    *,
    hazard: HazardType,
    actions: list[InterventionAction],
    panel_w: int,
    panel_h: int,
) -> Image.Image:
    """Draw one urban twin map panel with roads, cascading power arcs, nodes, and interventions."""
    base = _render_terrain_rgb(city, frame, hazard=hazard, out_w=panel_w, out_h=panel_h).convert(
        "RGBA"
    )
    overlay = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    h, w = city.shape
    sx = panel_w / max(w - 1, 1)
    sy = panel_h / max(h - 1, 1)

    def _xy(row: float, col: float) -> tuple[int, int]:
        return int(round(col * sx)), int(round(row * sy))

    label_font = _load_font(12, mono=True)
    tag_font = _load_font(11, mono=True)

    # 1. Draw power-dependency faint lines (Substation -> Hospital)
    for node in city.nodes.values():
        if node.kind == "hospital" and node.power_source and node.power_source in city.nodes:
            sub = city.nodes[node.power_source]
            p_ok = frame.powered_substations.get(sub.node_id, True)
            p_col = (250, 204, 21, 110) if p_ok else (239, 68, 68, 130)
            draw.line([_xy(sub.row, sub.col), _xy(node.row, node.col)], fill=p_col, width=2)

    # 2. Draw road / bridge / tunnel edges
    blocked_set = set(metrics.blocked_edges)
    degraded_set = set(metrics.degraded_edges)
    action_edges = {a.target_id: a.kind for a in actions if frame.hour >= a.deploy_hour}

    for eid, edge in city.edges.items():
        p1 = _xy(city.nodes[edge.u].row, city.nodes[edge.u].col)
        p2 = _xy(city.nodes[edge.v].row, city.nodes[edge.v].col)
        width_px = 4 if edge.kind in ("bridge", "tunnel") else 3
        if eid in blocked_set:
            # Crimson severed road
            draw.line([p1, p2], fill=(239, 68, 68, 245), width=width_px + 1)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            draw.line([(mx - 5, my - 5), (mx + 5, my + 5)], fill=(255, 255, 255, 255), width=2)
            draw.line([(mx - 5, my + 5), (mx + 5, my - 5)], fill=(255, 255, 255, 255), width=2)
        elif eid in degraded_set:
            draw.line([p1, p2], fill=(245, 158, 11, 220), width=width_px)
        else:
            draw.line([p1, p2], fill=(56, 189, 248, 185), width=width_px)

        if eid in action_edges:
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            draw.ellipse(
                (mx - 10, my - 10, mx + 10, my + 10),
                fill=(250, 204, 21, 235),
                outline=(15, 23, 42, 255),
                width=2,
            )
            short = "BRG" if action_edges[eid] == "temp_bridge" else "FIX"
            draw.text((mx + 12, my - 7), short, fill=(254, 240, 138, 255), font=tag_font)

    # 3. Draw nodes
    isolated_set = set(metrics.isolated_districts)
    offline_subs = set(metrics.offline_substations)
    action_nodes = {a.target_id: a.kind for a in actions if frame.hour >= a.deploy_hour}

    for nid, node in city.nodes.items():
        x, y = _xy(node.row, node.col)
        if node.kind == "residential":
            rad = 6 + int(node.population / 28_000)
            if nid in isolated_set:
                draw.ellipse(
                    (x - rad - 5, y - rad - 5, x + rad + 5, y + rad + 5),
                    fill=(239, 68, 68, 85),
                    outline=(248, 113, 113, 255),
                    width=2,
                )
                fill = (220, 38, 38, 240)
            else:
                fill = (16, 185, 129, 225)
            draw.ellipse(
                (x - rad, y - rad, x + rad, y + rad),
                fill=fill,
                outline=(241, 245, 249, 230),
                width=1,
            )
            draw.text((x + rad + 3, y - 6), nid, fill=(226, 232, 240, 235), font=label_font)

        elif node.kind == "hospital":
            cap = metrics.hospital_capacity.get(nid, 1.0)
            bg = (
                (16, 185, 129, 245)
                if cap >= 0.8
                else ((245, 158, 11, 245) if cap > 0.05 else (220, 38, 38, 245))
            )
            draw.rounded_rectangle(
                (x - 9, y - 9, x + 9, y + 9),
                radius=3,
                fill=bg,
                outline=(255, 255, 255, 255),
                width=2,
            )
            draw.line([(x - 5, y), (x + 5, y)], fill=(255, 255, 255, 255), width=2)
            draw.line([(x, y - 5), (x, y + 5)], fill=(255, 255, 255, 255), width=2)
            draw.text(
                (x + 12, y - 7),
                f"{nid}({int(cap * 100)}%)",
                fill=(240, 253, 244, 255),
                font=tag_font,
            )

        elif node.kind == "substation":
            ok = nid not in offline_subs
            bg = (250, 204, 21, 245) if ok else (220, 38, 38, 245)
            draw.polygon(
                [(x, y - 10), (x + 10, y), (x, y + 10), (x - 10, y)],
                fill=bg,
                outline=(15, 23, 42, 255),
            )
            draw.text(
                (x + 12, y - 7),
                f"{nid}{'' if ok else '[OFF]'}",
                fill=(254, 240, 138, 255) if ok else (254, 202, 202, 255),
                font=tag_font,
            )

        elif node.kind == "shelter":
            draw.polygon(
                [(x, y - 8), (x + 8, y + 6), (x - 8, y + 6)],
                fill=(129, 140, 248, 230),
                outline=(255, 255, 255, 220),
            )
            draw.text((x + 10, y - 6), nid, fill=(199, 210, 254, 230), font=tag_font)

        else:  # underpass / bridge hub
            draw.rectangle(
                (x - 5, y - 5, x + 5, y + 5), fill=(148, 163, 184, 220), outline=(15, 23, 42, 255)
            )
            if node.kind == "underpass":
                draw.text((x + 8, y - 6), nid, fill=(203, 213, 225, 235), font=tag_font)

        # Gold ring badge on active node-level intervention
        if nid in action_nodes:
            ak = action_nodes[nid]
            short = {
                "mobile_pump": "PUMP",
                "flood_barrier": "BARRIER",
                "mobile_generator": "GEN",
            }.get(ak, "ACT")
            draw.ellipse((x - 15, y - 15, x + 15, y + 15), outline=(250, 204, 21, 255), width=3)
            tb = draw.textbbox((0, 0), short, font=tag_font)
            tw = tb[2] - tb[0]
            bx0 = x - tw // 2 - 5
            bx1 = x + tw // 2 + 5
            draw.rounded_rectangle((bx0, y - 31, bx1, y - 16), radius=3, fill=(250, 204, 21, 245))
            draw.text((bx0 + 5, y - 29), short, fill=(15, 23, 42, 255), font=tag_font)

    return Image.alpha_composite(base, overlay).convert("RGB")


def render_comparison_frame(
    city: UrbanTwin,
    comparison: PlanComparison,
    frame_idx: int,
) -> Image.Image:
    """Render one complete 1600x920 dark-mode frame comparing Baseline vs Optimal at `frame_idx`."""
    W, H = 1600, 920
    pad = 24
    gap = 24
    panel_w = (W - pad * 2 - gap) // 2  # 764
    panel_h = 500
    top_y = 112

    canvas = Image.new("RGB", (W, H), (11, 17, 32))
    draw = ImageDraw.Draw(canvas)
    h1 = _load_font(26)
    sub = _load_font(16)
    banner_f = _load_font(15, mono=True)
    mono = _load_font(14, mono=True)
    small = _load_font(13, mono=True)

    bf = comparison.baseline.frames[frame_idx]
    bm = comparison.baseline.metrics[frame_idx]
    of = comparison.optimal.frames[frame_idx]
    om = comparison.optimal.metrics[frame_idx]
    hazard = comparison.hazard

    # Top title
    title = (
        f"resilience-wm: Spatial World Model ({hazard.upper()})  |  "
        f"{city.name}  |  t = {bf.hour:.0f}h / {comparison.baseline.frames[-1].hour:.0f}h"
    )
    draw.text((pad, 16), title, fill=(241, 245, 249), font=h1)
    draw.text(
        (pad, 48),
        f"Model-Based Counterfactual Planning ({comparison.rollouts_evaluated} rollouts)  |  "
        f"Mean R(t): {comparison.baseline.mean_resilience:.2f} -> {comparison.optimal.mean_resilience:.2f} "
        f"({comparison.resilience_gain:+.2f})  |  "
        f"Peak Isolated Residents: {comparison.baseline.peak_isolated_population:,} -> "
        f"{comparison.optimal.peak_isolated_population:,}",
        fill=(148, 163, 184),
        font=sub,
    )

    rx = pad + panel_w + gap
    by = 76
    bh = 32

    # Left banner
    draw.rectangle((pad, by, pad + panel_w, by + bh), fill=(153, 27, 27))
    draw.text(
        (pad + 12, by + 7),
        f"BASELINE (No Intervention)  |  R({bf.hour:.0f}h) = {bm.resilience_index:.2f}  |  "
        f"Isolated: {bm.isolated_population:,}  |  Blocked: {len(bm.blocked_edges)}",
        fill=(254, 242, 242),
        font=banner_f,
    )

    # Right banner
    draw.rectangle((rx, by, rx + panel_w, by + bh), fill=(6, 95, 70))
    draw.text(
        (rx + 12, by + 7),
        f"WORLD-MODEL OPTIMAL PLAN  |  R({of.hour:.0f}h) = {om.resilience_index:.2f}  |  "
        f"Isolated: {om.isolated_population:,}  |  Blocked: {len(om.blocked_edges)}",
        fill=(236, 253, 245),
        font=banner_f,
    )

    # Map panels
    left_map = _draw_city_panel(
        city, bf, bm, hazard=hazard, actions=[], panel_w=panel_w, panel_h=panel_h
    )
    right_map = _draw_city_panel(
        city,
        of,
        om,
        hazard=hazard,
        actions=comparison.selected_actions,
        panel_w=panel_w,
        panel_h=panel_h,
    )
    canvas.paste(left_map, (pad, top_y))
    canvas.paste(right_map, (rx, top_y))
    draw.rectangle((pad, top_y, pad + panel_w, top_y + panel_h), outline=(239, 68, 68), width=2)
    draw.rectangle((rx, top_y, rx + panel_w, top_y + panel_h), outline=(16, 185, 129), width=2)

    # Bottom-left: Synced Resilience Triangle R(t) plot
    chart_x1, chart_y1 = pad, top_y + panel_h + 18
    chart_x2, chart_y2 = pad + panel_w, H - 22
    draw.rounded_rectangle(
        (chart_x1, chart_y1, chart_x2, chart_y2),
        radius=10,
        fill=(15, 23, 42),
        outline=(51, 65, 85),
        width=1,
    )
    draw.text(
        (chart_x1 + 16, chart_y1 + 12),
        "Urban Resilience Curve R(t) — Bruneau Resilience Triangle Gain",
        fill=(226, 232, 240),
        font=mono,
    )

    px1, py1 = chart_x1 + 56, chart_y1 + 44
    px2, py2 = chart_x2 - 26, chart_y2 - 36
    # Grid lines
    for val in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = int(py2 - val * (py2 - py1))
        draw.line([(px1, gy), (px2, gy)], fill=(30, 41, 59), width=1)
        draw.text((chart_x1 + 12, gy - 7), f"{val:.2f}", fill=(148, 163, 184), font=small)

    n_pts = len(comparison.baseline.metrics)
    b_pts = []
    o_pts = []
    for i in range(n_pts):
        tx = int(px1 + (i / max(1, n_pts - 1)) * (px2 - px1))
        by_pt = int(py2 - comparison.baseline.metrics[i].resilience_index * (py2 - py1))
        oy_pt = int(py2 - comparison.optimal.metrics[i].resilience_index * (py2 - py1))
        b_pts.append((tx, by_pt))
        o_pts.append((tx, oy_pt))
        hr_lbl = f"{int(comparison.baseline.metrics[i].hour)}h"
        draw.text((tx - 8, py2 + 8), hr_lbl, fill=(148, 163, 184), font=small)

    # Shaded green resilience gain polygon between baseline and optimal curves
    if len(b_pts) >= 2:
        poly_overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(poly_overlay)
        pdraw.polygon(o_pts + list(reversed(b_pts)), fill=(16, 185, 129, 65))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), poly_overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)

        draw.line(b_pts, fill=(239, 68, 68), width=3)
        draw.line(o_pts, fill=(16, 185, 129), width=3)

    # Current-time vertical cursor
    cx = b_pts[frame_idx][0]
    draw.line([(cx, py1), (cx, py2)], fill=(250, 204, 21), width=2)
    draw.ellipse(
        (
            b_pts[frame_idx][0] - 5,
            b_pts[frame_idx][1] - 5,
            b_pts[frame_idx][0] + 5,
            b_pts[frame_idx][1] + 5,
        ),
        fill=(239, 68, 68),
        outline=(255, 255, 255),
    )
    draw.ellipse(
        (
            o_pts[frame_idx][0] - 5,
            o_pts[frame_idx][1] - 5,
            o_pts[frame_idx][0] + 5,
            o_pts[frame_idx][1] + 5,
        ),
        fill=(16, 185, 129),
        outline=(255, 255, 255),
    )

    # Legend inside chart (cleanly spaced)
    draw.rectangle((px2 - 252, py1 + 6, px2 - 240, py1 + 16), fill=(16, 185, 129))
    draw.text((px2 - 234, py1 + 4), "Optimal Plan", fill=(167, 243, 208), font=small)
    draw.rectangle((px2 - 112, py1 + 6, px2 - 100, py1 + 16), fill=(239, 68, 68))
    draw.text((px2 - 94, py1 + 4), "Baseline", fill=(254, 202, 202), font=small)

    # Bottom-right: Counterfactual Leave-One-Out Attribution & Legend Card
    card_x1, card_y1 = rx, top_y + panel_h + 18
    card_x2, card_y2 = rx + panel_w, H - 22
    draw.rounded_rectangle(
        (card_x1, card_y1, card_x2, card_y2),
        radius=10,
        fill=(15, 23, 42),
        outline=(51, 65, 85),
        width=1,
    )
    draw.text(
        (card_x1 + 16, card_y1 + 12),
        "Counterfactual Interventions & Leave-One-Out Attribution",
        fill=(250, 204, 21),
        font=mono,
    )

    cy = card_y1 + 44
    for idx, attr in enumerate(comparison.attributions[:4], start=1):
        name = attr.action.display_name
        if len(name) > 48:
            name = name[:45] + "..."
        rescued_str = f"{max(0, attr.marginal_isolated_rescued):,}"
        line = f"{idx}. {name:<48s} dR={attr.marginal_resilience_gain:+.3f}  Rescued={rescued_str}"
        draw.text((card_x1 + 16, cy), line[:94], fill=(226, 232, 240), font=small)
        cy += 28

    # Compact map legend at bottom of card
    ly = card_y2 - 64
    draw.line([(card_x1 + 16, ly - 8), (card_x2 - 16, ly - 8)], fill=(30, 41, 59), width=1)
    draw.text(
        (card_x1 + 18, ly),
        "LEGEND:  [+] Hospital (% cap)   <> Substation   (o) District (Red=Isolated)",
        fill=(148, 163, 184),
        font=small,
    )
    draw.text(
        (card_x1 + 18, ly + 22),
        "ROADS :  Cyan=Open   Amber=Degraded   Red+X=Severed   Gold Ring=Intervention",
        fill=(148, 163, 184),
        font=small,
    )

    return canvas


def render_rollout_gif(
    city: UrbanTwin,
    comparison: PlanComparison,
    output_path: str | Path,
    *,
    frame_duration_ms: int = 850,
    scale_width: int = 1280,
) -> Path:
    """Render an animated GIF showing `0h -> 6h` Baseline vs Optimal evolution."""
    frames: list[Image.Image] = []
    n = len(comparison.baseline.frames)
    for i in range(n):
        fr = render_comparison_frame(city, comparison, i)
        if scale_width and fr.width != scale_width:
            sh = int(fr.height * scale_width / fr.width)
            fr = fr.resize((scale_width, sh), Image.Resampling.LANCZOS)
        frames.append(fr.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))

    # Hold final frame longer so viewers can read the full Resilience Triangle curve
    durations = [frame_duration_ms] * len(frames)
    if durations:
        durations[-1] = frame_duration_ms * 3

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return out


def render_hero_banner(
    city: UrbanTwin,
    comparison: PlanComparison,
    output_path: str | Path,
    *,
    frame_idx: int | None = None,
) -> Path:
    """Render a crisp peak-impact comparison PNG for README / papers / slides."""
    if frame_idx is None:
        # Pick the timestamp where Baseline vs Optimal divergence is clearest (prefers mid-rollout)
        n = len(comparison.baseline.metrics)
        idx = max(
            range(n),
            key=lambda i: (
                round(
                    comparison.optimal.metrics[i].resilience_index
                    - comparison.baseline.metrics[i].resilience_index,
                    4,
                ),
                -abs(i - n // 2),
            ),
        )
    else:
        idx = frame_idx
    img = render_comparison_frame(city, comparison, idx)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def render_interactive_sandbox_html(
    city: UrbanTwin,
    comparison: PlanComparison,
    output_path: str | Path,
) -> Path:
    """Generate a compact zero-CDN single-file interactive HTML Urban Resilience Sandbox."""
    h, w = city.shape
    # Downsample 2x and quantize to compact integers (0..255) so HTML stays < 250KB
    ds = 2 if h >= 48 and w >= 64 else 1
    dh, dw = h // ds, w // ds

    def _quantize_field(arr: np.ndarray) -> list[list[int]]:
        sub = arr[: dh * ds : ds, : dw * ds : ds]
        return np.clip(np.round(sub * 100.0), 0, 255).astype(int).tolist()

    def _serialize_eval(ev) -> dict[str, object]:
        return {
            "mean_resilience": round(ev.mean_resilience, 4),
            "peak_isolated": ev.peak_isolated_population,
            "frames": [
                {
                    "hour": fr.hour,
                    "field": _quantize_field(fr.hazard_field),
                    "metrics": m.to_dict(),
                }
                for fr, m in zip(ev.frames, ev.metrics, strict=True)
            ],
        }

    data_blob = {
        "city_name": city.name,
        "hazard": comparison.hazard,
        "rows": dh,
        "cols": dw,
        "orig_rows": h,
        "orig_cols": w,
        "dem": np.round(city.dem[: dh * ds : ds, : dw * ds : ds], 1).tolist(),
        "river": city.river_mask[: dh * ds : ds, : dw * ds : ds].astype(int).tolist(),
        "nodes": [
            {
                "id": n.node_id,
                "name": n.name,
                "kind": n.kind,
                "row": n.row,
                "col": n.col,
                "pop": n.population,
                "power_source": n.power_source,
            }
            for n in city.nodes.values()
        ],
        "edges": [
            {"id": e.edge_id, "u": e.u, "v": e.v, "kind": e.kind} for e in city.edges.values()
        ],
        "actions": [
            {
                "kind": a.kind,
                "target_id": a.target_id,
                "display_name": a.display_name,
                "delta_r": round(attr.marginal_resilience_gain, 4),
                "rescued": attr.marginal_isolated_rescued,
            }
            for a, attr in zip(comparison.selected_actions, comparison.attributions, strict=True)
        ],
        "baseline": _serialize_eval(comparison.baseline),
        "optimal": _serialize_eval(comparison.optimal),
    }

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>resilience-wm — Interactive Urban Resilience World-Model Sandbox</title>
<style>
  :root {{ --bg:#0b1120; --panel:#111827; --border:#1f2937; --fg:#e2e8f0; --muted:#94a3b8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:var(--bg); color:var(--fg); padding:20px; }}
  .wrap {{ max-width:1360px; margin:0 auto; }}
  header {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:14px;
            background:var(--panel); border:1px solid var(--border); border-radius:12px;
            padding:16px 22px; margin-bottom:16px; }}
  h1 {{ margin:0 0 4px; font-size:20px; }}
  .sub {{ color:var(--muted); font-size:13px; }}
  .controls {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap;
               background:var(--panel); border:1px solid var(--border); border-radius:12px;
               padding:12px 18px; margin-bottom:16px; }}
  .btn {{ background:#1e293b; color:var(--fg); border:1px solid #334155; border-radius:8px;
          padding:7px 14px; cursor:pointer; font-size:13px; font-weight:600; }}
  .btn.active {{ background:#065f46; border-color:#10b981; color:#ecfdf5; }}
  .btn.warn.active {{ background:#7f1d1d; border-color:#ef4444; color:#fef2f2; }}
  .grid {{ display:grid; grid-template-columns:1.55fr 1fr; gap:16px; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; }}
  canvas {{ width:100%; height:auto; border-radius:8px; border:1px solid #334155; display:block; }}
  .kpi-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:14px; }}
  .kpi {{ background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:12px; }}
  .kpi .v {{ font-size:24px; font-weight:800; }}
  .kpi .k {{ font-size:11px; color:var(--muted); text-transform:uppercase; }}
  .act-item {{ background:#0f172a; border:1px solid #1e293b; border-radius:8px;
               padding:10px 12px; margin-bottom:8px; font-size:13px; }}
  .gain {{ color:#34d399; font-weight:700; }}
  svg {{ width:100%; height:190px; background:#0f172a; border-radius:8px; border:1px solid #1e293b; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>resilience-wm &mdash; Interactive Urban Resilience World-Model Sandbox</h1>
      <div class="sub" id="subtitle"></div>
    </div>
    <div style="display:flex;gap:8px;">
      <button id="btnBase" class="btn warn" onclick="setMode('baseline')">Baseline (No Action)</button>
      <button id="btnOpt" class="btn active" onclick="setMode('optimal')">World-Model Optimal Plan</button>
    </div>
  </header>

  <div class="controls">
    <button id="playBtn" class="btn" onclick="togglePlay()">&#9654; Play Rollout</button>
    <label style="font-size:13px;">Time: <strong id="hourLbl">0h</strong></label>
    <input id="timeSlider" type="range" min="0" max="6" value="3" step="1" style="flex:1;min-width:180px;" oninput="render()" />
    <label style="font-size:13px;"><input id="chkHeat" type="checkbox" checked onchange="render()" /> Hazard Field</label>
    <label style="font-size:13px;"><input id="chkPower" type="checkbox" checked onchange="render()" /> Power Links</label>
  </div>

  <div class="grid">
    <div class="panel">
      <canvas id="mapCanvas" width="864" height="576"></canvas>
    </div>
    <div class="panel">
      <div class="kpi-row">
        <div class="kpi"><div class="v" id="kpiR" style="color:#34d399">--</div><div class="k">Resilience R(t)</div></div>
        <div class="kpi"><div class="v" id="kpiIso" style="color:#f87171">--</div><div class="k">Isolated Pop</div></div>
        <div class="kpi"><div class="v" id="kpiBlk" style="color:#fbbf24">--</div><div class="k">Severed Roads</div></div>
      </div>
      <div style="font-size:13px;font-weight:600;margin-bottom:6px;">Resilience Triangle R(t) Comparison</div>
      <svg id="chartSvg" viewBox="0 0 460 190"></svg>
      <div style="font-size:13px;font-weight:600;margin:14px 0 8px;color:#fde047;">Counterfactual Optimal Plan & Attribution</div>
      <div id="actionList"></div>
    </div>
  </div>
</div>

<script>
const DATA = {json.dumps(data_blob)};
let mode = 'optimal';
let timer = null;

document.getElementById('subtitle').textContent =
  `${{DATA.city_name}}  •  Hazard: ${{DATA.hazard.toUpperCase()}}  •  ` +
  `Mean R(t): ${{DATA.baseline.mean_resilience.toFixed(3)}} -> ${{DATA.optimal.mean_resilience.toFixed(3)}}`;

const slider = document.getElementById('timeSlider');
slider.max = DATA.baseline.frames.length - 1;
slider.value = Math.min(3, DATA.baseline.frames.length - 1);

const nodeMap = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));
document.getElementById('actionList').innerHTML = DATA.actions.map((a, i) =>
  `<div class="act-item"><strong>${{i+1}}. ${{a.display_name}}</strong><br/>` +
  `<span class="gain">dR = +${{a.delta_r.toFixed(3)}}</span> &bull; Rescued: <strong>${{Math.max(0, a.rescued).toLocaleString()}}</strong> residents</div>`
).join('');

function setMode(m) {{
  mode = m;
  document.getElementById('btnBase').classList.toggle('active', m === 'baseline');
  document.getElementById('btnOpt').classList.toggle('active', m === 'optimal');
  render();
}}

function togglePlay() {{
  if (timer) {{
    clearInterval(timer); timer = null;
    document.getElementById('playBtn').innerHTML = '&#9654; Play Rollout';
  }} else {{
    document.getElementById('playBtn').innerHTML = '&#9646;&#9646; Pause';
    timer = setInterval(() => {{
      slider.value = (parseInt(slider.value) + 1) % DATA.baseline.frames.length;
      render();
    }}, 850);
  }}
}}

function drawChart(idx) {{
  const svg = document.getElementById('chartSvg');
  const bf = DATA.baseline.frames, of = DATA.optimal.frames;
  const x0 = 42, y0 = 18, w = 398, h = 142;
  const px = i => x0 + (i / Math.max(1, bf.length - 1)) * w;
  const py = r => y0 + (1 - r) * h;
  const bPts = bf.map((f, i) => `${{px(i)}},${{py(f.metrics.resilience_index)}}`).join(' ');
  const oPts = of.map((f, i) => `${{px(i)}},${{py(f.metrics.resilience_index)}}`).join(' ');
  const poly = of.map((f, i) => `${{px(i)}},${{py(f.metrics.resilience_index)}}`)
    .concat(bf.slice().reverse().map((f, j) => `${{px(bf.length-1-j)}},${{py(f.metrics.resilience_index)}}`)).join(' ');
  const cx = px(idx);
  svg.innerHTML = `
    <line x1="${{x0}}" y1="${{py(0.5)}}" x2="${{x0+w}}" y2="${{py(0.5)}}" stroke="#1e293b" stroke-dasharray="4 4"/>
    <polygon points="${{poly}}" fill="rgba(16,185,129,0.22)"/>
    <polyline points="${{bPts}}" fill="none" stroke="#ef4444" stroke-width="2.5"/>
    <polyline points="${{oPts}}" fill="none" stroke="#10b981" stroke-width="2.5"/>
    <line x1="${{cx}}" y1="${{y0}}" x2="${{cx}}" y2="${{y0+h}}" stroke="#facc15" stroke-width="1.8"/>
    <text x="6" y="${{y0+6}}" fill="#94a3b8" font-size="11">1.0</text>
    <text x="6" y="${{py(0.5)+4}}" fill="#94a3b8" font-size="11">0.5</text>
    <text x="6" y="${{y0+h}}" fill="#94a3b8" font-size="11">0.0</text>
  `;
}}

function render() {{
  const idx = parseInt(slider.value);
  const fr = DATA[mode].frames[idx];
  const m = fr.metrics;
  document.getElementById('hourLbl').textContent = `${{fr.hour}}h`;
  document.getElementById('kpiR').textContent = m.resilience_index.toFixed(3);
  document.getElementById('kpiR').style.color = m.resilience_index >= 0.75 ? '#34d399' : (m.resilience_index >= 0.5 ? '#fbbf24' : '#f87171');
  document.getElementById('kpiIso').textContent = m.isolated_population.toLocaleString();
  document.getElementById('kpiBlk').textContent = m.blocked_edges.length;
  drawChart(idx);

  const canvas = document.getElementById('mapCanvas');
  const ctx = canvas.getContext('2d');
  const cw = canvas.width / DATA.cols, ch = canvas.height / DATA.rows;
  const showHeat = document.getElementById('chkHeat').checked;
  const showPower = document.getElementById('chkPower').checked;

  // Raster background
  for (let r = 0; r < DATA.rows; r++) {{
    for (let c = 0; c < DATA.cols; c++) {{
      let rr = 18 + DATA.dem[r][c] * 1.4, gg = 26 + DATA.dem[r][c] * 1.8, bb = 42 + DATA.dem[r][c] * 2.1;
      if (DATA.river[r][c]) {{ rr = 14; gg = 56; bb = 108; }}
      if (showHeat) {{
        const val = fr.field[r][c] / 100.0;
        if (DATA.hazard === 'flood' && val > 0.06) {{
          const a = Math.min(0.86, (val - 0.06) / 0.7);
          rr = rr*(1-a) + 35*a; gg = gg*(1-a) + 165*a; bb = bb*(1-a) + 255*a;
        }} else if (DATA.hazard === 'earthquake' && val > 0.14) {{
          const a = Math.min(0.76, (val - 0.14) / 0.52);
          rr = rr*(1-a) + 245*a; gg = gg*(1-a) + 95*a; bb = bb*(1-a) + 45*a;
        }}
      }}
      ctx.fillStyle = `rgb(${{rr|0}},${{gg|0}},${{bb|0}})`;
      ctx.fillRect(c * cw, r * ch, cw + 0.6, ch + 0.6);
    }}
  }}

  const sx = canvas.width / Math.max(1, DATA.orig_cols - 1);
  const sy = canvas.height / Math.max(1, DATA.orig_rows - 1);
  const xy = n => [n.col * sx, n.row * sy];
  const blocked = new Set(m.blocked_edges);
  const degraded = new Set(m.degraded_edges);
  const isolated = new Set(m.isolated_districts);
  const offlineSubs = new Set(m.offline_substations);
  const activeTargets = mode === 'optimal' ? new Set(DATA.actions.map(a => a.target_id)) : new Set();

  if (showPower) {{
    DATA.nodes.filter(n => n.kind === 'hospital' && n.power_source).forEach(h => {{
      const s = nodeMap[h.power_source];
      const [x1, y1] = xy(s), [x2, y2] = xy(h);
      ctx.strokeStyle = offlineSubs.has(s.id) ? 'rgba(239,68,68,0.65)' : 'rgba(250,204,21,0.45)';
      ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([]);
    }});
  }}

  DATA.edges.forEach(e => {{
    const [x1, y1] = xy(nodeMap[e.u]), [x2, y2] = xy(nodeMap[e.v]);
    ctx.strokeStyle = blocked.has(e.id) ? '#ef4444' : (degraded.has(e.id) ? '#f59e0b' : 'rgba(56,189,248,0.85)');
    ctx.lineWidth = (e.kind === 'bridge' || e.kind === 'tunnel') ? 4 : 2.6;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    if (activeTargets.has(e.id)) {{
      ctx.fillStyle = '#facc15'; ctx.beginPath(); ctx.arc((x1+x2)/2, (y1+y2)/2, 7, 0, 6.28); ctx.fill();
    }}
  }});

  ctx.font = '11px ui-monospace, SFMono-Regular, monospace';
  DATA.nodes.forEach(n => {{
    const [x, y] = xy(n);
    if (n.kind === 'residential') {{
      const rad = 6 + n.pop / 28000;
      if (isolated.has(n.id)) {{
        ctx.fillStyle = 'rgba(239,68,68,0.35)'; ctx.beginPath(); ctx.arc(x, y, rad+6, 0, 6.28); ctx.fill();
      }}
      ctx.fillStyle = isolated.has(n.id) ? '#dc2626' : '#10b981';
      ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.28); ctx.fill();
      ctx.fillStyle = '#f1f5f9'; ctx.fillText(n.id, x + rad + 4, y + 4);
    }} else if (n.kind === 'hospital') {{
      const cap = m.hospital_capacity[n.id] ?? 1.0;
      ctx.fillStyle = cap >= 0.8 ? '#10b981' : (cap > 0.05 ? '#f59e0b' : '#dc2626');
      ctx.fillRect(x-8, y-8, 16, 16);
      ctx.fillStyle = '#fff'; ctx.fillRect(x-5, y-1.5, 10, 3); ctx.fillRect(x-1.5, y-5, 3, 10);
      ctx.fillText(`${{n.id}}(${{Math.round(cap*100)}}%)`, x + 11, y + 4);
    }} else if (n.kind === 'substation') {{
      ctx.fillStyle = offlineSubs.has(n.id) ? '#dc2626' : '#facc15';
      ctx.beginPath(); ctx.moveTo(x, y-9); ctx.lineTo(x+9, y); ctx.lineTo(x, y+9); ctx.lineTo(x-9, y); ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#fef08a'; ctx.fillText(n.id, x + 11, y + 4);
    }} else if (n.kind === 'shelter') {{
      ctx.fillStyle = '#818cf8';
      ctx.beginPath(); ctx.moveTo(x, y-8); ctx.lineTo(x+8, y+6); ctx.lineTo(x-8, y+6); ctx.closePath(); ctx.fill();
    }} else {{
      ctx.fillStyle = '#94a3b8'; ctx.fillRect(x-4, y-4, 8, 8);
    }}
    if (activeTargets.has(n.id)) {{
      ctx.strokeStyle = '#facc15'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(x, y, 14, 0, 6.28); ctx.stroke();
    }}
  }});
}}

render();
</script>
</body>
</html>
"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out
