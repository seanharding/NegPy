"""Prototype: a roll color reference from measured film constants instead of frame content.

Roll Analysis pools each frame's own end points, which follow the subject. This builds the
roll's locked end points from measurements instead: neutral picks, or the gray ramp of a
SpyderCheckr 24 fitted as the film's neutral axis and carried to other rolls relative to their
film base. Brightness stays the frame's or the roll's; Cast Removal is off. Renders a roll with
today's reference and with the measured one, for comparison. Frames need an edit in this
machine's edits database or in the JSON that NEGPY_PROTO_EDITS names (see `export`).

    uv run python scripts/proto_measured_reference.py preview <raw>
    uv run python scripts/proto_measured_reference.py run <roll folder> <pick raw> <pick> [<pick>] --out <dir>
    uv run python scripts/proto_measured_reference.py chart <bare film raw> <chart raw> "x,y;x,y;x,y;x,y" --out <dir>
    uv run python scripts/proto_measured_reference.py run <roll folder> <any raw> --reference <dir>/reference.json --rebate-frame <raw> --out <dir>
    uv run python scripts/proto_measured_reference.py export <roll folder>... --out edits.json

A pick is y0,y1,x0,x1 as fractions of the post-geometry frame (see `preview`).
"""

import argparse
import io
import json
import os
import sqlite3
import sys
from dataclasses import replace

import numpy as np
from PIL import Image, ImageDraw

from negpy.domain.interfaces import PipelineContext
from negpy.domain.models import WorkspaceConfig
from negpy.domain.types import LUMA_B, LUMA_G, LUMA_R
from negpy.features.exposure.normalization import effective_crosstalk_matrix, to_log_density, unmix_log_image
from negpy.features.geometry.processor import GeometryProcessor
from negpy.services.capture.calibration import TARGET_FRACTION
from negpy.services.rendering.image_processor import ImageProcessor

_LUMA = np.array([LUMA_R, LUMA_G, LUMA_B])


def edits_db() -> str:
    return os.path.join(os.environ.get("NEGPY_USER_DIR", os.path.expanduser("~/Source/NegPy-dev")), "edits.db")


def _key(path: str) -> str:
    """Roll folder and file name: the same on any machine that syncs the scans."""
    return "/".join(path.rstrip("/").split("/")[-2:])


def exported_edits() -> dict:
    """Edits exported from another machine (`export` command), keyed by roll folder and file name."""
    path = os.environ.get("NEGPY_PROTO_EDITS", "")
    return json.load(open(path)) if path and os.path.exists(path) else {}


def lookup_edit(raw: str):
    """(file hash, flat edit) from this machine's edits database, else from the exported edits."""
    row = None
    if os.path.exists(edits_db()):
        try:
            row = (
                sqlite3.connect(edits_db())
                .execute("SELECT file_hash, settings_json FROM file_settings WHERE file_path = ?", (raw,))
                .fetchone()
            )
        except sqlite3.OperationalError:
            row = None
    if row is not None:
        return row[0], json.loads(row[1])
    entry = exported_edits().get(_key(raw))
    return (entry["hash"], entry["edit"]) if entry else None


def stored_edit(raw: str) -> tuple[str, dict]:
    found = lookup_edit(raw)
    if found is None:
        sys.exit(f"no stored edit for {raw}: open it in NegPy once, or set NEGPY_PROTO_EDITS")
    return found


def measurement_space(ip: ImageProcessor, raw: str, cfg: WorkspaceConfig) -> tuple[np.ndarray, np.ndarray]:
    """(linear, log) post-geometry frame in the space Roll Analysis measures bounds in."""
    source, _ir, _cs = ip._load_source_f32(raw, cfg)
    ctx = PipelineContext(original_size=(source.shape[1], source.shape[0]), scale_factor=1.0, process_mode=cfg.process.process_mode)
    linear = GeometryProcessor(cfg.geometry).process(source, ctx)
    log = unmix_log_image(to_log_density(linear), effective_crosstalk_matrix(cfg.process, cfg.process.process_mode))
    return linear, log


def parse_pick(text: str) -> tuple[float, float, float, float]:
    y0, y1, x0, x1 = (float(v) for v in text.split(","))
    return y0, y1, x0, x1


def patch_value(log: np.ndarray, pick) -> np.ndarray:
    """Per-channel median, so a window or a seam inside the patch does not move it."""
    h, w = log.shape[:2]
    y0, y1, x0, x1 = pick
    return np.median(log[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w), :3].reshape(-1, 3), axis=0)


def luma_free(v: np.ndarray) -> np.ndarray:
    return v - float(v @ _LUMA)


def measured_bounds(picks: list[np.ndarray], floors, ceils) -> tuple[tuple, tuple]:
    """Locked (floors, ceils) whose color comes from neutral picks and whose luma is the given
    bounds'. One pick: both ends take its color (equal layer contrast). Two or more: each end
    sits on the neutral axis fitted through them, at the end's own luma."""
    floors, ceils = np.asarray(floors, float), np.asarray(ceils, float)
    if len(picks) == 1:
        chroma = luma_free(picks[0])
        f, c = chroma + floors @ _LUMA, chroma + ceils @ _LUMA
    else:
        pts = np.array(picks)
        lum = pts @ _LUMA
        slope, icept = np.polyfit(lum, pts, 1)  # each channel's density against the neutral's luma
        f, c = icept + slope * (floors @ _LUMA), icept + slope * (ceils @ _LUMA)
    return tuple(float(v) for v in f), tuple(float(v) for v in c)


def preview(raw: str, out: str, picks=()) -> None:
    """The post-geometry frame as a positive, with a 10% grid and any picks drawn on it."""
    ip = ImageProcessor()
    _h, flat = stored_edit(raw)
    _linear, log = measurement_space(ip, raw, WorkspaceConfig.from_flat_dict(flat))
    lo, hi = np.percentile(log[..., :3], (1, 99), axis=(0, 1))
    pos = np.clip((hi - log[..., :3]) / (hi - lo), 0, 1) ** (1 / 2.2)
    im = Image.fromarray((pos * 255).astype(np.uint8))
    im.thumbnail((1200, 1200))
    d, (w, h) = ImageDraw.Draw(im), im.size
    for i in range(1, 10):
        d.line([(w * i / 10, 0), (w * i / 10, h)], fill=(255, 255, 0))
        d.line([(0, h * i / 10), (w, h * i / 10)], fill=(255, 255, 0))
        d.text((w * i / 10 + 2, 2), f".{i}", fill=(255, 255, 0))
        d.text((2, h * i / 10 + 2), f".{i}", fill=(255, 255, 0))
    for y0, y1, x0, x1 in picks:
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(0, 255, 255), width=3)
    im.save(out)
    print("saved", out, im.size)


def calibrated_base(cfg: WorkspaceConfig) -> np.ndarray:
    """The film base in measurement space under a calibrated preset, which sets the light so the
    raw base is equal in every channel: the sensor unmix is exact at the base (matrix times
    base), then the film crosstalk unmix. Only its color is meaningful."""
    from negpy.features.process.sensor import effective_sensor_matrix

    base = np.full(3, TARGET_FRACTION)
    matrix = effective_sensor_matrix(cfg.process)
    if matrix is not None:
        base = np.asarray(matrix, float).reshape(3, 3) @ base
    log = np.log10(base).astype(np.float32).reshape(1, 1, 3)
    return unmix_log_image(log, effective_crosstalk_matrix(cfg.process, cfg.process.process_mode)).reshape(3).astype(float)


def roll_frames(folder: str) -> list[tuple[str, str, dict]]:
    """(raw path, file hash, flat edit) for each frame in `folder` with an edit, in file order."""
    folder = folder.rstrip("/")
    frames = []
    for name in sorted(os.listdir(folder)):
        raw = os.path.join(folder, name)
        found = lookup_edit(raw) if os.path.isfile(raw) else None
        if found is not None:
            frames.append((raw, found[0], found[1]))
    return frames


def render(ip: ImageProcessor, raw: str, file_hash: str, flat: dict, edge: int) -> Image.Image:
    cfg = WorkspaceConfig.from_flat_dict(flat)
    cfg = replace(
        cfg,
        export=replace(cfg.export, export_fmt="JPEG", jpeg_quality=90, export_resolution_mode="target_px", export_target_long_edge_px=edge),
    )
    out = ip.process_export(raw, cfg, cfg.export, file_hash, prefer_gpu=True)
    return Image.open(io.BytesIO(out[0] if isinstance(out, tuple) else out)).convert("RGB")


def contact_sheet(images: list[tuple[str, Image.Image]], out: str, cell: int = 300, cols: int = 6) -> None:
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 16)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(images):
        t = im.copy()
        t.thumbnail((cell - 6, cell - 6))
        x, y = (i % cols) * cell, (i // cols) * (cell + 16)
        sheet.paste(t, (x + (cell - t.width) // 2, y + (cell - t.height) // 2))
        d.text((x + 4, y + cell + 2), name, fill="black")
    sheet.save(out, quality=88)


def srgb_lab(rgb: np.ndarray) -> np.ndarray:
    """CIELAB D65 of sRGB values in 0-1, the space reference_lab adapts the chart to."""
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def own_bounds(log: np.ndarray) -> tuple[tuple, tuple]:
    """The frame's own (floors, ceils) as Roll Analysis measures them, for its brightness only."""
    from negpy.features.exposure.normalization import analyze_log_exposure_bounds_from_log

    b = analyze_log_exposure_bounds_from_log(np.ascontiguousarray(log[..., :3]))
    return tuple(b.floors), tuple(b.ceils)


def locked_edit(flat: dict, values: list[np.ndarray], floors, ceils) -> dict:
    """The edit rendering with the measured reference at the given bounds' brightness, exactly:
    both averages on, so mix_luma_color_bounds never slides the ends off the neutral axis. Cast
    Removal is off: it pulls midtones toward an axis estimated from the frame's content."""
    f, c = measured_bounds(values, floors, ceils)
    return {
        **flat,
        "locked_floors": f,
        "locked_ceils": c,
        "use_luma_average": True,
        "use_color_average": True,
        "use_cast_average": False,
        "cast_removal_strength": 0.0,
        "lock_bounds": False,
    }


def chart_order(values: np.ndarray, spec) -> np.ndarray:
    """The orientation whose top row reads as the gray ramp: densest at Card White, steadily
    thinner to Card Black, with the least color spread."""
    from negpy.features.process.chart import ORIENTATIONS

    best, best_score = ORIENTATIONS[0], np.inf
    for order in ORIENTATIONS:
        printed = np.empty_like(values)
        printed[order] = values
        grays = printed[:6]
        luma = grays @ _LUMA
        if not np.all(np.diff(luma) > 0):
            continue
        score = float(np.std(np.array([luma_free(g) for g in grays[:5]]), axis=0).sum())
        if score < best_score:
            best, best_score = order, score
    return best


def chart(base_raw: str, chart_raw: str, corners: np.ndarray, out: str, like: str, edge: int) -> None:
    """Fit the neutral axis to a SpyderCheckr's gray ramp and score the chart rendered with it."""
    from negpy.features.process.chart import SPYDERCHECKR_24, _patch_medians, delta_e2000, reference_lab

    os.makedirs(out, exist_ok=True)
    ip = ImageProcessor()
    like_edit = stored_edit(like)[1] if like else {}

    def edit_for(raw: str) -> tuple[str, dict]:
        found = lookup_edit(raw)
        flat = dict(found[1]) if found else dict(like_edit)
        flat.update({k: like_edit[k] for k in _LIKE_FIELDS if k in like_edit})
        flat.update(crop_rect=None, crop_from_auto=False)
        return (found[0] if found else os.path.basename(raw)), flat

    _bh, base_flat = edit_for(base_raw)
    _bl, base_log = measurement_space(ip, base_raw, WorkspaceConfig.from_flat_dict(base_flat))
    h, w = base_log.shape[:2]
    base = np.median(base_log[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :3].reshape(-1, 3), axis=0)
    print("base log", base.round(4), "luma-free", luma_free(base).round(4))

    ch, chart_flat = edit_for(chart_raw)
    linear, log = measurement_space(ip, chart_raw, WorkspaceConfig.from_flat_dict(chart_flat))
    h, w = log.shape[:2]
    corners_px = corners * np.array([w, h])
    grid = _patch_medians(np.ascontiguousarray(log[..., :3]), corners_px, None)
    order = chart_order(grid, SPYDERCHECKR_24)
    printed = np.empty_like(grid)
    printed[order] = grid
    grays = printed[:5]
    for name, v in zip(SPYDERCHECKR_24.patch_names[:6], printed[:6]):
        print(f"  {name:12s} log {v.round(4)}  luma-free {luma_free(v).round(4)}")
    lum = grays @ _LUMA
    slope, icept = np.polyfit(lum, grays, 1)
    resid = grays - (icept + np.outer(lum, slope))
    print("neutral axis slope (per channel, vs luma)", slope.round(4), "| fit residual max", float(np.abs(resid).max()).__round__(4))
    print("pack, 40% gray minus base, luma-free", (luma_free(printed[2]) - luma_free(base)).round(4))
    json.dump(
        {
            "base": base.tolist(),
            "grays": printed[:6].tolist(),
            "slope": slope.tolist(),
            "intercept": icept.tolist(),
            "settings_from": like or chart_raw,
        },
        open(os.path.join(out, "reference.json"), "w"),
        indent=1,
    )

    ref = reference_lab(SPYDERCHECKR_24)
    lf, lc = own_bounds(log)
    variants = {
        "own balance": dict(chart_flat),
        "one gray (40%)": locked_edit(chart_flat, [printed[2]], lf, lc),
        "neutral axis": locked_edit(chart_flat, list(grays), lf, lc),
        "axis, Digital stock": {**locked_edit(chart_flat, list(grays), lf, lc), "paper_profile": "neutral"},
    }
    print(f"\n{'render':22s} {'mean dE00':>9s} {'max':>6s}   gray ramp a*/b* (white, 20, 40, 60, 80, black)")
    for label, flat in variants.items():
        im = render(ip, chart_raw, ch, flat, edge)
        slug = label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("%", "")
        im.save(os.path.join(out, f"chart_{slug}.jpg"), quality=92)
        a = np.asarray(im, dtype=np.float32)[..., :3] / 255.0
        cp = corners * np.array([im.width, im.height])
        rgb = np.empty((24, 3))
        rgb[order] = _patch_medians(np.ascontiguousarray(a), cp, None)
        lab = srgb_lab(rgb)
        # One exposure gain on the mid grays, as measure_lab does: brightness is not the question.
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        y_ref = ((ref[1:5, 0] + 16) / 116) ** 3
        y = np.maximum(lin[1:5] @ np.array([0.2126, 0.7152, 0.0722]), 1e-6)
        g = float(np.exp(np.mean(np.log(y_ref / y))))
        lin2 = np.clip(lin * g, 0, 1)
        lab = srgb_lab(np.where(lin2 <= 0.0031308, lin2 * 12.92, 1.055 * lin2 ** (1 / 2.4) - 0.055))
        de = delta_e2000(ref, lab)
        ramp = " ".join(f"{lab[i, 1]:+.1f}/{lab[i, 2]:+.1f}" for i in range(6))
        print(f"{label:22s} {de.mean():9.2f} {de.max():6.2f}   {ramp}")
        np.savetxt(
            os.path.join(out, f"dE_{slug}.txt"),
            np.column_stack([de, lab, ref]),
            fmt="%.2f",
            header="dE00 L a b refL refa refb (printed order)",
        )
    print("outputs in", out)


_LIKE_FIELDS = ("crosstalk_matrix", "crosstalk_strength", "crosstalk_profile", "paper_profile")


def rebate_base(log: np.ndarray) -> np.ndarray:
    """The film base from the rebate strip a holder leaves along the frame's top edge."""
    h, w = log.shape[:2]
    return np.median(log[: max(2, h // 150), w // 10 : 9 * w // 10, :3].reshape(-1, 3), axis=0)


def center_base(log: np.ndarray) -> np.ndarray:
    """The film base from a frame of bare film: the median of its center."""
    h, w = log.shape[:2]
    return np.median(log[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :3].reshape(-1, 3), axis=0)


def reference_axis(reference: dict, target_base: np.ndarray) -> list[np.ndarray]:
    """Two points on a chart-fitted neutral axis, moved by the target roll's base color against
    the chart roll's: the pack is stored relative to the base."""
    slope, icept = np.asarray(reference["slope"]), np.asarray(reference["intercept"])
    shift = luma_free(np.asarray(target_base)) - luma_free(np.asarray(reference["base"]))
    return [icept + slope * lum + shift for lum in (-1.6, -0.4)]


def run(
    folder: str,
    pick_raw: str,
    picks,
    out: str,
    edge: int,
    skip_stored: bool = False,
    straight: bool = False,
    chroma=None,
    like: str = "",
    reference: str = "",
    base_raw: str = "",
    rebate_frame: str = "",
) -> None:
    os.makedirs(out, exist_ok=True)
    ip = ImageProcessor()
    _h, flat = stored_edit(pick_raw)
    cfg = WorkspaceConfig.from_flat_dict(flat)
    like_edit = stored_edit(like)[1] if like else {}
    measure_flat = {**flat, **{k: like_edit[k] for k in _LIKE_FIELDS if k in like_edit}}
    values = []
    if picks:
        _linear, log = measurement_space(ip, pick_raw, cfg)
        values = [patch_value(log, p) for p in picks]
    for p, v in zip(picks, values):
        print("pick", p, "log density", v.round(4), "luma-free", luma_free(v).round(4))
    variants = {"stored": None}
    if values:
        variants["v0_one_neutral"] = values[-1:]
    if len(values) >= 2:
        variants["v1_neutral_axis"] = values
    if chroma is not None:
        variants["v0_given_neutral"] = [np.asarray(chroma, float)]
    if straight:
        base = calibrated_base(cfg)
        print("calibrated base", base.round(4), "luma-free", luma_free(base).round(4))
        variants["straight_film"] = [base]
    if reference:
        ref = json.load(open(reference))
        source = base_raw or rebate_frame
        if not source:
            sys.exit("--reference needs --base-raw or --rebate-frame for the target roll's base")
        found = lookup_edit(source)
        log = measurement_space(
            ip,
            source,
            WorkspaceConfig.from_flat_dict(
                {**(found[1] if found else measure_flat), **{k: like_edit[k] for k in _LIKE_FIELDS if k in like_edit}}
            ),
        )[1]
        target = center_base(log) if base_raw else rebate_base(log)
        print(
            "target base",
            target.round(4),
            "luma-free",
            luma_free(target).round(4),
            "| chart roll base luma-free",
            luma_free(np.asarray(ref["base"])).round(4),
        )
        variants["chart_reference"] = reference_axis(ref, target)
    if skip_stored:
        del variants["stored"]
    sheets: dict[str, list] = {k: [] for k in variants}
    for raw, file_hash, frame in roll_frames(folder):
        name = os.path.basename(raw).rsplit("_", 1)[-1].split(".")[0]
        for label, vals in variants.items():
            edit = dict(frame)
            if vals is not None:
                if like:
                    edit.update({k: like_edit[k] for k in _LIKE_FIELDS if k in like_edit})
                # Brightness from the roll's reference when the frame has one, else measured from the frame.
                rolled = any(v != 0.0 for v in frame.get("locked_floors", (0.0, 0.0, 0.0)))
                if rolled:
                    lf, lc = frame["locked_floors"], frame["locked_ceils"]
                else:
                    lf, lc = own_bounds(measurement_space(ip, raw, WorkspaceConfig.from_flat_dict(edit))[1])
                edit = locked_edit(edit, vals, lf, lc)
            im = render(ip, raw, file_hash, edit, edge)
            im.save(os.path.join(out, f"{name}_{label}.jpg"), quality=90)
            sheets[label].append((name, im))
        print("rendered", name)
    for label, images in sheets.items():
        contact_sheet(images, os.path.join(out, f"sheet_{label}.jpg"))
    print("sheets in", out)


def export(folders: list[str], out: str) -> None:
    """Write this machine's edits for the frames in `folders` to JSON, for NEGPY_PROTO_EDITS elsewhere."""
    data = {}
    db = sqlite3.connect(edits_db())
    for folder in folders:
        folder = folder.rstrip("/")
        for p, h, j in db.execute("SELECT file_path, file_hash, settings_json FROM file_settings WHERE file_path LIKE ?", (folder + "/%",)):
            if os.path.dirname(p) == folder:
                data[_key(p)] = {"hash": h, "edit": json.loads(j)}
    json.dump(data, open(out, "w"))
    print(f"exported {len(data)} edits to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview")
    p.add_argument("raw")
    p.add_argument("--out", default="preview.png")
    p.add_argument("--pick", action="append", default=[])
    r = sub.add_parser("run")
    r.add_argument("folder")
    r.add_argument("pick_raw")
    r.add_argument("picks", nargs="*", help="y0,y1,x0,x1 neutral patches, darkest first")
    r.add_argument("--out", required=True)
    r.add_argument("--edge", type=int, default=1200)
    r.add_argument("--skip-stored", action="store_true", help="do not render the stored edits again")
    r.add_argument("--straight", action="store_true", help="also render from the calibrated base alone, with no neutral")
    r.add_argument("--chroma", help="r,g,b log density of a neutral measured elsewhere (only its color is used)")
    r.add_argument("--like", default="", help="copy crosstalk and print stock from this frame's edit to the measured renders")
    r.add_argument("--reference", default="", help="reference.json from the chart command")
    r.add_argument("--base-raw", default="", help="a bare-film frame from this roll, for --reference")
    r.add_argument("--rebate-frame", default="", help="a frame of this roll whose top edge shows rebate, for --reference")
    e = sub.add_parser("export")
    e.add_argument("folders", nargs="+", help="roll folders whose edits to export")
    e.add_argument("--out", required=True)
    c = sub.add_parser("chart")
    c.add_argument("base_raw", help="a frame of bare film")
    c.add_argument("chart_raw", help="a frame of the SpyderCheckr 24")
    c.add_argument("corners", help="the four corner-patch centers as x,y fractions: x,y;x,y;x,y;x,y")
    c.add_argument("--out", required=True)
    c.add_argument("--like", default="", help="frame whose edit supplies settings when a frame has none stored")
    c.add_argument("--edge", type=int, default=1600)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.folders, args.out)
        return
    if args.cmd == "chart":
        corners = np.array([[float(v) for v in pt.split(",")] for pt in args.corners.split(";")])
        chart(args.base_raw, args.chart_raw, corners, args.out, args.like, args.edge)
        return
    if args.cmd == "preview":
        preview(args.raw, args.out, [parse_pick(s) for s in args.pick])
    else:
        run(
            args.folder,
            args.pick_raw,
            [parse_pick(s) for s in args.picks],
            args.out,
            args.edge,
            args.skip_stored,
            args.straight,
            [float(v) for v in args.chroma.split(",")] if args.chroma else None,
            args.like,
            args.reference,
            args.base_raw,
            args.rebate_frame,
        )


if __name__ == "__main__":
    main()
