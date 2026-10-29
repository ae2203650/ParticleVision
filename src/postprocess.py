"""
Post-processing of the final 4-class map: fill hollow microplastic particles.

This module holds the ONE implementation. The segmentation pipeline (PipelinePostprocess, called by
app/lab_structure.run_lab_batch), the Fix Segmentations page, tools/fix_segmented_folder.py and the evaluation all
call postprocess_microplastic(); nothing else re-implements it. It chooses the method by particle size:
reconstruct_circles() for 7 and 10 um particles, close_microplastic() for 5 um and when no size is known.

The definitions below describe fill_microplastic_holes(), the hole filling of the former three-step method.
  * MP = microplastic pixels (class 3, "Fines" in the code). D(r) = MP dilated by a disk x^2 + y^2 <= r^2
    (r = 1: 3 x 3 cross); D(0) = MP. Computed on a copy padded with r non-MP pixels, so the border adds nothing.
  * Enclosed: a pixel is enclosed by a mask if it is outside the mask and cannot reach the image border through
    4-connected pixels outside the mask (background 4-connectivity, scipy's default for hole filling). So a ring
    closed only diagonally encloses its interior, and a ring cut by the image border encloses nothing.
  * Inside pixels = non-MP pixels of  erode( fill_holes( D(r) ), disk r ).  For r = 0 these are the holes of MP.
    For r > 0 this is a binary closing of MP in which the holes are filled between the dilation and the erosion,
    so gaps of up to 2r px in a ring are bridged (an ordinary closing does not close a 1-px gap in a thin ring
    whose interior is wider than the disk).
  * Candidate regions = 4-connected components of the inside pixels. A region is FILLED (set to class 3) only if
      (a) its area <= max_hole_area, and
      (b) it encloses something: it contains a pixel that is enclosed by MP itself, or a pixel farther than r
          from MP (outside D(r)). Regions made only of pixels next to MP (closing bridges between separate
          particles, closed notches, slits) are dropped, so separate particles are never merged.
    Pixels of D(r) that are not MP and are 4-adjacent to a filled region (the gap of a C-shaped ring) are filled
    too. A C-ring whose entire interior lies within r of MP (a hollow of 1-2 px) is left as it is.
  * Only non-MP pixels change, and only to class 3; no MP pixel is ever removed.
"""

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label

MP_CLASS = 3
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)  # 4-connectivity


def _disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


def _enclosed(mask: np.ndarray) -> np.ndarray:
    """Pixels outside `mask` that cannot reach the image border through 4-connected non-mask pixels.
    Same result as scipy.ndimage.binary_fill_holes(mask) & ~mask (tested), with one labelling pass."""
    lab, n = label(~mask, structure=_CROSS)
    if n == 0:
        return np.zeros_like(mask)
    outside = np.zeros(n + 1, dtype=bool)
    outside[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
    outside[0] = False
    return (lab > 0) & ~outside[lab]


def fill_microplastic_holes(
    class_map: np.ndarray, closing_radius: int, max_hole_area: int, return_info: bool = False
):
    """Fill hollow microplastic particles in an H x W class-index map (values 0-3). Not used by the ParticleVision pipeline (kept from earlier experiments).

    Returns a new map with the same shape and dtype; with return_info=True also a dict with every candidate
    region (area, centre, filled or not, and why). See the module docstring for the exact rule."""
    if class_map.ndim != 2:
        raise ValueError(f"class_map must be 2-D (H x W), got shape {class_map.shape}")
    r, max_hole_area = int(closing_radius), int(max_hole_area)
    mp = class_map == MP_CLASS
    holes_mp = _enclosed(mp)
    if r > 0:
        se = _disk(r)
        d = binary_dilation(np.pad(mp, r), structure=se)  # D(r), padded
        outside = ~d & ~_enclosed(d)  # outside D(r), connected to the border
        inside = (~binary_dilation(outside, structure=se))[r:-r, r:-r] & ~mp
        near_mp = d[r:-r, r:-r]
        qualifies = holes_mp | (inside & ~near_mp)
    else:
        inside = qualifies = holes_mp
        near_mp = mp
    reg, n = label(inside, structure=_CROSS)
    out = class_map.copy()
    info = {"closing_radius": r, "max_hole_area": max_hole_area, "regions": [], "gap_pixels": 0}
    if n:
        area = np.bincount(reg.ravel(), minlength=n + 1)
        encl = np.bincount(reg.ravel(), weights=qualifies.ravel(), minlength=n + 1) > 0
        keep = (area <= max_hole_area) & encl
        keep[0] = False
        filled = keep[reg]
        if r > 0 and filled.any():
            gaps = near_mp & ~mp & ~inside & binary_dilation(filled, structure=_CROSS)
            info["gap_pixels"] = int(gaps.sum())
            filled |= gaps
        out[filled] = MP_CLASS
        if return_info:
            for k, sl in enumerate(find_objects(reg), 1):
                if sl is None:
                    continue
                info["regions"].append(
                    {
                        "area": int(area[k]),
                        "cy": (sl[0].start + sl[0].stop - 1) // 2,
                        "cx": (sl[1].start + sl[1].stop - 1) // 2,
                        "filled": bool(keep[k]),
                        "reason": "filled"
                        if keep[k]
                        else ("too large" if encl[k] else "encloses nothing (bridge / notch)"),
                    }
                )
    if return_info:
        info["holes_filled"] = sum(x["filled"] for x in info["regions"])
        info["pixels_changed"] = int((out != class_map).sum())
        return out, info
    return out


def postprocess_settings() -> dict:
    """The settings written into run logs / params files, so every output records how it was made (src/config.py)."""
    import hashlib
    import os

    from src.config import CFG

    with open(os.path.abspath(__file__), "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return {
        "method": "by_size",
        "method_by_size": {str(k): v for k, v in sorted(CFG.postprocess_method_by_size.items())},
        "default_method": "closing",
        "radius_by_size": {
            str(k): v for k, v in sorted(CFG.postprocess_closing_radius_by_size.items())
        },
        "default_radius": CFG.postprocess_closing_default_radius,
        "use_size": CFG.postprocess_closing_use_size,
        "max_hole_area": CFG.postprocess_closing_max_hole_area,
        "max_added_region": CFG.postprocess_closing_max_added_region,
        "circle_radii_by_size": {
            str(k): list(v) for k, v in sorted(CFG.postprocess_circle_radii_by_size.items())
        },
        "circle_min_support": CFG.postprocess_circle_min_support,
        "circle_band": CFG.postprocess_circle_band,
        "circle_min_visible": CFG.postprocess_circle_min_visible,
        "circle_max_core_fill": CFG.postprocess_circle_max_core_fill,
        "circle_nms_factor": CFG.postprocess_circle_nms_factor,
        "circle_disc_margin": CFG.postprocess_circle_disc_margin,
        "circle_max_overlaps": CFG.postprocess_circle_max_overlaps,
        "postprocess_py_md5": md5,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Per-object ring completion (step 2 of the former three-step method). Not used by the ParticleVision pipeline (kept from earlier experiments).
# Ran after fill_microplastic_holes, on the final 4-class map.
#   Every microplastic object (4-connected component) is treated on its OWN mask, in a small crop:
#   * method "object_closing" (option A): D = dilate(object, disk r); C = erode(fill_holes(D), disk r) | object.
#       With require_enclosure=True only objects whose dilation encloses something (i.e. rings, also C-shaped ones
#       with gaps <= 2r px) are changed; with False every small object gets its concavities closed.
#       grain_as_wall=True: grain pixels in the crop count as wall when testing for enclosure (a ring whose open side
#       is a grain edge counts as closed); grain pixels themselves never become microplastic.
#   * method "convex_hull" (option B): hull H of the object's pixels (pixel corners); filled only if
#       H <= max_hull_area px, solidity S = area / H <= max_solidity (clearly hollow) and the hull is roundish
#       (minor / major axis of the hull mask >= min_roundness).
#   For both: only objects with a bounding box <= max_object_size x max_object_size and area >= min_object_area are
#   considered, and a pixel may become microplastic only if it is Air or Water (never grain), is not part of a
#   different microplastic object, and is not 4-adjacent to a different object or to pixels already added for a
#   different object. So separate particles are never merged or bridged.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
AIR_CLASS, WATER_CLASS = 1, 2


def _hull_mask(own: np.ndarray):
    """Convex hull of a binary crop (hull of the pixel corners; a pixel is inside if its centre is), or None."""
    from scipy.spatial import ConvexHull, QhullError

    ys, xs = np.nonzero(own)
    corners = np.concatenate(
        [np.stack([xs + dx, ys + dy], 1) for dx in (0, 1) for dy in (0, 1)]
    ).astype(float)
    try:
        hull = ConvexHull(corners)
    except (QhullError, ValueError):
        return None
    yy, xx = np.mgrid[0 : own.shape[0], 0 : own.shape[1]]
    eq = hull.equations  # a*x + b*y + c <= 0 inside every facet
    val = (
        eq[:, 0, None, None] * (xx + 0.5) + eq[:, 1, None, None] * (yy + 0.5) + eq[:, 2, None, None]
    )
    inside = np.all(val <= 1e-9, axis=0)
    return inside | own


def _roundness(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov(np.stack([ys, xs]).astype(float)))
    return float(np.sqrt(max(ev[0], 0) / ev[1])) if ev[1] > 0 else 0.0


def complete_microplastic_rings(
    class_map: np.ndarray,
    method: str,
    max_object_size: int = 15,
    min_object_area: int = 5,
    closing_radius: int = 1,
    require_enclosure: bool = True,
    grain_as_wall: bool = False,
    connectivity: int = 4,
    enclosure_radius: Optional[int] = None,
    max_hull_area: int = 60,
    max_solidity: float = 0.7,
    min_roundness: float = 0.5,
    return_info: bool = False,
):
    """Per-object ring completion (see the block comment above). Returns a new map (and info with every changed
    object: its added pixels, centre and - for the hull - H, S and roundness)."""
    from scipy.ndimage import binary_erosion, binary_fill_holes

    if method not in ("object_closing", "convex_hull"):
        raise ValueError(f"unknown ring completion method {method!r}")
    out = class_map.copy()
    mp = class_map == MP_CLASS
    # connectivity 8 (a tested variant, not used): ring pieces that touch only diagonally form one object
    lab, n = label(mp, structure=_CROSS if connectivity == 4 else np.ones((3, 3), dtype=bool))
    info = {"method": method, "objects": []}
    if n == 0:
        return (out, info) if return_info else out
    area = np.bincount(lab.ravel(), minlength=n + 1)
    taken = np.zeros(mp.shape, dtype=bool)  # pixels added for any object so far
    H, W = mp.shape
    r = int(closing_radius)
    pad = r + 1 if method == "object_closing" else 1
    se = _disk(r) if method == "object_closing" else None
    for k, sl in enumerate(find_objects(lab), 1):
        if sl is None or area[k] < min_object_area:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if h > max_object_size or w > max_object_size:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        lc, cc, tc = lab[y0:y1, x0:x1], class_map[y0:y1, x0:x1], taken[y0:y1, x0:x1]
        own = lc == k
        others = ((lc > 0) & ~own) | tc
        allowed = (
            ((cc == AIR_CLASS) | (cc == WATER_CLASS))
            & ~others
            & ~binary_dilation(others, structure=_CROSS)
        )
        rec = {}
        if method == "object_closing":
            p = np.pad(own, r)
            d = binary_dilation(p, structure=se)
            if grain_as_wall:  # a ring closed by a grain edge counts as closed
                d |= np.pad(cc == 0, r)
            f = binary_fill_holes(d)
            encl = bool((f & ~d).any())
            if not encl and enclosure_radius is not None and enclosure_radius != r:
                # tested variant: a narrow hollow (<= ~2r px) is swallowed by the r-dilation; test it with a
                # smaller widening too (the fill itself still uses radius r, so gaps up to ~2r px are bridged)
                re_ = int(enclosure_radius)
                d2 = binary_dilation(np.pad(own, re_), structure=_disk(re_))
                encl = bool((binary_fill_holes(d2) & ~d2).any())
            if require_enclosure and not encl:
                continue
            new = (binary_erosion(f, structure=se)[r:-r, r:-r] | own) & ~own & allowed
        else:
            hull = _hull_mask(own)
            if hull is None:
                continue
            hull_area = int(hull.sum())
            solidity = area[k] / hull_area
            rnd = _roundness(hull)
            if hull_area > max_hull_area or solidity > max_solidity or rnd < min_roundness:
                continue
            new = hull & ~own & allowed
            rec = {
                "hull_area": hull_area,
                "solidity": round(float(solidity), 3),
                "roundness": round(rnd, 3),
            }
        if not new.any():
            continue
        out[y0:y1, x0:x1][new] = MP_CLASS
        taken[y0:y1, x0:x1] |= new
        if return_info:
            ys, xs = np.nonzero(new)
            info["objects"].append(
                {
                    "object_area": int(area[k]),
                    "added": int(new.sum()),
                    "cy": int(ys.mean()) + y0,
                    "cx": int(xs.mean()) + x0,
                    "label": k,
                    "bbox": (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop),
                    **rec,
                }
            )
    if return_info:
        info["pixels_changed"] = int((out != class_map).sum())
        return out, info
    return out


def postprocess_microplastic_steps(
    class_map: np.ndarray,
    closing_radius: Optional[int] = None,
    max_hole_area: Optional[int] = None,
    ring_method: Optional[str] = None,
    ring_closing_radius: Optional[int] = None,
    ring_max_object_size: Optional[int] = None,
    circle_fit: Optional[bool] = None,
    circle_r_min: Optional[float] = None,
    circle_r_max: Optional[float] = None,
    circle_max_rms: Optional[float] = None,
    circle_min_arc: Optional[float] = None,
    return_info: bool = False,
):
    """The former three-step method. Not used by the ParticleVision pipeline (kept from earlier experiments). Three steps in order:
    1. fill_microplastic_holes(closing_radius, max_hole_area)
    2. complete_microplastic_rings(ring_method, closing radius, max object size, enclosure required) unless "none"
    3. fill_fitted_circles(radius window, max RMS, min arc) if circle_fit
    """
    from src.config import CFG

    r = CFG.postprocess_closing_radius if closing_radius is None else closing_radius
    a = CFG.postprocess_max_hole_area if max_hole_area is None else max_hole_area
    rm = CFG.postprocess_ring_method if ring_method is None else ring_method
    rr = CFG.postprocess_ring_closing_radius if ring_closing_radius is None else ring_closing_radius
    rs = (
        CFG.postprocess_ring_max_object_size
        if ring_max_object_size is None
        else ring_max_object_size
    )
    cf = CFG.postprocess_circle_fit if circle_fit is None else circle_fit
    c0 = CFG.postprocess_circle_r_min if circle_r_min is None else circle_r_min
    c1 = CFG.postprocess_circle_r_max if circle_r_max is None else circle_r_max
    cr = CFG.postprocess_circle_max_rms if circle_max_rms is None else circle_max_rms
    ca = CFG.postprocess_circle_min_arc if circle_min_arc is None else circle_min_arc
    out, hinfo = fill_microplastic_holes(class_map, r, a, return_info=True)
    rinfo = {"objects": []}
    if rm and rm != "none":
        out, rinfo = complete_microplastic_rings(
            out, rm, closing_radius=rr, max_object_size=rs, require_enclosure=True, return_info=True
        )
    cinfo = {"objects": []}
    if cf:
        out, cinfo = fill_fitted_circles(out, c0, c1, cr, ca, return_info=True)
    if not return_info:
        return out
    info = {
        "settings": {
            "closing_radius": r,
            "max_hole_area": a,
            "ring_method": rm,
            "ring_closing_radius": rr,
            "ring_max_object_size": rs,
            "circle_fit": cf,
            "circle_r_min": c0,
            "circle_r_max": c1,
            "circle_max_rms": cr,
            "circle_min_arc": ca,
        },
        "regions": hinfo["regions"],
        "holes_filled": hinfo["holes_filled"],
        "holes_left_too_large": sum(x["reason"] == "too large" for x in hinfo["regions"]),
        "hole_pixels": hinfo["pixels_changed"],
        "rings": rinfo["objects"],
        "rings_completed": len(rinfo["objects"]),
        "ring_pixels": int(sum(o["added"] for o in rinfo["objects"])),
        "circles": cinfo["objects"],
        "circles_filled": len(cinfo["objects"]),
        "circle_pixels": int(sum(o["added"] for o in cinfo["objects"])),
        "pixels_changed": int((out != class_map).sum()),
    }
    return out, info


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Circle fit (step 3 of the former three-step method; radius 2.8-4.2 px, RMS <= 1 px, arc >= 270 deg). Not used by the ParticleVision pipeline (kept from earlier experiments).
# For each ring-like microplastic
# object (8-connected pieces grouped), a circle is fitted to its pixel centres (algebraic least squares, Kasa). The
# fitted disc is filled only if: r_min <= radius <= r_max, RMS distance of the pixels to the circle <= max_rms px, and
# the pixels cover >= min_arc_deg of the circle (360 deg minus the largest angular gap). Only Air / Water pixels inside
# the disc (radius + 0.5 px) become microplastic: never grain, never another object, never 4-adjacent to another
# object or to pixels added for another object. Objects larger than max_object_size, smaller than min_object_area,
# or already >= 85 % of their disc (solid) are skipped.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def fit_circle(ys: np.ndarray, xs: np.ndarray):
    """(cy, cx, radius, rms, arc_deg) of the least-squares circle through pixel centres, or None."""
    if len(ys) < 5:
        return None
    y, x = ys.astype(float), xs.astype(float)
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = -(x * x + y * y)
    try:
        (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = -D / 2, -E / 2
    r2 = cx * cx + cy * cy - F
    if not np.isfinite(r2) or r2 <= 0:
        return None
    R = float(np.sqrt(r2))
    dist = np.hypot(x - cx, y - cy)
    rms = float(np.sqrt(np.mean((dist - R) ** 2)))
    ang = np.sort(np.degrees(np.arctan2(y - cy, x - cx)))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 360]))
    return float(cy), float(cx), R, rms, float(360 - gaps.max())


def fill_fitted_circles(
    class_map: np.ndarray,
    r_min: float,
    r_max: float,
    max_rms: float = 1.0,
    min_arc_deg: float = 270,
    max_object_size: Optional[int] = None,
    min_object_area: int = 6,
    return_info: bool = False,
):
    """Fill ring-like microplastic objects with their fitted disc (see the block comment above)."""
    out = class_map.copy()
    mp = class_map == MP_CLASS
    lab, n = label(mp, structure=np.ones((3, 3), dtype=bool))
    info = {"objects": []}
    if n == 0:
        return (out, info) if return_info else out
    area = np.bincount(lab.ravel(), minlength=n + 1)
    max_size = int(max_object_size or np.ceil(4 * r_max))  # about two particle diameters
    taken = np.zeros(mp.shape, dtype=bool)
    H, W = mp.shape
    pad = int(np.ceil(r_max)) + 2
    for k, sl in enumerate(find_objects(lab), 1):
        if sl is None or area[k] < min_object_area:
            continue
        if sl[0].stop - sl[0].start > max_size or sl[1].stop - sl[1].start > max_size:
            continue
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, H)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, W)
        lc, cc, tc = lab[y0:y1, x0:x1], class_map[y0:y1, x0:x1], taken[y0:y1, x0:x1]
        own = lc == k
        ys, xs = np.nonzero(own)
        fit = fit_circle(ys, xs)
        if fit is None:
            continue
        cy, cx, R, rms, arc = fit
        if not (r_min <= R <= r_max) or rms > max_rms or arc < min_arc_deg:
            continue
        yy, xx = np.mgrid[0 : own.shape[0], 0 : own.shape[1]]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= (R + 0.5) ** 2
        if (own & disc).sum() >= 0.85 * disc.sum():
            continue  # already solid
        others = ((lc > 0) & ~own) | tc
        allowed = (
            ((cc == AIR_CLASS) | (cc == WATER_CLASS))
            & ~others
            & ~binary_dilation(others, structure=_CROSS)
        )
        new = disc & ~own & allowed
        if not new.any():
            continue
        out[y0:y1, x0:x1][new] = MP_CLASS
        taken[y0:y1, x0:x1] |= new
        if return_info:
            ny, nx = np.nonzero(new)
            info["objects"].append(
                {
                    "object_area": int(area[k]),
                    "added": int(new.sum()),
                    "radius": round(R, 2),
                    "rms": round(rms, 2),
                    "arc_deg": round(arc, 1),
                    "cy": int(ny.mean()) + y0,
                    "cx": int(nx.mean()) + x0,
                }
            )
    if return_info:
        info["pixels_changed"] = int((out != class_map).sum())
        return out, info
    return out


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# CLOSING (close_microplastic): used for 5 um particles and for images with no particle size in the folder name.
# Accepts that closely spaced particles may merge.
#   1. closed = binary closing of the microplastic mask with a disk of radius r (x^2 + y^2 <= r^2), computed on a copy
#      padded with r non-microplastic pixels (so the image edge adds nothing).
#   2. New microplastic pixels = closed & ~mp, ONLY where the original class is Air or Water (never grain).
#   3. Then every region enclosed by the new microplastic mask (background 4-connectivity) of at most max_hole_area px
#      is filled - only its Air / Water pixels. Larger enclosed regions (e.g. a bubble surrounded by particles) stay.
#   No microplastic pixel is ever removed. Optional: the radius chosen from the particle size in the folder name.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def particle_size_from_path(path: str) -> Optional[int]:
    """Particle diameter (um) from an experiment folder name such as '...-SCA-D10 (40)', or None."""
    import re

    m = re.search(r"-D(\d+)(?:\s*\(\d+\))?(?:[\\/]|$)", path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def close_microplastic(
    class_map: np.ndarray,
    radius: int,
    max_hole_area: int,
    max_added_region: Optional[int] = None,
    return_info: bool = False,
):
    """Morphological closing of the microplastic class (see the block comment above)."""
    from scipy.ndimage import binary_closing

    if class_map.ndim != 2:
        raise ValueError(f"class_map must be 2-D (H x W), got shape {class_map.shape}")
    r = int(radius)
    mp = class_map == MP_CLASS
    water_air = (class_map == AIR_CLASS) | (class_map == WATER_CLASS)
    out = class_map.copy()
    if r > 0:
        closed = binary_closing(np.pad(mp, r), structure=_disk(r))[r:-r, r:-r]
        new_close = closed & ~mp & water_air
    else:
        new_close = np.zeros_like(mp)
    mp2 = mp | new_close
    enclosed = _enclosed(mp2)
    reg, n = label(enclosed, structure=_CROSS)
    new_holes = np.zeros_like(mp)
    if n:
        area = np.bincount(reg.ravel(), minlength=n + 1)
        keep = area <= int(max_hole_area)
        keep[0] = False
        new_holes = keep[reg] & water_air
    added = new_close | new_holes
    dropped = 0
    if max_added_region is not None and added.any():
        # safeguard (default 50 px): an added patch (8-connected) larger than max_added_region px is not
        # a single particle being completed but a cluster being turned into a blob - keep it as it was
        alab, an = label(added, structure=np.ones((3, 3), dtype=bool))
        asz = np.bincount(alab.ravel(), minlength=an + 1)
        big = asz > int(max_added_region)
        big[0] = False
        dropped = int(big[1:].sum())
        added &= ~big[alab]
    out[added] = MP_CLASS
    if not return_info:
        return out
    return out, {
        "radius": r,
        "max_hole_area": int(max_hole_area),
        "closing_pixels": int((added & new_close).sum()),
        "hole_pixels": int((added & new_holes & ~new_close).sum()),
        "pixels_changed": int(added.sum()),
        "max_added_region": max_added_region,
        "added_regions_dropped": dropped,
    }


def _size(source_path: Optional[str], particle_size: Optional[int]) -> Optional[int]:
    """The particle size to use: a chosen size (particle_size) wins over the folder name."""
    if particle_size is not None:
        return int(particle_size)
    return particle_size_from_path(source_path) if source_path else None


def radius_for(
    source_path: Optional[str],
    radius_by_size: Optional[dict] = None,
    default_radius: Optional[int] = None,
    use_size: Optional[bool] = None,
    particle_size: Optional[int] = None,
):
    """(closing radius, particle size or None) for an image: the chosen particle_size, else the experiment folder name."""
    from src.config import CFG

    table = (
        CFG.postprocess_closing_radius_by_size
        if radius_by_size is None
        else {int(k): v for k, v in radius_by_size.items()}
    )
    default = CFG.postprocess_closing_default_radius if default_radius is None else default_radius
    use = CFG.postprocess_closing_use_size if use_size is None else use_size
    size = _size(source_path, particle_size)
    if use and size in table:
        return int(table[size]), size
    return int(default), size


def method_for(
    source_path: Optional[str],
    method_by_size: Optional[dict] = None,
    use_size: Optional[bool] = None,
    particle_size: Optional[int] = None,
):
    """("closing" | "circles", particle size or None) for an image: the chosen particle_size, else the experiment
    folder name. No size found (or use_size False) -> closing."""
    from src.config import CFG

    table = (
        CFG.postprocess_method_by_size
        if method_by_size is None
        else {int(k): v for k, v in method_by_size.items()}
    )
    use = CFG.postprocess_closing_use_size if use_size is None else use_size
    size = _size(source_path, particle_size)
    if use and size in table:
        return table[size], size
    return "closing", size


def postprocess_microplastic(
    class_map: np.ndarray,
    source_path: Optional[str] = None,
    radius: Optional[int] = None,
    radius_by_size: Optional[dict] = None,
    default_radius: Optional[int] = None,
    use_size: Optional[bool] = None,
    max_hole_area: Optional[int] = None,
    max_added_region: Optional[int] = None,
    method_by_size: Optional[dict] = None,
    circle_radii_by_size: Optional[dict] = None,
    circle_min_support: Optional[float] = None,
    circle_max_overlaps: Optional[int] = None,
    particle_size: Optional[int] = None,
    return_info: bool = False,
):
    """THE post-processing of the final map (version 1.1.0), chosen by the particle size in the image's
    experiment folder name (src/config.py):
      7 and 10 um -> reconstruct_circles (radius window per size, support >= 60 %, at most half the ring in grain,
                     core < 50 % red, overlap rule: a disc overlapping >= 2 other discs is not drawn);
      5 um        -> close_microplastic, r 1, holes <= 39 px, added patches <= 50 px;
      no size     -> close_microplastic, r 2, same limits.
    `particle_size` (5 / 7 / 10) overrides the size in the folder name; `radius` forces closing with that radius.
    Every caller (segmentation pipeline, Fix Segmentations page, folder tool, evaluation) uses this function."""
    from src.config import CFG

    a = CFG.postprocess_closing_max_hole_area if max_hole_area is None else max_hole_area
    c = CFG.postprocess_closing_max_added_region if max_added_region is None else max_added_region
    if radius is not None:
        method, size = "closing", _size(source_path, particle_size)
    else:
        method, size = method_for(source_path, method_by_size, use_size, particle_size)
    base = {
        "method": method,
        "particle_size_detected": size,
        "radius_used": None,
        "closing_pixels": 0,
        "hole_pixels": 0,
        "added_regions_dropped": 0,
        "circles_drawn": 0,
        "circles_skipped_overlap": 0,
    }
    if method == "circles":
        table = (
            CFG.postprocess_circle_radii_by_size
            if circle_radii_by_size is None
            else {int(k): v for k, v in circle_radii_by_size.items()}
        )
        radii = [float(x) for x in table[size]]
        margin = CFG.postprocess_circle_disc_margin
        out, cinfo = reconstruct_circles(
            class_map,
            radii,
            min_support=CFG.postprocess_circle_min_support
            if circle_min_support is None
            else circle_min_support,
            band=CFG.postprocess_circle_band,
            min_visible=CFG.postprocess_circle_min_visible,
            max_interior_fill=CFG.postprocess_circle_max_core_fill,
            nms_distance=CFG.postprocess_circle_nms_factor * (float(np.median(radii)) + margin),
            disc_margin=margin,
            max_overlaps=CFG.postprocess_circle_max_overlaps
            if circle_max_overlaps is None
            else circle_max_overlaps,
            return_info=True,
        )
        if not return_info:
            return out
        base.update(
            radius_used="/".join(f"{x:.1f}" for x in radii),
            circles_drawn=cinfo["n_circles"],
            circles_skipped_overlap=cinfo["skipped"]["overlaps"],
            pixels_changed=cinfo["pixels_changed"],
            circles=cinfo["circles"],
        )
        return out, base
    r = (
        int(radius)
        if radius is not None
        else radius_for(source_path, radius_by_size, default_radius, use_size, particle_size)[0]
    )
    out, info = close_microplastic(class_map, r, a, max_added_region=c, return_info=True)
    if not return_info:
        return out
    base.update(info)
    base.update(radius_used=r)
    return out, base


def describe_info(d: dict) -> str:
    """'size D10, circles r 3.0/3.3/3.6 px (57 drawn, 3 skipped: overlap)' / 'size D05, closing r 1' /
    'size not found, closing r 2' / 'post-processing off' - for run logs, the folder tool and the previews
    (d = a postprocess_microplastic info dict or a summary row with the same keys)."""
    sz = d.get("particle_size_detected")
    size = ("D%02d" % int(sz)) if sz not in ("", None) else "not found"
    if d.get("method") == "off":
        return f"size {size}, post-processing off"
    if d.get("method") == "circles":
        return (
            f"size {size}, circles r {d.get('radius_used')} px ({d.get('circles_drawn', 0)} drawn, "
            f"{d.get('circles_skipped_overlap', 0)} skipped: overlap)"
        )
    return f"size {size}, closing r {d.get('radius_used')}"


PARTICLE_SIZES = (
    5,
    7,
    10,
)  # the sizes a user can choose for a whole run (None = from the folder name)


class PipelinePostprocess:
    """The post-processing settings of ONE segmentation run (version 1.1.0+), and the single place where the
    pipeline applies it: app/lab_structure.run_lab_batch calls apply() for every image, whichever runner made
    the map (every backend the app can use), on the full-image class map after stitching, fusion and argmax,
    before the file is saved and the class areas are counted.
      enabled        "Fill hollow microplastic particles" (default CFG.postprocess_in_pipeline = on)
      particle_size  None = from each image's experiment folder name ("Auto"); 5 / 7 / 10 = for the whole run
      save_raw       also save the unfilled map as Fused_St_Filt_Comp_Seg_Raw.tif"""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        particle_size: Optional[int] = None,
        save_raw: Optional[bool] = None,
    ):
        from src.config import CFG

        if particle_size is not None and int(particle_size) not in PARTICLE_SIZES:
            raise ValueError(
                f"particle size must be one of {PARTICLE_SIZES} um or None (from the folder name)"
            )
        self.enabled = CFG.postprocess_in_pipeline if enabled is None else bool(enabled)
        self.particle_size = None if particle_size is None else int(particle_size)
        self.save_raw = CFG.postprocess_save_raw if save_raw is None else bool(save_raw)

    def apply(self, class_map: np.ndarray, source_path: Optional[str] = None):
        """(final class map, info). Off: the map unchanged, info["method"] == "off"."""
        if not self.enabled:
            n = int((class_map == MP_CLASS).sum())
            return class_map, {
                "method": "off",
                "particle_size_detected": _size(source_path, self.particle_size),
                "radius_used": None,
                "circles_drawn": 0,
                "circles_skipped_overlap": 0,
                "pixels_changed": 0,
                "mp_px_raw": n,
                "mp_px_final": n,
            }
        out, info = postprocess_microplastic(
            class_map, source_path=source_path, particle_size=self.particle_size, return_info=True
        )
        info.update(
            mp_px_raw=int((class_map == MP_CLASS).sum()), mp_px_final=int((out == MP_CLASS).sum())
        )
        return out, info

    def record(self) -> dict:
        """Everything that decides the output, for run_settings.json / run_log.txt and the Continue check
        (app version, on / off, the size setting, raw copy, and - when on - every parameter + the code MD5)."""
        from src.config import PARTICLEVISION_VERSION

        d = {
            "app_version": PARTICLEVISION_VERSION,
            "postprocess": "on" if self.enabled else "off",
            "particle_size_setting": "auto" if self.particle_size is None else self.particle_size,
            "save_raw": self.save_raw,
        }
        d["postprocess_settings"] = postprocess_settings()
        return d

    def describe(self) -> str:
        if not self.enabled:
            return "off (the model's map is saved as it is)"
        from src.config import CFG

        size = (
            "from each experiment folder name (Auto)"
            if self.particle_size is None
            else f"{self.particle_size} um for every image (chosen)"
        )
        radii = CFG.postprocess_circle_radii_by_size
        return (
            f"on - particle size {size}; D05 closing r {CFG.postprocess_closing_radius_by_size[5]}; "
            f"D07 circles r {'/'.join(f'{x:.1f}' for x in radii[7])} px; "
            f"D10 circles r {'/'.join(f'{x:.1f}' for x in radii[10])} px; no size found: closing r "
            f"{CFG.postprocess_closing_default_radius}; circles: support >= {CFG.postprocess_circle_min_support:.0%}, "
            f"not drawn if overlapping >= {CFG.postprocess_circle_max_overlaps} other circles; closing: holes <= "
            f"{CFG.postprocess_closing_max_hole_area} px, added patches <= {CFG.postprocess_closing_max_added_region} px"
            + (
                "; the unfilled map is also saved (Fused_St_Filt_Comp_Seg_Raw.tif)"
                if self.save_raw
                else ""
            )
        )


# ---- Circle reconstruction: the method for 7 and 10 um particles (via postprocess_microplastic) ------------------
# A Hough-style detector of ring-shaped microplastic particles of a known radius. For every candidate centre on a
# half-pixel grid and every radius r of the window, the "ring band" is the set of pixels whose centre lies at
# |distance - r| <= band. Votes = microplastic pixels in the band (accumulated sparsely: each microplastic pixel votes
# for every centre it could belong to). A centre is accepted if
#   * support  = microplastic / non-grain pixels in the band >= min_support (grain is not counted against a ring that
#                touches a grain),
#   * visible  = non-grain / all pixels in the band >= min_visible (at most half of the ring hidden in grain),
#   * at least half of the core (distance <= r - 1) is non-grain and less than max_interior_fill of it is already
#     microplastic (a ring is hollow; a solid speck or a solid blob is left alone),
# then greedy non-maximum suppression (highest support first) keeps centres at least nms_distance apart.
# Each accepted centre gets a filled disc of radius r + disc_margin; only Air / Water pixels inside it become
# microplastic (never grain, never removed), and nothing between neighbouring discs is filled.


def _circle_offsets(r: float, band: float, py: int, px: int, pad: int):
    """Integer pixel offsets (dy, dx) from grid point (i, j) for a centre at (i + py/2, j + px/2): ring band, core."""
    d = np.arange(-pad, pad + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    dist = np.hypot(dy - py / 2, dx - px / 2)
    ring = np.abs(dist - r) <= band
    inner = dist <= r - 1.0  # the core, clear of the ring line
    return (dy[ring], dx[ring]), (dy[inner], dx[inner])


def reconstruct_circles(
    class_map: np.ndarray,
    radii,
    min_support: float = 0.5,
    band: float = 0.5,
    min_visible: float = 0.5,
    max_interior_fill: float = 0.5,
    nms_distance: Optional[float] = None,
    disc_margin: float = 0.5,
    max_overlaps: Optional[int] = None,
    max_mixed: Optional[float] = None,
    mixed_window: int = 5,
    return_info: bool = False,
):
    """Detect ring-shaped microplastic particles with radius in `radii` (px, ring centre line) and draw each one as a
    filled disc (see the block comment above). Evaluation only."""
    radii = [float(r) for r in radii]
    out = class_map.copy()
    H, W = class_map.shape
    pad = int(np.ceil(max(radii) + max(band, disc_margin))) + 2
    cp = np.pad(
        class_map, pad, constant_values=0
    )  # outside the image counts as grain (not visible)
    mp, ng = cp == MP_CLASS, cp != 0
    Hp, Wp = cp.shape
    ys, xs = np.nonzero(mp)
    D = (
        float(nms_distance)
        if nms_distance is not None
        else 1.5 * (float(np.median(radii)) + disc_margin)
    )
    cand = []  # (support, votes, cy, cx, r, visible, interior_fill)
    for py in (0, 1):
        for px in (0, 1):
            for r in radii:
                (ry, rx), (iy, ix) = _circle_offsets(r, band, py, px, pad - 1)
                n_band = len(ry)
                if n_band == 0 or len(ys) == 0:
                    continue
                thr = int(np.ceil(min_support * min_visible * n_band))
                idx = ((ys[:, None] - ry[None, :]) * Wp + (xs[:, None] - rx[None, :])).ravel()
                idx = idx[(idx >= 0) & (idx < Hp * Wp)]
                votes = np.bincount(idx, minlength=Hp * Wp)
                c = np.flatnonzero(votes >= max(thr, 1))
                if len(c) == 0:
                    continue
                ci, cj = c // Wp, c % Wp
                ok = (ci >= pad) & (ci < Hp - pad) & (cj >= pad) & (cj < Wp - pad)
                c, ci, cj = c[ok], ci[ok], cj[ok]
                v = votes[c]
                vis_n = ng[ci[:, None] + ry[None, :], cj[:, None] + rx[None, :]].sum(1)
                sup = v / np.maximum(vis_n, 1)
                vis = vis_n / n_band
                keep = (sup >= min_support) & (vis >= min_visible)
                if len(iy):
                    in_ng = ng[ci[:, None] + iy[None, :], cj[:, None] + ix[None, :]]
                    in_mp = mp[ci[:, None] + iy[None, :], cj[:, None] + ix[None, :]]
                    n_ng = in_ng.sum(1)
                    fill = in_mp.sum(1) / np.maximum(n_ng, 1)
                    keep &= (n_ng >= 0.5 * len(iy)) & (fill < max_interior_fill)
                else:
                    fill = np.zeros(len(c))
                for k in np.flatnonzero(keep):
                    cand.append(
                        (
                            float(sup[k]),
                            int(v[k]),
                            ci[k] - pad + py / 2,
                            cj[k] - pad + px / 2,
                            r,
                            float(vis[k]),
                            float(fill[k]),
                        )
                    )
    cand.sort(key=lambda t: (-t[0], -t[1]))
    cell = max(D, 1.0)
    grid, accepted = {}, []
    for t in cand:
        cy, cx = t[2], t[3]
        gy, gx = int(cy // cell), int(cx // cell)
        if any(
            (cy - a[2]) ** 2 + (cx - a[3]) ** 2 < D * D
            for oy in (-1, 0, 1)
            for ox in (-1, 0, 1)
            for a in grid.get((gy + oy, gx + ox), ())
        ):
            continue
        grid.setdefault((gy, gx), []).append(t)
        accepted.append(t)
    aw = (class_map == 1) | (class_map == 2)
    # optional safeguards against dense mixed clusters (the circle is then not drawn; that place stays as it was):
    #   max_overlaps: skip a circle whose disc overlaps max_overlaps or more other accepted discs
    #   max_mixed:    skip a circle if more than max_mixed of its disc is "mixed" - a non-grain pixel whose
    #                 mixed_window x mixed_window neighbourhood holds Air / Water and >= 2 separate red fragments
    #                 (8-connected), i.e. red and Air / Water stripes rather than one ring
    n_over = np.zeros(len(accepted), dtype=int)
    if max_overlaps is not None and accepted:
        P = np.array([(a[2], a[3], a[4] + disc_margin) for a in accepted])
        for i in range(len(P)):
            d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
            n_over[i] = int(((d < P[:, 2] + P[i, 2]) & (d > 0)).sum())
    mixed = None
    if max_mixed is not None and accepted:
        from scipy.ndimage import maximum_filter, minimum_filter

        flab, _ = label(class_map == MP_CLASS, structure=np.ones((3, 3), dtype=bool))
        big = np.iinfo(flab.dtype).max
        hi = maximum_filter(flab, size=mixed_window)
        lo = minimum_filter(np.where(flab > 0, flab, big), size=mixed_window)
        mixed = (
            (hi > 0)
            & (lo < big)
            & (hi != lo)
            & maximum_filter(aw, size=mixed_window)
            & (class_map != 0)
        )
    circles, skipped, skipped_list = [], {"overlaps": 0, "mixed": 0}, []
    for n_i, (sup, v, cy, cx, r, vis, fill) in enumerate(accepted):
        R = r + disc_margin
        y0, y1 = max(int(np.floor(cy - R)), 0), min(int(np.ceil(cy + R)) + 1, H)
        x0, x1 = max(int(np.floor(cx - R)), 0), min(int(np.ceil(cx + R)) + 1, W)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = np.hypot(yy - cy, xx - cx) <= R
        mix = float(mixed[y0:y1, x0:x1][disc].mean()) if mixed is not None else 0.0
        if max_overlaps is not None and n_over[n_i] >= max_overlaps:
            skipped["overlaps"] += 1
            skipped_list.append(("overlaps", cy, cx))
            continue
        if max_mixed is not None and mix > max_mixed:
            skipped["mixed"] += 1
            skipped_list.append(("mixed", cy, cx))
            continue
        add = disc & aw[y0:y1, x0:x1] & (out[y0:y1, x0:x1] != MP_CLASS)
        out[y0:y1, x0:x1][add] = MP_CLASS
        circles.append(
            {
                "cy": cy,
                "cx": cx,
                "r": r,
                "support": sup,
                "votes": v,
                "visible": vis,
                "interior_fill": fill,
                "overlaps": int(n_over[n_i]),
                "mixed": mix,
                "added": int(add.sum()),
            }
        )
    if not return_info:
        return out
    return out, {
        "circles": circles,
        "n_circles": len(circles),
        "candidates": len(cand),
        "skipped": skipped,
        "skipped_list": skipped_list,
        "pixels_changed": int((out != class_map).sum()),
    }
