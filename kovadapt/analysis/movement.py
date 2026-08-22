"""Movement analysis: flick segmentation and aggregate metrics.

All computation is vectorized numpy over uniform-grid velocities; a full
run's trace (~10^5 packets) analyzes in a few milliseconds.

Model: aimed movements are click-anchored. For each left click we look back
up to `lookback` seconds, find the movement onset (speed rising through a
fraction of the segment peak), and characterize the flick:

  amplitude      net displacement (counts) from onset to click
  angle          direction of net displacement (radians, aim convention: +y up)
  peak_speed     max speed on the segment (counts/s)
  time_to_peak   onset -> peak speed (s)
  overshoot      how far past the endpoint the path traveled along the flick
                 axis, as a fraction of amplitude (classic flick overshoot)
  corrections    number of corrective submovements after the peak (speed
                 valleys followed by re-acceleration)
  duration       onset -> click (s)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..telemetry.trace import MouseTrace, ResampleCache


@dataclass(slots=True)
class Flick:
    t_click: float
    t_onset: float
    duration: float
    amplitude: float
    angle: float            # radians, 0 = right, pi/2 = up
    peak_speed: float
    time_to_peak: float
    overshoot: float        # fraction of amplitude, >= 0
    corrections: int
    # 1-based index in the trace's original click stream.  This deliberately
    # counts clicks that were too small to become Flick objects, so a report's
    # "click 17" always points at the same red x in the replay.
    click_index: int = 0
    # Linked from KovaaK's per-kill shot counts when the scenario exposes a
    # one-hit acquisition.  None means the stats could not safely identify
    # this click; False is a real miss, not a low-quality heuristic.
    hit: bool | None = None
    # Phase metrics.  A correction begins at the first post-peak low-speed
    # valley (<15% of peak) that is followed by re-acceleration (>35%).
    # Speeds remain in counts/s here; reports carry the run's deg/count for
    # an honest conversion to view-angle speed at render time.
    primary_mean_speed: float = 0.0
    micro_adjust_speed: float = 0.0
    brake_ms: float = 0.0
    transition_slowdown: float = 0.0
    # primary_only | smooth_terminal | discrete_adjust | repair_chain.
    # "primary_only" means no secondary kinematic phase was detected; it
    # deliberately does not claim that target-centre video would reveal no
    # tiny visual correction.
    phase_kind: str = "primary_only"

    @property
    def horizontal(self) -> str:
        c = np.cos(self.angle)
        return "right" if c > 0.25 else ("left" if c < -0.25 else "vertical")


def _smooth(v: np.ndarray, rate: float, tau_ms: float = 8.0) -> np.ndarray:
    """Causal exponential moving average (cheap, phase-lag ~tau).

    Vectorized as a truncated exponential FIR plus the acc=v[0] initial-
    condition term; the kernel is cut where its weight reaches 1e-9, so the
    deviation from the exact recurrence is far below the onset/hysteresis
    thresholds downstream.
    """
    if v.size == 0:
        return v
    a = 1.0 - np.exp(-1.0 / (rate * tau_ms / 1000.0))
    n = v.size
    klen = min(n, int(np.ceil(np.log(1e-9) / np.log(1.0 - a))))
    k = a * (1.0 - a) ** np.arange(klen)
    decay = (1.0 - a) ** (np.arange(n, dtype=np.float64) + 1.0)
    return np.convolve(np.asarray(v, dtype=np.float64), k)[:n] + v[0] * decay


#: Smallest movement that counts as a FLICK, as an ANGLE. One degree, and the
#: number is measured rather than chosen.
#:
#: Overshoot is a FRACTION of amplitude, so a small amplitude with a small
#: absolute error produces an enormous ratio, and a mis-segmented onset
#: underestimates amplitude directly. Below about a degree the statistic stops
#: measuring aim and starts measuring segmentation error. `directional_bias`
#: takes a plain mean, so a handful of those artefacts outvote hundreds of real
#: flicks — and that verdict is what writes Left/RightStrafeTimeMult into the
#: generated .sce.
#:
#: Swept over the five real traces here, max overshoot per bucket against the
#: floor that would exclude it:
#:
#:     0.34 deg  17.42        0.84 deg   3.86
#:     0.50 deg  16.83        1.01 deg   0.68   <- the cliff
#:     0.67 deg   3.86        1.34 deg   0.68
#:
#: It is flat above the knee, so a higher floor discards real flicks and buys
#: nothing: 2.0 degrees would drop ~25 more of the 600 for no change in the
#: statistic.
#:
#: THIS SHIPPED AS 2.0 IN v0.5.2 AND THE LABEL WAS WRONG, not the behaviour.
#: The knee was measured in COUNTS (90.9) — that part never passed through a
#: conversion and still holds — then written down in degrees using
#: `YAW_DEG_PER_COUNT * game_sens`, where `game_sens` was the untouched default
#: of 1.0 and the real setting was the Valorant scale at 0.16. Those differ by
#: 1.96x, so a floor honestly measured at 1.01 degrees was recorded as 2.0.
#: The lesson is in `sens.deg_per_count`: KovaaK's writes the scale, the
#: sensitivity and the DPI into every stats file, and the app was asking a
#: settings field that nobody had ever set.
MIN_FLICK_DEG = 1.0
#: KovaaK's (Quake/Source lineage) yaw: degrees turned per mouse count at
#: in-game sensitivity 1.0. Defined here rather than in `sens` because this is
#: the leaf both need and `sens` imports `report`, which imports this — a cycle
#: the other way round. `sens.YAW_DEG_PER_COUNT` re-exports it.
YAW_DEG_PER_COUNT = 0.022
#: The floor in COUNTS at the KovaaK's-native scale, sens 1.0 — the reference
#: `sens.min_flick_counts_for` falls back to when a run's scale cannot be
#: resolved. It does NOT depend on DPI: a count is a count, and DPI decides how
#: much desk a count costs, not how far the view turns. Sens and SCALE decide
#: that, which is why the fallback is a last resort rather than a default.
MIN_FLICK_COUNTS = MIN_FLICK_DEG / YAW_DEG_PER_COUNT   # 45.5 at sens 1.0


def segment_flicks(
    trace: MouseTrace,
    rate: float = 500.0,
    lookback: float = 0.8,
    onset_frac: float = 0.08,
    min_amplitude: float = MIN_FLICK_COUNTS,
    *,
    grid: ResampleCache | None = None,
) -> list[Flick]:
    # `grid` (a ResampleCache over `trace`) lets build_report share one
    # resample across analysis passes; bare-trace calls behave identically.
    tg, vx, vy = (grid if grid is not None else trace).resample(rate)
    if tg.size == 0 or trace.clicks.size == 0:
        return []
    vy = -vy  # aim convention: +y up
    vxs, vys = _smooth(vx, rate), _smooth(vy, rate)
    speed = np.hypot(vxs, vys)
    x = np.concatenate([[0.0], np.cumsum(vxs)]) / rate  # position (counts)
    y = np.concatenate([[0.0], np.cumsum(vys)]) / rate

    flicks: list[Flick] = []
    n = tg.size
    for k, tc in enumerate(trace.clicks):
        ic = int(np.searchsorted(tg, tc))
        if ic <= 1:
            continue
        ic = min(ic, n)
        # Never look back past the previous click: at real kill pacing
        # (<0.8s between kills) the window would swallow the previous flick.
        lo = tc - lookback
        if k > 0:
            lo = max(lo, float(trace.clicks[k - 1]))
        i0 = max(int(np.searchsorted(tg, lo)), 0)
        seg = speed[i0:ic]
        if seg.size < 4:
            continue
        ipk_rel = int(np.argmax(seg))
        pk = seg[ipk_rel]
        if pk <= 0:
            continue
        # onset: last sample before the peak with speed below onset_frac*peak
        below = np.nonzero(seg[: ipk_rel + 1] < onset_frac * pk)[0]
        ion = i0 + (int(below[-1]) if below.size else 0)
        ipk = i0 + ipk_rel

        dx_net = x[ic] - x[ion]
        dy_net = y[ic] - y[ion]
        amplitude = float(np.hypot(dx_net, dy_net))
        if amplitude < min_amplitude:
            continue
        ux, uy = dx_net / amplitude, dy_net / amplitude

        # overshoot: max projection along flick axis beyond the click point
        proj = (x[ion:ic + 1] - x[ion]) * ux + (y[ion:ic + 1] - y[ion]) * uy
        overshoot = float(max(proj.max() - amplitude, 0.0) / amplitude)

        # corrective submovements after the peak: speed collapses (below 15%
        # of peak), then re-accelerates (above 35%). Hysteresis rejects the
        # sample-level jitter of integer mouse counts.
        post = speed[ipk:ic]
        corrections = 0
        primary_end = ic
        phase_kind = "primary_only"
        if post.size >= 3:
            m = np.where(post < 0.15 * pk, -1, np.where(post > 0.35 * pk, 1, 0))
            nz = np.flatnonzero(m)
            states = m[nz]
            if states.size >= 2:
                transitions = np.flatnonzero(
                    (states[:-1] == -1) & (states[1:] == 1))
                corrections = int(transitions.size)
                if transitions.size:
                    first = int(transitions[0])
                    primary_end = ipk + int(nz[first])
                    phase_kind = ("repair_chain" if corrections >= 2
                                  else "discrete_adjust")

        # A secondary phase need not fall below 15% of peak and then produce
        # a separate speed peak.  Smooth terminal control can appear as a
        # shallower re-acceleration or a direction change while the movement
        # keeps decelerating.  Detect both, but only after peak speed and only
        # when the terminal displacement is large enough to survive integer
        # mouse-count noise.  This is a kinematic phase label, not a claim
        # about target-centre error (Raw Input has no target pixels).
        if phase_kind == "primary_only" and post.size >= 6:
            smooth_rel: int | None = None

            # Type-2-like secondary submovement: the deceleration turns back
            # into acceleration without reaching the stronger 15/35% valley
            # used by the discrete-correction counter.
            ds = np.diff(post)
            troughs = np.flatnonzero((ds[:-1] < 0) & (ds[1:] > 0)) + 1
            for trough in troughs:
                level = float(post[trough] / pk)
                rebound = (float(np.max(post[trough + 1:]) - post[trough])
                           if trough + 1 < post.size else 0.0)
                if 0.15 <= level <= 0.65 and rebound >= 0.06 * pk:
                    smooth_rel = int(trough)
                    break

            # Curved terminal homing: compare the velocity around peak with
            # the displacement of the low-speed tail. Seven degrees is above
            # the angular quantisation of a >=5-count tail while remaining
            # sensitive to the narrow wrist/finger adjustment static clicking
            # is meant to expose.
            below_tail = np.flatnonzero(post <= 0.35 * pk)
            tail_rel = int(below_tail[0]) if below_tail.size else -1
            if tail_rel >= 0 and ipk + tail_rel < ic:
                tail_i = ipk + tail_rel
                tail_dx = x[ic] - x[tail_i]
                tail_dy = y[ic] - y[tail_i]
                tail_amp = float(np.hypot(tail_dx, tail_dy))
                p0, p1 = max(ipk - 2, ion), min(ipk + 3, ic)
                pvx = float(np.mean(vxs[p0:p1]))
                pvy = float(np.mean(vys[p0:p1]))
                pnorm = float(np.hypot(pvx, pvy))
                if tail_amp >= max(5.0, 0.03 * amplitude) and pnorm > 0:
                    cosine = np.clip(
                        (pvx * tail_dx + pvy * tail_dy) / (pnorm * tail_amp),
                        -1.0, 1.0)
                    turn_deg = float(np.degrees(np.arccos(cosine)))
                    if turn_deg >= 10.0:
                        smooth_rel = (tail_rel if smooth_rel is None
                                      else min(smooth_rel, tail_rel))

            if smooth_rel is not None:
                primary_end = ipk + smooth_rel
                phase_kind = "smooth_terminal"

        primary_slice = speed[ion:max(primary_end + 1, ion + 1)]
        primary_mean = float(np.mean(primary_slice)) if primary_slice.size else 0.0
        micro_mean = 0.0
        brake_ms = 0.0
        slowdown = 0.0
        if phase_kind != "primary_only" and primary_mean > 0:
            # Include the valley in the adjustment phase: the controlled
            # deceleration is part of the transition, not dead time to hide.
            adjust = speed[primary_end:ic]
            micro_mean = float(np.mean(adjust)) if adjust.size else 0.0
            brake_ms = max(float(tg[primary_end] - tg[ipk]) * 1000.0, 0.0)
            slowdown = float(np.clip(1.0 - micro_mean / primary_mean, -1.0, 1.0))

        flicks.append(
            Flick(
                t_click=float(tc),
                t_onset=float(tg[ion]),
                duration=float(tc - tg[ion]),
                amplitude=amplitude,
                angle=float(np.arctan2(dy_net, dx_net)),
                peak_speed=float(pk),
                time_to_peak=float(tg[ipk] - tg[ion]),
                overshoot=overshoot,
                corrections=corrections,
                click_index=k + 1,
                primary_mean_speed=primary_mean,
                micro_adjust_speed=micro_mean,
                brake_ms=brake_ms,
                transition_slowdown=slowdown,
                phase_kind=phase_kind,
            )
        )
    return flicks


# ---------------------------------------------------------------- aggregates
def directional_bias(flicks: list[Flick]) -> dict:
    """Left/right (and vertical) split of flick quality.

    bias_score in [-1, 1]: positive = *left* flicks are worse (higher
    overshoot + more corrections), i.e. left is the weak side.
    """
    def agg(fs: list[Flick]) -> dict:
        if not fs:
            return {"n": 0, "overshoot": 0.0, "corrections": 0.0,
                    "peak_speed": 0.0, "duration": 0.0}
        return {
            "n": len(fs),
            "overshoot": float(np.mean([f.overshoot for f in fs])),
            "corrections": float(np.mean([f.corrections for f in fs])),
            "peak_speed": float(np.mean([f.peak_speed for f in fs])),
            "duration": float(np.mean([f.duration for f in fs])),
        }

    left = agg([f for f in flicks if f.horizontal == "left"])
    right = agg([f for f in flicks if f.horizontal == "right"])
    vertical = agg([f for f in flicks if f.horizontal == "vertical"])

    def badness(a: dict) -> float:
        return a["overshoot"] + 0.15 * a["corrections"]

    score = 0.0
    if left["n"] >= 3 and right["n"] >= 3:
        b_l, b_r = badness(left), badness(right)
        denom = b_l + b_r
        score = float((b_l - b_r) / denom) if denom > 0 else 0.0
    return {"left": left, "right": right, "vertical": vertical, "bias_score": score}


def apply_shot_outcomes(flicks: list[Flick], outcomes: list[dict],
                        tolerance: float = 0.010) -> None:
    """Stamp known per-click hit results onto segmented flicks.

    Both timestamps originate in the Raw Input trace, so this is normally an
    exact match.  The small tolerance keeps old JSON round-trips and float
    formatting from silently dropping a label; it is far shorter than any
    plausible interval between deliberate static-click shots.
    """
    known = sorted(
        ((float(o["t_click"]), bool(o["hit"])) for o in outcomes
         if "t_click" in o and "hit" in o),
        key=lambda item: item[0],
    )
    if not known:
        return
    times = np.asarray([item[0] for item in known], dtype=np.float64)
    for flick in flicks:
        i = int(np.searchsorted(times, flick.t_click))
        candidates = [j for j in (i - 1, i) if 0 <= j < len(known)]
        if not candidates:
            continue
        best = min(candidates, key=lambda j: abs(known[j][0] - flick.t_click))
        if abs(known[best][0] - flick.t_click) <= tolerance:
            flick.hit = known[best][1]


def click_phase_metrics(flicks: list[Flick]) -> dict:
    """Outcome-aware static-click technique digest.

    This deliberately describes what the trace proves, not target geometry it
    cannot see.  In particular, ``uncorrected_miss`` means the player fired a
    shot KovaaK's marked as a miss without a detectable corrective
    submovement after peak speed.  It does not pretend to know the target's
    pixel-space centre.
    """
    labeled = [f for f in flicks if f.hit is not None]
    if not labeled:
        return {}
    hits = [f for f in labeled if f.hit]
    misses = [f for f in labeled if not f.hit]
    primary_hits = sum(f.phase_kind == "primary_only" for f in hits)
    smooth_hits = sum(f.phase_kind == "smooth_terminal" for f in hits)
    discrete_hits = sum(f.phase_kind == "discrete_adjust" for f in hits)
    repair_hits = sum(f.phase_kind == "repair_chain" for f in hits)
    uncorrected_misses = sum(f.phase_kind == "primary_only" for f in misses)
    corrected_misses = len(misses) - uncorrected_misses
    transitions = [f for f in labeled
                   if f.phase_kind != "primary_only" and f.primary_mean_speed > 0
                   and f.micro_adjust_speed > 0]
    primary_speeds = [f.primary_mean_speed for f in labeled
                      if f.primary_mean_speed > 0]
    def click_indexes(items: list[Flick]) -> list[int]:
        return [f.click_index for f in items if f.click_index > 0]
    primary_hit_items = [f for f in hits if f.phase_kind == "primary_only"]
    smooth_hit_items = [f for f in hits if f.phase_kind == "smooth_terminal"]
    discrete_hit_items = [f for f in hits if f.phase_kind == "discrete_adjust"]
    repair_hit_items = [f for f in hits if f.phase_kind == "repair_chain"]
    uncorrected_miss_items = [f for f in misses
                              if f.phase_kind == "primary_only"]
    corrected_miss_items = [f for f in misses
                            if f.phase_kind != "primary_only"]
    return {
        "labeled": len(labeled),
        "hits": len(hits),
        "misses": len(misses),
        "linked_hit_rate": len(hits) / len(labeled),
        # Backward-compatible aliases: old reports/UI called primary-only
        # hits "direct" and all one-phase corrections "corrected". New code
        # reads the explicit four-way fields below.
        "direct_hits": primary_hits,
        "direct_hit_rate": primary_hits / len(hits) if hits else 0.0,
        "corrected_hits": smooth_hits + discrete_hits,
        "primary_only_hits": primary_hits,
        "primary_only_hit_rate": primary_hits / len(hits) if hits else 0.0,
        "smooth_terminal_hits": smooth_hits,
        "discrete_adjust_hits": discrete_hits,
        "repair_chain_hits": repair_hits,
        "uncorrected_misses": uncorrected_misses,
        "corrected_misses": corrected_misses,
        "uncorrected_miss_rate": uncorrected_misses / len(labeled),
        "uncorrected_share_of_misses": (
            uncorrected_misses / len(misses) if misses else 0.0),
        # Concrete click references let every aggregate conclusion be checked
        # against the trajectory.  Old reports simply lack these optional
        # lists and continue to render their counts as before.
        "labeled_clicks": click_indexes(labeled),
        "hit_clicks": click_indexes(hits),
        "miss_clicks": click_indexes(misses),
        "primary_only_hit_clicks": click_indexes(primary_hit_items),
        "smooth_terminal_hit_clicks": click_indexes(smooth_hit_items),
        "discrete_adjust_hit_clicks": click_indexes(discrete_hit_items),
        "repair_chain_hit_clicks": click_indexes(repair_hit_items),
        "uncorrected_miss_clicks": click_indexes(uncorrected_miss_items),
        "corrected_miss_clicks": click_indexes(corrected_miss_items),
        "transition_clicks": click_indexes(transitions),
        "mean_peak_speed_counts_s": float(np.mean(
            [f.peak_speed for f in labeled])),
        "mean_primary_speed_counts_s": (
            float(np.mean(primary_speeds)) if primary_speeds else 0.0),
        "transition_samples": len(transitions),
        "mean_micro_adjust_speed_counts_s": (
            float(np.mean([f.micro_adjust_speed for f in transitions]))
            if transitions else 0.0),
        "mean_brake_ms": (float(np.mean([f.brake_ms for f in transitions]))
                          if transitions else 0.0),
        "mean_transition_slowdown": (
            float(np.mean([f.transition_slowdown for f in transitions]))
            if transitions else 0.0),
    }


def region_deficits(flicks: list[Flick], cols: int = 3, rows: int = 3) -> dict[str, float]:
    """Per-region weakness signal from real flick data.

    Direction AND amplitude aware: each flick credits the wall region it was
    aimed toward, at a ring distance from the grid center set by its
    amplitude relative to the run's own flick-amplitude distribution
    (amplitude / p90, clipped to 1). Short flicks credit the inner cells,
    long flicks the edges — skill decays with distance from the mouse rest
    position, so large-angle weakness lives at the extremes (analysis/kb.py:
    p-rest-position, dx-region-deficit). The unit direction is stretched
    onto the square (divided by max(|ux|, |uy|)) so diagonal flicks can
    reach the corner cells at any grid size.

    Deficit per region = mean (overshoot + correction penalty + slowness
    penalty) over regions with >= 2 observations, z-scored across regions.
    Feeds the bandit as *observed* rewards, replacing (or blending with)
    run-level attribution. Keys follow the r{row}c{col} cross-module
    contract in aim convention (+y up: an upward flick credits a higher row).
    """
    if not flicks:
        return {}
    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
    buckets: dict[str, list[float]] = {}
    durs = np.array([f.duration for f in flicks])
    med_dur = float(np.median(durs))
    ref_amp = max(float(np.percentile([f.amplitude for f in flicks], 90)), 1e-9)
    for f in flicks:
        # direction stretched onto the square, scaled by the amplitude ring
        ux, uy = np.cos(f.angle), np.sin(f.angle)
        m = max(abs(ux), abs(uy), 1e-9)
        ring = min(f.amplitude / ref_amp, 1.0)
        c = int(np.clip(round(center_c + (ux / m) * ring * center_c), 0, cols - 1))
        r = int(np.clip(round(center_r + (uy / m) * ring * center_r), 0, rows - 1))
        slowness = max(f.duration / med_dur - 1.0, 0.0) if med_dur > 0 else 0.0
        # A known miss is direct outcome evidence.  Without this term a shot
        # fired off-target with zero correction scored as a perfect movement
        # and taught the region bandit that the missed area was a strength.
        # 0.50 is an editorial calibration: large enough to survive the
        # within-run z-score, smaller than a severe overshoot/repair chain.
        miss_penalty = 0.50 if f.hit is False else 0.0
        buckets.setdefault(f"r{r}c{c}", []).append(
            f.overshoot + 0.15 * f.corrections + 0.25 * slowness
            + miss_penalty
        )
    means = {k: float(np.mean(v)) for k, v in buckets.items() if len(v) >= 2}
    if not means:
        return {}
    vals = np.array(list(means.values()))
    mu, sd = vals.mean(), vals.std() or 1.0
    return {k: float((v - mu) / sd) for k, v in means.items()}


def movement_heatmap(
    trace: MouseTrace,
    bins: int = 64,
    rate: float = 250.0,
    *,
    grid: ResampleCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D occupancy histogram of crosshair displacement (recentered every
    click, so it shows where your aim travels relative to each engagement)."""
    tg, vx, vy = (grid if grid is not None else trace).resample(rate)
    if tg.size == 0:
        z = np.zeros((bins, bins))
        e = np.linspace(-1, 1, bins + 1)
        return z, e, e
    x = np.cumsum(vx) / rate
    y = np.cumsum(-vy) / rate
    # recenter at each click
    anchors = np.searchsorted(tg, trace.clicks)
    offset_x, offset_y = np.zeros_like(x), np.zeros_like(y)
    last = 0
    ox = oy = 0.0
    for a in anchors:
        a = min(int(a), x.size - 1)
        offset_x[last:a] = ox
        offset_y[last:a] = oy
        ox, oy = x[a], y[a]
        last = a
    offset_x[last:] = ox
    offset_y[last:] = oy
    rx, ry = x - offset_x, y - offset_y
    lim = max(float(np.percentile(np.abs(np.concatenate([rx, ry])), 99)), 1.0)
    hist, xe, ye = np.histogram2d(
        rx, ry, bins=bins, range=[[-lim, lim], [-lim, lim]]
    )
    return hist, xe, ye
