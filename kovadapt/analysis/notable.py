"""Notable-moment detection: the clips-worthy events of a run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .movement import Flick

NOTABLE_PER_KIND = 10


@dataclass(slots=True)
class NotableMoment:
    t_start: float          # epoch seconds (pad applied for clip extraction)
    t_end: float
    kind: str               # overshoot | hesitation | unconfirmed_miss | slow_flick | clean_flick
    severity: float         # 0..1 within this run
    text: str               # plain-language description for the UI
    click_index: int = 0    # 1-based click number shared with the replay


def _fmt_dir(f: Flick) -> str:
    octants = ["right", "up-right", "up", "up-left", "left",
               "down-left", "down", "down-right"]
    return octants[int(round(f.angle / (np.pi / 4))) % 8]


def find_notable_moments(
    flicks: list[Flick], top_k: int = NOTABLE_PER_KIND, pad: float = 1.0
) -> list[NotableMoment]:
    """Flag the most instructive moments: worst overshoots, worst hesitations
    (correction chains), misses without terminal control, slowest flicks
    relative to amplitude, and clean flicks as positive references. Each
    category contributes at most ``top_k`` independently."""
    if not flicks:
        return []
    out: list[NotableMoment] = []

    def add(f: Flick, kind: str, severity: float, text: str) -> None:
        out.append(NotableMoment(f.t_onset - pad, f.t_click + pad, kind,
                                 float(np.clip(severity, 0, 1)), text,
                                 f.click_index))

    # Worst overshoots
    by_os = sorted(flicks, key=lambda f: f.overshoot, reverse=True)
    os_max = by_os[0].overshoot or 1.0
    for f in by_os[:top_k]:
        if f.overshoot < 0.15:
            break
        add(f, "overshoot", f.overshoot / os_max,
            f"Overshot a {_fmt_dir(f)} flick by {f.overshoot:.0%} of its distance, "
            f"then corrected {f.corrections}x before shooting.")

    # Worst hesitations (many corrective submovements)
    by_corr = sorted(flicks, key=lambda f: (f.corrections, f.duration), reverse=True)
    c_max = max(by_corr[0].corrections, 1)
    for f in by_corr[:top_k]:
        if f.corrections < 3:
            break
        add(f, "hesitation", f.corrections / (c_max + 1),
            f"Hesitated on a {_fmt_dir(f)} target: {f.corrections} micro-corrections "
            f"over {f.duration * 1000:.0f}ms before committing.")

    # Misses fired straight off the primary movement. These were previously
    # indistinguishable from a clean direct hit and could even be selected as
    # the positive benchmark below. Per-shot labels come only from one-hit
    # KovaaK's acquisitions; unknown outcomes are intentionally ignored.
    raw_misses = [f for f in flicks
                  if f.hit is False and f.phase_kind == "primary_only"]
    by_unconfirmed = sorted(
        raw_misses, key=lambda f: (f.overshoot, f.duration), reverse=True)
    denom = max(len(raw_misses), 1)
    for rank, f in enumerate(by_unconfirmed[:top_k]):
        add(f, "unconfirmed_miss", 1.0 - rank / (denom + 1),
            f"Missed a {_fmt_dir(f)} flick with no detected terminal-control "
            "phase before firing.")

    # Slowest flicks for their size (duration normalized by sqrt amplitude ~ Fitts)
    if len(flicks) >= 5:
        norm = np.array([f.duration / np.sqrt(max(f.amplitude, 1.0)) for f in flicks])
        thresh = float(np.percentile(norm, 90))
        idx = np.argsort(norm)[::-1][:top_k]
        for i in idx:
            f = flicks[int(i)]
            if norm[int(i)] <= thresh:
                break
            add(f, "slow_flick", float(norm[int(i)] / norm[int(idx[0])]),
                f"Slow {_fmt_dir(f)} acquisition: {f.duration * 1000:.0f}ms for a "
                f"{f.amplitude:.0f}-count flick (bottom 10% of this run's pace).")

    # Clean reference flicks. Keep the largest movements: they are the hardest
    # clean executions to reproduce, and therefore the most useful positive
    # comparison set. This category follows the same per-kind limit as every
    # problem category instead of being silently hard-coded to one item.
    clean = [f for f in flicks
             if f.hit is not False and f.overshoot < 0.05 and f.corrections <= 1]
    if clean:
        by_amplitude = sorted(clean, key=lambda f: f.amplitude, reverse=True)
        largest = max(by_amplitude[0].amplitude, 1.0)
        for f in by_amplitude[:top_k]:
            add(f, "clean_flick", f.amplitude / largest,
                f"Reference: a clean {f.amplitude:.0f}-count {_fmt_dir(f)} flick — "
                f"{f.duration * 1000:.0f}ms, no overshoot. This is your benchmark.")

    out.sort(key=lambda m: m.severity, reverse=True)
    return out
