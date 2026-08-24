"""Trajectory replay: animated crosshair path with flick-quality overlays.

Visual language (all derived from the recorded MouseTrace — no video):

    blue→orange line  instantaneous mouse speed, slow→fast
    green halo        clean flicks (low overshoot, <= 1 correction)
    red halo          flawed flicks (overshoot > 10% or >= 2 corrections)
    red ✕             shots (left clicks)
    bright dot+trail  playhead sweeping in (scaled) real time
    numbered red x    1-based click index shared with analysis conclusions

The path/flicks/shots checkboxes in the control bar hide layers without
touching the item architecture — they only flip setVisible on the items.

Lightweight by construction: the overlays are exactly two PlotCurveItems
regardless of flick count (NaN-separated segments), the path is uniformly
reduced to at most 50k points, and its colour bands are assembled with NumPy
rather than a Python loop per segment. On a real desktop the plot gets its own
OpenGL viewport; path and flick vectors are also raster-cached after range or
theme changes, so playback does not redraw 50k antialiased segments per frame.
The 125 Hz playhead moves one cached graphics item; the more expensive trail
and slider refresh at 30 Hz. Playback time comes from QElapsedTimer, so speed
is wall-clock accurate under load.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QElapsedTimer, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication, QPixmap, QTransform
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..telemetry.trace import MouseTrace
from . import theme
from .i18n import tr

_MAX_POINTS = 50_000          # decimation cap for the drawn path
_TRAIL_SECONDS = 1.2          # live comet-trail length (full path stays dim below)
_SLIDER_STEPS = 1000
_PLAYHEAD_INTERVAL_MS = 8     # 125 timer wakes/s; display presents up to its refresh rate
_SLOW_LAYER_DIVISOR = 4       # trail + slider refresh at ~31 Hz
# Flick quality thresholds (match analysis conventions: overshoot_rate uses
# 0.1, notable "clean" uses <= 1 correction).
_FLAWED_OVERSHOOT = 0.10
_FLAWED_CORRECTIONS = 2
_SPEED_BANDS = 6
# Sequential and colour-blind-friendly enough for a quantitative layer; red
# and green remain reserved for the quality halo around this line.
_SPEED_COLORS_DARK = (
    "#6272FF", "#3B9EFF", "#25C4C8", "#55D68B", "#D4D64B", "#FFB347")
_SPEED_COLORS_LIGHT = (
    "#3546B0", "#1769AA", "#007C83", "#2D8245", "#827D00", "#B65B00")


def _band_path(x: np.ndarray, y: np.ndarray,
               segment_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized PlotCurve data for selected line segments.

    Consecutive segments share their endpoint; disconnected runs receive one
    NaN separator. This avoids the old Python ``extend`` loop for every one of
    up to 50k segments and usually uploads substantially fewer coordinates.
    """
    idx = np.asarray(segment_indices, dtype=np.intp)
    if not idx.size:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    points = np.empty(idx.size * 2, dtype=np.intp)
    points[0::2] = idx
    points[1::2] = idx + 1
    xs = np.asarray(x[points], dtype=np.float64)
    ys = np.asarray(y[points], dtype=np.float64)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    if breaks.size:
        positions = 2 * (breaks + 1)
        xs = np.insert(xs, positions, np.nan)
        ys = np.insert(ys, positions, np.nan)
    return xs, ys


class TrajectoryReplay(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._t = np.empty(0)
        self._x = np.empty(0)
        self._y = np.empty(0)
        self._point_speed = np.empty(0)
        self._click_times = np.empty(0)
        self._click_x = np.empty(0)
        self._click_y = np.empty(0)
        self._click_numbers = np.empty(0, dtype=np.int32)
        self._speed_bounds = (0.0, 0.0)
        self._deg_per_count = 0.0
        self._pos = 0.0          # playhead (s from segment start)
        self._speed = 0.5        # default half speed: flicks are fast
        self._clock = QElapsedTimer()
        self._clock_base = 0.0   # _pos when the clock (re)started
        self._frame_seq = 0
        self._head_band = -1
        self._active_shot = -1
        self._cache_building = False
        self._gpu_checked = False

        self.plot = pg.PlotWidget()
        # PlotCurveItem 0.14 has a native VBO path, but it is only selected
        # when the GraphicsView owns an OpenGL viewport. Keep headless tests
        # and remote/minimal Qt platforms on the software fallback.
        platform = QGuiApplication.platformName().lower()
        self._gpu_backend = platform not in {"offscreen", "minimal", "minimalegl"}
        if self._gpu_backend:
            try:
                self.plot.useOpenGL(True)
            except Exception:
                self._gpu_backend = False
        self.plot.setAspectLocked(True)
        self.plot.hideAxis("bottom")
        self.plot.hideAxis("left")
        # chrome-min: the canvas is pure trajectory — no context menu, no
        # autorange button (the surrounding ASCII viz has no chrome either)
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self._full = self.plot.plot([], [])
        self._speed_curves = [self.plot.plot([], [], connect="finite")
                              for _ in range(_SPEED_BANDS)]
        self._good = self.plot.plot([], [], connect="finite")
        self._bad = self.plot.plot([], [], connect="finite")
        self._live = self.plot.plot([], [])
        # Whole-run curves are static while the playhead moves. QGraphicsItem
        # cache modes still replay the entire antialiased painter path (and
        # benchmark worse here), so cache the composed pixels explicitly.
        self._static_cache = QGraphicsPixmapItem()
        self.plot.addItem(self._static_cache, ignoreBounds=True)
        # The head owns one cached spot at local (0, 0). Animation uses
        # QGraphicsItem.setPos(), avoiding ScatterPlotItem.setData() and its
        # per-frame data/bounds/cache rebuild.
        self._head = pg.ScatterPlotItem(size=10, pen=None)
        self._shots = pg.ScatterPlotItem(
            size=14, brush=None, symbol="x", hoverable=True,
            # pyqtgraph 0.14 calls tips with x=, y= and data= keywords.
            # Parameter names are therefore API, even though only the click
            # number is displayed.
            tip=lambda x, y, data: f"第 {int(data)} 次点击")
        self._shot_label = pg.TextItem(anchor=(0.5, 1.35))
        self.plot.addItem(self._head)
        self.plot.addItem(self._shots)
        self.plot.addItem(self._shot_label)
        self._full.setZValue(0)
        self._good.setZValue(1)
        self._bad.setZValue(1)
        for curve in self._speed_curves:
            curve.setZValue(2)
        self._static_cache.setZValue(2)
        self._static_cache.hide()
        self._live.setZValue(3)
        self._head.setZValue(4)
        self._shots.setZValue(4)
        self._shot_label.setZValue(5)
        self._shot_label.hide()
        self._shots.sigClicked.connect(self._shot_clicked)

        self.btn = QPushButton(tr("Replay"))
        self.btn.clicked.connect(self.toggle)
        self.speed_btn = QPushButton("0.5x")
        self.speed_btn.clicked.connect(self._cycle_speed)
        # layer toggles: hide/show existing items, never restructure them
        self.toggle_path = QCheckBox(tr("path"))
        self.toggle_path.setToolTip("显示当前时间窗口内的完整鼠标轨迹")
        self.toggle_flicks = QCheckBox(tr("flicks"))
        self.toggle_flicks.setToolTip(
            "显示甩枪质量：绿色表示干净，红色表示过冲或二次修正")
        self.toggle_shots = QCheckBox(tr("shots"))
        self.toggle_shots.setToolTip(
            "以 ✕ 标出每次射击的位置；悬停或播放到该点可查看点击序号")
        for box in (self.toggle_path, self.toggle_flicks, self.toggle_shots):
            box.setChecked(True)
        self.toggle_path.toggled.connect(self._set_path_visible)
        self.toggle_flicks.toggled.connect(self._set_flicks_visible)
        self.toggle_shots.toggled.connect(self._set_shots_visible)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, _SLIDER_STEPS)
        self.scrub.sliderMoved.connect(self._scrubbed)
        self.info = QLabel("")
        self.info.setProperty("dim", True)
        self.legend = QLabel("")
        self.legend.setTextFormat(Qt.RichText)
        self.legend.setProperty("dim", True)

        bar = QHBoxLayout()
        bar.addWidget(self.btn)
        bar.addWidget(self.speed_btn)
        bar.addWidget(self.toggle_path)
        bar.addWidget(self.toggle_flicks)
        bar.addWidget(self.toggle_shots)
        bar.addWidget(self.scrub, 1)
        bar.addWidget(self.info)
        sub = QHBoxLayout()
        sub.addWidget(self.legend)
        sub.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addLayout(sub)
        lay.addWidget(self.plot, 1)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(_PLAYHEAD_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._cache_timer = QTimer(self)
        self._cache_timer.setSingleShot(True)
        self._cache_timer.setInterval(40)
        self._cache_timer.timeout.connect(self._rebuild_static_cache)
        self.plot.getViewBox().sigRangeChanged.connect(
            self._schedule_static_cache)
        self.restyle()

    # ------------------------------------------------------------------
    def restyle(self, *_pal) -> None:
        """Re-pen every curve from the active palette (called on theme switch)."""
        pal = theme.current()
        self.plot.setBackground(pal.bg_alt)
        # ~40% alpha: the full path is context, not the story — the comet
        # trail and flick overlays must read on top of it
        full_pen = pg.mkColor(pal.fg_dim)
        full_pen.setAlphaF(0.4)
        self._full.setPen(pg.mkPen(full_pen, width=1))
        good = pg.mkColor(pal.good)
        bad = pg.mkColor(pal.bad)
        good.setAlphaF(0.35)
        bad.setAlphaF(0.35)
        # Quality is a wide translucent halo; the quantitative speed colour
        # stays visible as the narrower line above it.
        self._good.setPen(pg.mkPen(good, width=6))
        self._bad.setPen(pg.mkPen(bad, width=6))
        colors = (_SPEED_COLORS_DARK if pal.is_dark else _SPEED_COLORS_LIGHT)
        for curve, color in zip(self._speed_curves, colors):
            curve.setPen(pg.mkPen(QColor(color), width=2.5))
        self._live.setPen(pg.mkPen(pal.accent, width=2))
        current_speed = (float(self._point_speed[min(
            max(int(np.searchsorted(self._t, self._pos)) - 1, 0),
            self._point_speed.size - 1)]) if self._point_speed.size else 0.0)
        self._head.setBrush(pg.mkBrush(
            self._color_for_speed(current_speed) if self._point_speed.size
            else QColor(pal.accent)))
        self._head_band = self._speed_band(current_speed)
        self._shots.setPen(pg.mkPen(pal.bad, width=2))
        self._shot_label.setColor(QColor(pal.fg))
        self._update_legend()
        if hasattr(self, "_cache_timer"):
            self._schedule_static_cache()

    def _color_for_speed(self, value: float) -> QColor:
        """Palette-aware speed-band colour for the playhead."""
        colors = (_SPEED_COLORS_DARK if theme.current().is_dark
                  else _SPEED_COLORS_LIGHT)
        return QColor(colors[self._speed_band(value)])

    def _speed_band(self, value: float) -> int:
        """Quantise speed without allocating a QColor on every animation tick."""
        lo, hi = self._speed_bounds
        if hi <= lo:
            return 0
        norm = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        return min(int(norm * _SPEED_BANDS), _SPEED_BANDS - 1)

    def _update_legend(self) -> None:
        pal = theme.current()
        colors = (_SPEED_COLORS_DARK if pal.is_dark else _SPEED_COLORS_LIGHT)
        lo, hi = self._speed_bounds
        if self._deg_per_count > 0 and hi > 0:
            bounds = (f"{lo * self._deg_per_count:.0f}–"
                      f"{hi * self._deg_per_count:.0f}°/秒")
        elif hi > 0:
            bounds = f"{lo:.0f}–{hi:.0f} 计数/秒"
        else:
            bounds = "按当前窗口自动缩放"
        ramp = "".join(
            f"<span style='color:{c}'>━</span>" for c in colors)
        self.legend.setText(
            f"速度：慢 {ramp} 快（{bounds}）&nbsp;&nbsp;"
            f"<span style='color:{pal.good}'>▬</span> 干净甩枪外框&nbsp;&nbsp;"
            f"<span style='color:{pal.bad}'>▬</span> 问题甩枪外框&nbsp;&nbsp;"
            f"<span style='color:{pal.bad}'>✕</span> 射击点")

    # ------------------------------------------------------------------
    def _set_flicks_visible(self, on: bool) -> None:
        """One toggle drives both flick overlays (they are a single layer)."""
        if self._t.size and self._static_cache.isVisible():
            self._schedule_static_cache()
        else:
            self._good.setVisible(on)
            self._bad.setVisible(on)

    def _set_path_visible(self, on: bool) -> None:
        if self._t.size and self._static_cache.isVisible():
            self._schedule_static_cache()
        else:
            self._full.setVisible(on)
            for curve in self._speed_curves:
                curve.setVisible(on)

    def _set_shots_visible(self, on: bool) -> None:
        self._shots.setVisible(on)
        self._shot_label.setVisible(on and self._active_shot >= 0)

    def _drop_static_cache(self) -> None:
        """Return to vectors while data/range/style is being replaced."""
        self._cache_timer.stop()
        self._static_cache.hide()
        self._static_cache.setPixmap(QPixmap())
        self._full.setVisible(self.toggle_path.isChecked())
        for curve in self._speed_curves:
            curve.setVisible(self.toggle_path.isChecked())
        self._good.setVisible(self.toggle_flicks.isChecked())
        self._bad.setVisible(self.toggle_flicks.isChecked())

    def _schedule_static_cache(self, *_args) -> None:
        """Debounce resize/range/theme changes into one cache rebuild."""
        if self._t.size and self.isVisible() and not self._cache_building:
            self._cache_timer.start()

    def _rebuild_static_cache(self) -> None:
        """Compose static path/flick vectors once at the current view range.

        Click markers stay as a real ScatterPlotItem so hover/click numbering
        remains interactive. Only the expensive, immutable vector layers are
        flattened; the playhead and short trail continue to use data coords.
        """
        if self._cache_building or not self._t.size or not self.isVisible():
            return
        path_on = self.toggle_path.isChecked()
        flicks_on = self.toggle_flicks.isChecked()
        if not path_on and not flicks_on:
            self._drop_static_cache()
            self._full.hide()
            for curve in self._speed_curves:
                curve.hide()
            self._good.hide()
            self._bad.hide()
            return

        self._cache_building = True
        try:
            self._static_cache.hide()
            self._full.setVisible(path_on)
            for curve in self._speed_curves:
                curve.setVisible(path_on)
            self._good.setVisible(flicks_on)
            self._bad.setVisible(flicks_on)

            dynamic = (self._live, self._head, self._shots, self._shot_label)
            dynamic_visible = [item.isVisible() for item in dynamic]
            try:
                for item in dynamic:
                    item.hide()

                vb = self.plot.getViewBox()
                scene_rect = vb.sceneBoundingRect()
                top_left = self.plot.mapFromScene(scene_rect.topLeft())
                bottom_right = self.plot.mapFromScene(scene_rect.bottomRight())
                crop = QRect(top_left, bottom_right).normalized().intersected(
                    self.plot.viewport().rect())
                pixmap = (self.plot.viewport().grab(crop)
                          if crop.width() > 1 and crop.height() > 1 else QPixmap())
            finally:
                for item, visible in zip(dynamic, dynamic_visible):
                    item.setVisible(visible)
            self._full.hide()
            for curve in self._speed_curves:
                curve.hide()
            self._good.hide()
            self._bad.hide()

            if pixmap.isNull():
                # Capturing an unsupported OpenGL/remote viewport can fail.
                # Leave the correct vector rendering in place rather than a
                # fast blank canvas.
                self._full.setVisible(path_on)
                for curve in self._speed_curves:
                    curve.setVisible(path_on)
                self._good.setVisible(flicks_on)
                self._bad.setVisible(flicks_on)
                return

            self._static_cache.setPixmap(pixmap)
            bounds = self._static_cache.boundingRect()
            view = vb.viewRect()
            if bounds.width() <= 0 or bounds.height() <= 0:
                return
            sx = view.width() / bounds.width()
            sy = view.height() / bounds.height()
            self._static_cache.setTransform(QTransform(
                sx, 0.0, 0.0, -sy, view.left(), view.bottom()))
            self._static_cache.show()
        finally:
            self._cache_building = False

    # ------------------------------------------------------------------
    def load(self, trace: MouseTrace, t0: float | None = None,
             t1: float | None = None, label: str = "",
             flicks: list | None = None,
             deg_per_count: float = 0.0) -> None:
        """Show [t0, t1] of the trace (defaults: whole trace). `flicks` are
        analysis.movement.Flick objects for the SAME trace (absolute epoch
        times); the ones inside the window become quality overlays."""
        self.stop()
        self._drop_static_cache()
        seg = trace if t0 is None else trace.window(t0, t1 if t1 is not None else trace.t[-1])
        # A uniform grid represents rest as zero speed. Raw packet timestamps
        # contain no samples while the hand is still, so differentiating the
        # raw path would assign the entire rest gap to the first moving packet
        # and paint a fast launch as artificially slow.
        t, vx, vy = seg.resample(500.0)
        if t.size >= 2:
            x = np.cumsum(vx, dtype=np.float64) / 500.0
            y = -np.cumsum(vy, dtype=np.float64) / 500.0
            point_speed = np.hypot(vx, vy)
        else:
            t, x, y = seg.path()
            dt = np.diff(t, prepend=t[0] if t.size else 0.0)
            distance = np.hypot(np.diff(x, prepend=0.0),
                                np.diff(y, prepend=0.0))
            point_speed = np.divide(
                distance, dt, out=np.zeros_like(distance), where=dt > 0)
        if t.size < 2:
            self._t = np.empty(0)
            self._point_speed = np.empty(0)
            self._clear_clicks()
            self._speed_bounds = (0.0, 0.0)
            self._deg_per_count = max(float(deg_per_count or 0.0), 0.0)
            for item in (self._full, self._good, self._bad, self._live,
                         *self._speed_curves):
                item.setData([], [])
            self._head.setData([], [])
            self.info.setText("此时间窗口内没有鼠标移动")
            self._update_legend()
            self._sync_transport()
            return
        # Uniformly reduce only the DRAWING representation. linspace keeps
        # both endpoints and enforces the cap exactly; integer division used
        # to leave 50,001..99,999 samples completely undecimated.
        if t.size > _MAX_POINTS:
            keep = np.linspace(0, t.size - 1, _MAX_POINTS, dtype=np.intp)
            t, x, y = t[keep], x[keep], y[keep]
            point_speed = point_speed[keep]
        base = t[0]
        self._t = t - base
        self._x, self._y = x, y
        self._point_speed = point_speed
        self._deg_per_count = max(float(deg_per_count or 0.0), 0.0)
        # The six coloured curves cover every segment, so uploading the same
        # 50k-point geometry once more as a grey underlay only consumes scene
        # graph/GPU work without adding information.
        self._full.setData([], [])
        self._draw_speed_path()
        self._live.setData([], [])
        self._head.setData(np.array([0.0]), np.array([0.0]))
        self._head.setPos(float(x[0]), float(y[0]))
        self._head.setBrush(pg.mkBrush(self._color_for_speed(
            float(point_speed[0]) if point_speed.size else 0.0)))
        self._head_band = self._speed_band(
            float(point_speed[0]) if point_speed.size else 0.0)
        clicks = seg.clicks - base
        ci = np.clip(np.searchsorted(self._t, clicks), 0, t.size - 1)
        # Preserve numbering from the complete trace even in a notable-moment
        # window.  A skipped/too-small movement still consumes a click number.
        click_numbers = np.searchsorted(trace.clicks, seg.clicks) + 1
        self._click_times = np.asarray(clicks, dtype=np.float64)
        self._click_x = np.asarray(x[ci], dtype=np.float64)
        self._click_y = np.asarray(y[ci], dtype=np.float64)
        self._click_numbers = np.asarray(click_numbers, dtype=np.int32)
        self._shots.setData(
            self._click_x, self._click_y, data=self._click_numbers)
        self._set_active_shot(-1)
        self._draw_flicks(base, flicks or [])
        self._pos = 0.0
        self._frame_seq = 0
        self.scrub.setValue(0)
        self.info.setText(label or f"{self._t[-1]:.2f} 秒 · {seg.clicks.size} 次射击")
        self.plot.autoRange()
        self._schedule_static_cache()
        self._sync_transport()

    def _clear_clicks(self) -> None:
        self._click_times = np.empty(0)
        self._click_x = np.empty(0)
        self._click_y = np.empty(0)
        self._click_numbers = np.empty(0, dtype=np.int32)
        self._shots.setData([], [])
        self._set_active_shot(-1)

    def _set_active_shot(self, index: int) -> None:
        """Move the single reusable number label to a click marker."""
        if index < 0 or index >= self._click_numbers.size:
            self._active_shot = -1
            self._shot_label.hide()
            return
        if index != self._active_shot:
            self._active_shot = index
            self._shot_label.setText(f"第 {int(self._click_numbers[index])} 次点击")
            self._shot_label.setPos(
                float(self._click_x[index]), float(self._click_y[index]))
        self._shot_label.setVisible(self.toggle_shots.isChecked())

    def _shot_clicked(self, _item, points, _event) -> None:
        """Pin the sequence label when a shot marker is clicked."""
        if not points:
            return
        number = int(points[0].data())
        matches = np.flatnonzero(self._click_numbers == number)
        if matches.size:
            self._set_active_shot(int(matches[0]))

    def _draw_speed_path(self) -> None:
        """Six NaN-separated curves, quantised by robust window speed."""
        for curve in self._speed_curves:
            curve.setData([], [])
        if self._t.size < 2 or self._point_speed.size != self._t.size:
            self._speed_bounds = (0.0, 0.0)
            self._update_legend()
            return
        moving = self._point_speed[self._point_speed > 0]
        if not moving.size:
            self._speed_bounds = (0.0, 0.0)
            self._update_legend()
            return
        lo, hi = np.percentile(moving, [10, 95])
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1.0
        self._speed_bounds = (lo, hi)
        segment_speed = 0.5 * (self._point_speed[:-1] + self._point_speed[1:])
        norm = np.clip((segment_speed - lo) / (hi - lo), 0.0, 1.0)
        bands = np.minimum((norm * _SPEED_BANDS).astype(int), _SPEED_BANDS - 1)
        for band, curve in enumerate(self._speed_curves):
            xs, ys = _band_path(self._x, self._y,
                                np.flatnonzero(bands == band))
            curve.setData(xs, ys)
        self._update_legend()

    def _draw_flicks(self, base: float, flicks: list) -> None:
        """Two NaN-separated polylines: clean (green) and flawed (red)."""
        good: list[list[float]] = [[], []]
        bad: list[list[float]] = [[], []]
        tmax = self._t[-1] if self._t.size else 0.0
        for f in flicks:
            o, c = f.t_onset - base, f.t_click - base
            if c <= 0 or o >= tmax or c <= o:
                continue
            i0, i1 = np.searchsorted(self._t, [o, c])
            i1 = min(int(i1) + 1, self._t.size)
            if i1 - i0 < 2:
                continue
            flawed = (f.overshoot > _FLAWED_OVERSHOOT
                      or f.corrections >= _FLAWED_CORRECTIONS)
            dest = bad if flawed else good
            dest[0].extend(self._x[i0:i1].tolist() + [np.nan])
            dest[1].extend(self._y[i0:i1].tolist() + [np.nan])
        self._good.setData(good[0], good[1])
        self._bad.setData(bad[0], bad[1])

    # ------------------------------------------------------------------
    def clear(self, message: str = "本局没有鼠标轨迹") -> None:
        """Empty the plot (used when a report arrives without telemetry, so
        the previous run's path can't masquerade as the current one)."""
        self.stop()
        self._drop_static_cache()
        self._t = np.empty(0)
        self._point_speed = np.empty(0)
        self._clear_clicks()
        self._speed_bounds = (0.0, 0.0)
        for item in (self._full, self._good, self._bad, self._live,
                     *self._speed_curves):
            item.setData([], [])
        self._head.setData([], [])
        self.scrub.setValue(0)
        self.info.setText(message)
        self._update_legend()
        self._sync_transport()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._gpu_backend and not self._gpu_checked:
            # QOpenGLWidget obtains its context only after its first paint.
            # A delayed validation gives remote desktop / broken-driver
            # sessions a clean software fallback instead of a black canvas.
            QTimer.singleShot(100, self._validate_gpu_viewport)
        self._schedule_static_cache()

    def _validate_gpu_viewport(self) -> None:
        if not self._gpu_backend or not self.isVisible():
            return
        viewport = self.plot.viewport()
        valid = getattr(viewport, "isValid", None)
        if callable(valid) and not valid():
            self.plot.useOpenGL(False)
            self._gpu_backend = False
            self._drop_static_cache()
            self._schedule_static_cache()
        self._gpu_checked = True

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_static_cache()

    def _sync_transport(self) -> None:
        """Enable the transport only when there is something to transport.

        On a cold launch this panel rendered a full, LIVE control strip — an
        enabled Replay button, a speed cycle, three checked layer boxes, a
        scrub slider and a colour legend — over 531,912px of a single flat
        colour with no data behind any of it. Clicking Replay did nothing,
        which is the worst answer a control can give.
        """
        live = bool(self._t.size)
        for w in (self.btn, self.speed_btn, self.scrub,
                  self.toggle_path, self.toggle_flicks, self.toggle_shots):
            w.setEnabled(live)
        if not live:
            self.btn.setText(tr("Replay"))

    # ------------------------------------------------------------------
    def toggle(self) -> None:
        if self._timer.isActive():
            self.stop()
        elif self._t.size:
            if self._pos >= self._t[-1]:
                self._pos = 0.0
            self._clock_base = self._pos
            self._frame_seq = 0
            self._clock.start()
            self.btn.setText(tr("Stop"))
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.btn.setText(tr("Replay"))

    def _cycle_speed(self) -> None:
        order = [0.25, 0.5, 1.0]
        self._speed = order[(order.index(self._speed) + 1) % len(order)]
        self.speed_btn.setText(f"{self._speed:g}x")
        if self._timer.isActive():           # rebase so speed changes mid-play
            self._clock_base = self._pos
            self._clock.restart()

    def _scrubbed(self, v: int) -> None:
        if not self._t.size:
            return
        self.stop()
        self._pos = self._t[-1] * v / _SLIDER_STEPS
        self._render(force_slow=True)

    def _tick(self) -> None:
        if not self._t.size:
            self.stop()
            return
        self._pos = self._clock_base + self._clock.elapsed() / 1000.0 * self._speed
        if self._pos >= self._t[-1]:
            self._pos = self._t[-1]
            self.stop()
        self._frame_seq += 1
        force_slow = (self._pos >= self._t[-1]
                      or self._frame_seq % _SLOW_LAYER_DIVISOR == 0)
        if force_slow:
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(self._pos / self._t[-1] * _SLIDER_STEPS))
            self.scrub.blockSignals(False)
        self._render(force_slow=force_slow)

    def _render(self, *, force_slow: bool = False) -> None:
        i = int(np.searchsorted(self._t, self._pos))
        i = max(min(i, self._t.size - 1), 1)
        point = i - 1
        # Transform-only update: the cached one-point ScatterPlotItem keeps
        # its symbol atlas, bounds and data arrays across animation frames.
        self._head.setPos(float(self._x[point]), float(self._y[point]))
        band = self._speed_band(float(self._point_speed[point]))
        if band != self._head_band:
            self._head_band = band
            self._head.setBrush(pg.mkBrush(self._color_for_speed(
                float(self._point_speed[point]))))

        # Comet tail is perceptually smooth at ~30 Hz while the head remains
        # display-rate smooth.  This is the only animated layer that uploads
        # an array, so keep it off the 125 Hz hot path.
        if force_slow:
            j = int(np.searchsorted(
                self._t, self._t[point] - _TRAIL_SECONDS))
            self._live.setData(self._x[j:i], self._y[j:i],
                               skipFiniteCheck=True)

        shot = int(np.searchsorted(self._click_times, self._pos, side="right") - 1)
        self._set_active_shot(shot)
