"""Adaptability configuration tab: the full Settings surface, saved to
~/.kovadapt/settings.json. More knobs than KovaaK's official offering.

Sections: mouse & sensitivity first (feeds the model's per-task sensitivity
reasoning), then the everyday controls, the advanced engine internals
(exposed for power users; defaults reproduce the shipped behavior), the
trace-informed dodge/fatigue features, and per-archetype overrides. The
group boxes stack in one editorial column and flow in the page-space —
no nested scroll of their own.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from .i18n import archetype_name, tr
from .onboarding import HintBar

# Override keys editable per archetype (subset of Settings fields that map
# cleanly to "how should adaptation differ for this task type").
# (field, label, lo, hi) — the range is PER KEY. A blanket 0.0-3.0 for all
# five let the archetype pages save values the fields cannot mean: accuracy
# bands and min_movement are fractions, focus_weight is a share of spawn mass,
# so 3.0 is not a stricter setting, it is a nonsense one. It also crashed the
# "What changed" ladder, which plots focus_weight on a 0..100% rail and
# (correctly) refuses to paint a clamped position next to an unclamped number.
# Each range matches that field's own global spin above.
_ARCH_KEYS = (
    ("target_accuracy_low", "准确率下限", 0.30, 0.99),
    ("target_accuracy_high", "准确率上限", 0.35, 1.00),
    ("size_learning_rate", "目标尺寸调整速度", 0.05, 3.00),
    ("min_movement", "最低移动强度", 0.00, 1.00),
    ("focus_weight", "弱区训练权重", 0.00, 0.90),
)
_ARCH_EDITABLE = ("tracking", "switching")   # clicking is the baseline


def _dspin(val: float, lo: float, hi: float, step: float = 0.05, dec: int = 2,
           tip: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    w.setMinimumWidth(160)      # wide inputs for the editorial column
    if tip:
        w.setToolTip(tip)
    return w


def _ispin(val: int, lo: int, hi: int, tip: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setMinimumWidth(160)
    if tip:
        w.setToolTip(tip)
    return w


def _form(box: QGroupBox) -> QFormLayout:
    """Single-column form with the column's generous padding."""
    f = QFormLayout(box)
    f.setContentsMargins(16, 12, 16, 16)
    f.setHorizontalSpacing(28)
    f.setVerticalSpacing(12)
    return f


class ConfigView(QWidget):
    settings_changed = Signal(object)   # emits the saved Settings

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        s = settings

        # mouse & sensitivity — feeds the model's per-task sensitivity
        # reasoning. getattr-guarded: the fields may land in a later build.
        self.dpi = _dspin(
            float(getattr(s, "mouse_dpi", 0) or 800.0), 100.0, 32000.0, 50.0, 0,
            "鼠标驱动或配套软件中设置的硬件 DPI（CPI）。")
        self.sens = _dspin(
            float(getattr(s, "game_sens", 0) or 1.0), 0.01, 20.0, 0.01, 2,
            "KovaaK's 游戏内使用的鼠标灵敏度。")
        self.cm360 = QLabel("")
        self.cm360.setProperty("stat", True)
        cm_cap = QLabel(
            "自适应模型会用这个数值分析不同训练类型所需的灵敏度；cm/360 按 "
            "KovaaK's 每计数 0.022° 的 yaw 值换算。")
        cm_cap.setProperty("dim", True)
        cm_cap.setWordWrap(True)
        # "&&" is a literal ampersand. A bare "&" marks the next character as a
        # keyboard mnemonic, which is why this rendered as "Mouse _sensitivity".
        mouse = QGroupBox(tr("Mouse & sensitivity"))
        f = _form(mouse)
        f.addRow("鼠标 DPI", self.dpi)
        f.addRow("游戏内灵敏度", self.sens)
        f.addRow("转身 360° 所需厘米数", self.cm360)
        f.addRow(cm_cap)
        self.dpi.valueChanged.connect(self._update_cm360)
        self.sens.valueChanged.connect(self._update_cm360)
        self._update_cm360()

        # difficulty
        self.acc_lo = _dspin(s.target_accuracy_low, 0.1, 0.95)
        self.acc_hi = _dspin(s.target_accuracy_high, 0.15, 0.99)
        self.lr = _dspin(s.size_learning_rate, 0.1, 3.0, 0.1)
        self.scale_min = _dspin(s.min_target_scale, 0.1, 1.0)
        self.scale_max = _dspin(s.max_target_scale, 1.0, 5.0)
        diff = QGroupBox(tr("Difficulty controller"))
        f = _form(diff)
        f.addRow("目标准确率下限", self.acc_lo)
        f.addRow("目标准确率上限", self.acc_hi)
        f.addRow("尺寸调整速度", self.lr)
        f.addRow("目标最小缩放", self.scale_min)
        f.addRow("目标最大缩放", self.scale_max)

        # weakness targeting
        self.cols = _ispin(s.region_cols, 2, 5)
        self.rows = _ispin(s.region_rows, 1, 5)
        self.focus = _dspin(s.focus_weight, 0.0, 0.9)
        self.blend = _dspin(s.telemetry_blend, 0.0, 1.0)
        reg = QGroupBox(tr("Weak-region targeting (bandit)"))
        f = _form(reg)
        f.addRow("网格列数", self.cols)
        f.addRow("网格行数", self.rows)
        f.addRow("弱区目标生成权重", self.focus)
        f.addRow("鼠标遥测证据权重", self.blend)

        # stochastic movement
        self.theta = _dspin(s.ou_theta, 0.05, 2.0)
        self.sigma = _dspin(s.ou_sigma, 0.0, 1.5)
        self.mov_min = _dspin(s.min_movement, 0.0, 1.0)
        self.mov_max = _dspin(s.max_movement, 0.0, 1.0)
        mov = QGroupBox(tr("Anti-autopilot movement (Ornstein-Uhlenbeck)"))
        f = _form(mov)
        f.addRow("均值回归 θ", self.theta)
        f.addRow("扩散强度 σ", self.sigma)
        f.addRow("最低移动强度", self.mov_min)
        f.addRow("最高移动强度", self.mov_max)

        # telemetry / clips
        self.skip_splash_cb = QCheckBox(
            "跳过加载画面，直接进入主窗口")
        self.skip_splash_cb.setChecked(bool(getattr(s, "skip_splash", False)))
        # Motion intensity is one dial rather than a code change: taste is
        # personal, and on a 240 Hz panel a dropped frame is visible.
        self.motion = QComboBox()
        for label, value in (("完整 — 包含背景与所有过渡动画", "full"),
                             ("精简 — 只保留有意义的结果过渡", "reduced"),
                             ("关闭 — 所有内容立即显示", "off")):
            self.motion.addItem(label, value)
        cur = str(getattr(s, "motion", "full") or "full").lower()
        idx = self.motion.findData(cur)
        self.motion.setCurrentIndex(idx if idx >= 0 else 0)
        self.motion.setToolTip(
            "“完整”会播放背景和所有揭示动画；“精简”只保留本局结果等有意义的"
            "过渡并关闭环境循环；“关闭”会直接显示最终状态，不播放动画。")
        self.telemetry = QCheckBox(tr("Record raw mouse telemetry while watching"))
        self.telemetry.setChecked(s.telemetry_enabled)
        self.clips = QCheckBox(
            tr("Capture video clips of notable moments (needs kovadapt[clips])"))
        self.clips.setChecked(s.clips_enabled)
        self.clip_fps = _ispin(s.clip_fps, 10, 60)
        self.clip_buf = _dspin(s.clip_buffer_seconds, 30.0, 300.0, 10.0, 0)
        tel = QGroupBox(tr("Telemetry & clips"))
        f = _form(tel)
        f.addRow(self.skip_splash_cb)
        f.addRow("界面动效", self.motion)
        f.addRow(self.telemetry)
        f.addRow(self.clips)
        f.addRow("录像帧率", self.clip_fps)
        f.addRow("录像循环缓冲（秒）", self.clip_buf)

        # advanced engine internals
        self.half_life = _dspin(
            s.ewma_half_life, 1.0, 50.0, 1.0, 1,
            "旧训练记录对平均值的影响衰减到一半所需的局数。")
        self.coupling = _dspin(
            s.size_speed_coupling, 0.0, 1.0, 0.05, 2,
            "目标移动加快时，为保持公平而相应放大目标的程度。")
        self.pace_gain = _dspin(
            s.pace_coupling_gain, 0.0, 2.0, 0.05, 2,
            "本局节奏高于个人常态时，下一局目标移动强度提高多少。")
        self.min_shots = _ispin(
            s.min_shots_for_size, 0, 100,
            "本局射击数低于此值时，不允许尺寸控制器改变目标大小。")
        self.obs_noise = _dspin(
            s.bandit_obs_noise, 0.05, 2.0, 0.05, 2,
            "区域证据的观测噪声；数值越低，每局数据对模型判断的影响越大。")
        self.prior_var = _dspin(
            s.bandit_prior_var, 0.1, 5.0, 0.1, 2,
            "未探索区域的先验方差；数值越高，训练初期越倾向探索新区域。")
        self.decay = _dspin(
            s.bandit_posterior_decay, 0.0, 0.5, 0.01, 2,
            "每局结束后向先验状态回退的比例，使已改善的弱项能够重新接受验证；0 表示永不遗忘。")
        adv = QGroupBox(tr("Advanced engine internals"))
        f = _form(adv)
        f.addRow("EWMA 半衰期（局）", self.half_life)
        f.addRow("尺寸—速度耦合", self.coupling)
        f.addRow("节奏耦合增益", self.pace_gain)
        f.addRow("尺寸调整最低射击数", self.min_shots)
        f.addRow("Bandit 观测噪声", self.obs_noise)
        f.addRow("Bandit 先验方差", self.prior_var)
        f.addRow("Bandit 后验衰减", self.decay)

        # trace-informed dodge
        self.dodge_en = QCheckBox("让目标更常向你的弱侧移动")
        self.dodge_en.setChecked(s.dodge_bias_enabled)
        self.dodge_gain = _dspin(
            s.dodge_bias_gain, 0.0, 2.0, 0.1, 1,
            "把测得的左右方向差异转换成目标闪避时间的不对称程度。")
        dodge = QGroupBox(tr("Trace-informed dodge direction"))
        f = _form(dodge)
        f.addRow(self.dodge_en)
        f.addRow("方向偏差增益", self.dodge_gain)

        # fatigue
        self.fat_en = QCheckBox("检测甩枪质量下降并建议休息")
        self.fat_en.setChecked(s.fatigue_detection_enabled)
        self.fat_ease = QCheckBox("疲劳时降低难度（目标更大、移动更平缓）")
        self.fat_ease.setChecked(s.fatigue_easing)
        self.fat_sens = _dspin(
            s.fatigue_sensitivity, 0.1, 3.0, 0.1, 1,
            "高于 1 会更早判定疲劳，低于 1 会更晚判定。")
        self.fat_runs = _ispin(
            s.fatigue_min_runs, 2, 20,
            "在疲劳趋势被采信前，至少需要多少局带鼠标遥测的记录。")
        fat = QGroupBox(tr("Session fatigue"))
        f = _form(fat)
        f.addRow(self.fat_en)
        f.addRow(self.fat_ease)
        f.addRow("检测敏感度", self.fat_sens)
        f.addRow("最少局数", self.fat_runs)

        # per-archetype overrides
        self.arch_en = QCheckBox(
            "按训练类型分别适配（自动识别：点击 / 跟枪 / 目标切换）")
        self.arch_en.setChecked(s.archetype_enabled)
        self.arch_spins: dict[str, dict[str, QDoubleSpinBox]] = {}
        # A spin with no override shows the global value it inherits, which is
        # indistinguishable from a deliberate per-archetype choice. Remember
        # which keys were real overrides and what each spin was loaded with, so
        # _save() can tell "inherited" from "chosen".
        self._arch_explicit: dict[str, set[str]] = {}
        self._arch_base: dict[str, dict[str, float]] = {}
        arch = QGroupBox(tr("Per-archetype overrides (clicking is the baseline)"))
        av = QVBoxLayout(arch)
        av.setContentsMargins(16, 12, 16, 16)
        av.setSpacing(14)
        av.addWidget(self.arch_en)
        for name in _ARCH_EDITABLE:
            ov = (s.archetype_overrides or {}).get(name) or {}
            spins: dict[str, QDoubleSpinBox] = {}
            for key, cap, lo, hi in _ARCH_KEYS:
                spins[key] = _dspin(float(ov.get(key, getattr(s, key))), lo, hi)
            self.arch_spins[name] = spins
            self._remember_arch(name, ov)
            box = QGroupBox(archetype_name(name))
            row = _form(box)
            for key, cap, _lo, _hi in _ARCH_KEYS:
                row.addRow(cap, spins[key])
            av.addWidget(box)

        save = QPushButton(tr("Save settings"))
        save.setProperty("accent", True)
        save.clicked.connect(self._save)
        reset = QPushButton(tr("Reset to defaults"))
        reset.setToolTip("把所有参数恢复为软件默认值（保留路径设置）；点击“保存设置”后生效。")
        reset.clicked.connect(self._reset)
        self.status = QLabel("")
        self.status.setProperty("dim", True)
        bar = QHBoxLayout()
        bar.setSpacing(12)
        bar.addWidget(save)
        bar.addWidget(reset)
        bar.addWidget(self.status)
        bar.addStretch(1)

        # group boxes stack in one column and flow in the page-space —
        # the outer shell scrolls, so no nested scroll area here.
        lay = QVBoxLayout(self)
        # ZERO, explicitly. Every section view inherited Qt's ~9px default
        # layout margin, while the section's own H1, its divider rule and
        # every panel sit flush to shell._Section's column — so bare page
        # text was the only thing indented, and lined up with nothing on the
        # screen. The column IS the measure; panels pad their own contents.
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(18)
        lay.addWidget(HintBar(settings, (
            "每个参数都有悬停说明。默认值就是软件初始行为；随时可以点击"
            "<b>恢复默认值</b>回到初始设置。所有改动只有点击<b>保存设置</b>后才会生效。")))
        for box in (mouse, diff, reg, mov, tel, adv, dodge, fat, arch):
            lay.addWidget(box)
        lay.addLayout(bar)
        lay.addStretch(1)

    # ------------------------------------------------------------------
    def _remember_arch(self, name: str, ov: dict) -> None:
        """Snapshot an archetype's baseline: which knobs are real overrides
        and what the spins currently read. Anything loaded (init, reset, the
        readout refresh after a save) is by definition not a user choice."""
        self._arch_explicit[name] = {k for k, *_ in _ARCH_KEYS if k in ov}
        self._arch_base[name] = {k: self.arch_spins[name][k].value() for k, *_ in _ARCH_KEYS}

    def _update_cm360(self) -> None:
        """Live cm/360 readout (KovaaK's/Quake yaw: 0.022° per count)."""
        counts_per_deg = self.dpi.value() * self.sens.value() * 0.022
        if counts_per_deg <= 0:
            self.cm360.setText("—")
            return
        cm = 2.54 * 360.0 / counts_per_deg
        self.cm360.setText(f"{cm:.1f} cm / 360°")

    def _reset(self) -> None:
        """Load shipped defaults into every widget (Save still required)."""
        d = Settings(kovaaks_root=self.s.kovaaks_root, profile_dir=self.s.profile_dir)
        self.dpi.setValue(float(getattr(d, "mouse_dpi", 0) or 800.0))
        self.sens.setValue(float(getattr(d, "game_sens", 0) or 1.0))
        self.acc_lo.setValue(d.target_accuracy_low)
        self.acc_hi.setValue(d.target_accuracy_high)
        self.lr.setValue(d.size_learning_rate)
        self.scale_min.setValue(d.min_target_scale)
        self.scale_max.setValue(d.max_target_scale)
        self.cols.setValue(d.region_cols)
        self.rows.setValue(d.region_rows)
        self.focus.setValue(d.focus_weight)
        self.blend.setValue(d.telemetry_blend)
        self.theta.setValue(d.ou_theta)
        self.sigma.setValue(d.ou_sigma)
        self.mov_min.setValue(d.min_movement)
        self.mov_max.setValue(d.max_movement)
        self.skip_splash_cb.setChecked(bool(getattr(d, "skip_splash", False)))
        mi = self.motion.findData(str(getattr(d, "motion", "full")))
        self.motion.setCurrentIndex(mi if mi >= 0 else 0)
        self.telemetry.setChecked(d.telemetry_enabled)
        self.clips.setChecked(d.clips_enabled)
        self.clip_fps.setValue(d.clip_fps)
        self.clip_buf.setValue(d.clip_buffer_seconds)
        self.half_life.setValue(d.ewma_half_life)
        self.coupling.setValue(d.size_speed_coupling)
        self.pace_gain.setValue(d.pace_coupling_gain)
        self.min_shots.setValue(d.min_shots_for_size)
        self.obs_noise.setValue(d.bandit_obs_noise)
        self.prior_var.setValue(d.bandit_prior_var)
        self.decay.setValue(d.bandit_posterior_decay)
        self.dodge_en.setChecked(d.dodge_bias_enabled)
        self.dodge_gain.setValue(d.dodge_bias_gain)
        self.fat_en.setChecked(d.fatigue_detection_enabled)
        self.fat_ease.setChecked(d.fatigue_easing)
        self.fat_sens.setValue(d.fatigue_sensitivity)
        self.fat_runs.setValue(d.fatigue_min_runs)
        self.arch_en.setChecked(d.archetype_enabled)
        for name, spins in self.arch_spins.items():
            ov = (d.archetype_overrides or {}).get(name) or {}
            for key, *_ in _ARCH_KEYS:
                spins[key].setValue(float(ov.get(key, getattr(d, key))))
            self._remember_arch(name, ov)
        self.status.setText("已载入默认值；点击“保存设置”后应用")

    def _save(self) -> None:
        s = self.s
        # plain attribute set is safe even before the fields land on Settings
        s.mouse_dpi = self.dpi.value()
        s.game_sens = self.sens.value()
        s.target_accuracy_low = min(self.acc_lo.value(), self.acc_hi.value() - 0.01)
        s.target_accuracy_high = self.acc_hi.value()
        s.size_learning_rate = self.lr.value()
        s.min_target_scale = self.scale_min.value()
        s.max_target_scale = self.scale_max.value()
        s.region_cols = self.cols.value()
        s.region_rows = self.rows.value()
        s.focus_weight = self.focus.value()
        s.telemetry_blend = self.blend.value()
        s.ou_theta = self.theta.value()
        s.ou_sigma = self.sigma.value()
        s.min_movement = min(self.mov_min.value(), self.mov_max.value())
        s.max_movement = self.mov_max.value()
        s.skip_splash = self.skip_splash_cb.isChecked()
        s.motion = self.motion.currentData()
        s.telemetry_enabled = self.telemetry.isChecked()
        s.clips_enabled = self.clips.isChecked()
        s.clip_fps = self.clip_fps.value()
        s.clip_buffer_seconds = self.clip_buf.value()
        s.ewma_half_life = self.half_life.value()
        s.size_speed_coupling = self.coupling.value()
        s.pace_coupling_gain = self.pace_gain.value()
        s.min_shots_for_size = self.min_shots.value()
        s.bandit_obs_noise = self.obs_noise.value()
        s.bandit_prior_var = self.prior_var.value()
        s.bandit_posterior_decay = self.decay.value()
        s.dodge_bias_enabled = self.dodge_en.isChecked()
        s.dodge_bias_gain = self.dodge_gain.value()
        s.fatigue_detection_enabled = self.fat_en.isChecked()
        s.fatigue_easing = self.fat_ease.isChecked()
        s.fatigue_sensitivity = self.fat_sens.value()
        s.fatigue_min_runs = self.fat_runs.value()
        s.archetype_enabled = self.arch_en.isChecked()
        overrides = dict(s.archetype_overrides or {})
        for name, spins in self.arch_spins.items():
            ov = dict(overrides.get(name) or {})
            base, explicit = self._arch_base[name], self._arch_explicit[name]
            for key, *_ in _ARCH_KEYS:
                val = round(spins[key].value(), 4)
                glob = round(float(getattr(s, key)), 4)
                # An override may only exist where the user chose to differ.
                # Persisting an inherited value would freeze this archetype at
                # today's global and silently decouple it from every later edit
                # — and since the spin still holds the pre-edit global, the very
                # edit made above would never reach tracking/switching.
                chosen = key in explicit or abs(val - base[key]) > 1e-9
                if chosen and abs(val - glob) > 1e-9:
                    ov[key] = val
                else:
                    ov.pop(key, None)
                    spins[key].setValue(glob)   # keep the inherited readout honest
            # Keep the accuracy band ordered against what the archetype really
            # resolves to: a bound left inherited falls through to the global.
            lo = ov.get("target_accuracy_low", s.target_accuracy_low)
            hi = ov.get("target_accuracy_high", s.target_accuracy_high)
            if lo > hi - 0.01:
                lo = round(hi - 0.01, 4)
                ov["target_accuracy_low"] = lo
                spins["target_accuracy_low"].setValue(lo)
            overrides[name] = ov
            self._remember_arch(name, ov)
        s.archetype_overrides = overrides
        path = s.save()
        self.status.setText(
            f"已保存到 {path}；重新开始分析会话后全部生效")
        self.settings_changed.emit(s)
