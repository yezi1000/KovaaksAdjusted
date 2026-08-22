"""Dashboard tab: HERO + ONE TREND + PLAY.

Three hero numerals across the top — READINESS (how calibrated the adaptive
model is), FORM (recent accuracy against the player's own baseline) and LOAD
(session fatigue, or run volume when there is no fatigue reading) — then one
accuracy trend drawn as character art (gui/viz.py, never pyqtgraph), the Play
lockup, the overlay row, and the session log behind a "[ log ]" disclosure.

Every hero carries a state word AND a "because …" clause naming the evidence
it was computed from: `Hero` holds both and `HeroStat.show_hero()` is the only
way to fill a card, so there is no code path that renders a numeral without
its citation. A bare score is exactly the uncited claim this project forbids —
and when a value genuinely is not available the card shows a dash and the
clause says why, rather than a fake zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import launcher
from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile
from . import theme, viz
from .ascii_art import CatSlider
from .i18n import tr
from .onboarding import HintBar
from .overlay import OverlayWindow
from .workers import WatcherWorker

LOG_LABEL = "[ 日志 ]"
LOG_UNREAD = "[ 日志 • ]"        # a line landed while the disclosure was shut

FORM_WINDOW = 5          # runs averaged into the "recent" side of FORM
FORM_MIN_RUNS = 3        # below this there is no recent window worth naming
FORM_BAND_PP = 1.0       # |delta| under this reads as noise, not a direction


# ------------------------------------------------------------------- heroes
@dataclass(frozen=True)
class Hero:
    """One hero reading: the numeral, its state word, and the evidence.

    `because` is a required field, not an option — `unknown()` is the only way
    to produce a dash and it still demands one. Keeping the clause inside the
    value object is what makes "never show a hero number without its
    because-clause" a type-level guarantee instead of a habit.
    """

    value: str
    word: str
    because: str
    tone: str = "fg"         # fg | good | warn | bad | accent
    tip: str = ""            # fuller derivation, shown on hover

    @classmethod
    def unknown(cls, word: str, because: str, tip: str = "") -> "Hero":
        return cls("—", word, because, "fg", tip)


def readiness_hero(profile: PlayerProfile, region_count: int) -> Hero:
    """How calibrated the adaptive model is.

    Straight from PlayerProfile.readiness(): its score, its stage word, and
    its own detail strings as the clause — the model's readiness metric is
    already defined there and must not be re-derived here, or the dashboard
    and the CLI's `status` would disagree about the same profile.
    """
    if profile.run_count == 0:
        return Hero.unknown(
            "no runs yet",
            "because this scenario has no history — READINESS starts counting "
            "with your first run",
            "READINESS weights a settled accuracy baseline (50%), wall regions "
            "carrying evidence (35%) and directional-bias evidence (15%).")
    r = profile.readiness(region_count)
    tone = {"dialed in": "good", "calibrating": "accent"}.get(r["stage"], "fg")
    return Hero(f"{r['score']:.0%}", r["stage"],
                "because " + " · ".join(r["detail"]), tone, r["message"])


def form_hero(profile: PlayerProfile, half_life: float = 5.0) -> Hero:
    """Recent accuracy against the player's own baseline, in points."""
    accs = [float(h["accuracy"]) for h in profile.history if "accuracy" in h]
    if len(accs) < FORM_MIN_RUNS:
        return Hero.unknown(
            "too few runs",
            f"because FORM averages a recent window against your baseline and "
            f"that needs at least {FORM_MIN_RUNS} runs — this scenario has "
            f"{len(accs)}",
            "FORM = mean accuracy of the last few runs minus the profile's "
            "ewma_accuracy baseline, in percentage points.")
    if profile.ewma_accuracy <= 0.0:
        return Hero.unknown(
            "no baseline",
            "because the profile's accuracy EWMA is still zero — there is "
            "nothing to measure the recent runs against",
            "ewma_accuracy is seeded by the first run; a zero here means the "
            "profile predates that field or every run scored 0%.")
    window = accs[-FORM_WINDOW:]
    recent = sum(window) / len(window)
    delta = (recent - profile.ewma_accuracy) * 100.0
    if delta >= FORM_BAND_PP:
        word, tone = "climbing", "good"
    elif delta <= -FORM_BAND_PP:
        word, tone = "dipping", "warn"
    else:
        word, tone = "holding", "fg"
    return Hero(
        f"{delta:+.1f}pp", word,
        f"because your last {len(window)} runs average {recent:.1%} against a "
        f"{profile.ewma_accuracy:.1%} baseline EWMA",
        tone,
        f"Mean accuracy of the last {len(window)} runs minus ewma_accuracy "
        f"(half-life {half_life:g} runs), in percentage points. The size "
        "controller holds "
        "accuracy inside your archetype's band, so a dip often means the last "
        "plan raised difficulty rather than that you got worse.")


def _runs_today(history: list[dict]) -> int:
    """Runs stamped with today's local date. Profile timestamps are naive
    local ISO (written from Run.started), so a naive comparison is correct
    here — parsing failures are skipped rather than guessed at."""
    today = datetime.now().date()
    n = 0
    for h in history:
        try:
            if datetime.fromisoformat(str(h.get("ts", ""))).date() == today:
                n += 1
        except ValueError:
            continue
    return n


def _volume_word(n: int) -> tuple[str, str]:
    """Session-length cue for a run count. Thresholds are disclosed in the
    card's tooltip — they are a rough cue, not a fatigue measurement."""
    if n <= 0:
        return "idle", "fg"
    if n < 15:
        return "light", "good"
    if n < 40:
        return "steady", "good"
    return "heavy", "warn"


def load_hero(profile: PlayerProfile, fatigue: dict | None,
              min_runs: int, telemetry_on: bool = True) -> Hero:
    """Session fatigue when a report carried a trusted reading, else volume.

    `min_runs` is Settings.fatigue_min_runs. Below it SessionFatigueTracker
    reports score 0 at level "fresh", which means "no evidence yet" and NOT
    "you are fresh" — rendering that as a fatigue reading would be inventing
    signal, so the volume fallback runs instead and says what is missing.
    """
    fat = fatigue or {}
    needed = max(int(min_runs), 2)          # same floor the tracker applies
    runs = int(fat.get("runs", 0) or 0)
    if runs >= needed:
        score = float(fat.get("score", 0.0))
        level = str(fat.get("level", "fresh"))
        trend = float(fat.get("trend", 0.0))
        tone = {"fatigued": "bad", "declining": "warn"}.get(level, "good")
        # FatigueState.trend is a BADNESS slope (analysis/fatigue.py:34):
        # positive means overshoot and flick duration are RISING, i.e. quality
        # falling. Printing it as "flick quality trends +X%" stated the exact
        # opposite of the evidence — the direction word has to carry the sign.
        if trend > 0:
            drift = f"degrading {trend:.1%} per run"
        elif trend < 0:
            drift = f"improving {abs(trend):.1%} per run"
        else:
            drift = "flat"
        return Hero(
            f"{score:.0%}", level,
            f"because flick quality is {drift} across {runs} telemetry runs "
            f"this session",
            tone,
            fat.get("message")
            or ("Theil-Sen slope of a per-run badness composite (overshoot "
                "rate + flick duration) across this session's telemetry "
                "runs, normalized by the session median."))
    if not profile.history:
        return Hero.unknown(
            "no runs yet",
            "because LOAD reads this session's flick-quality trend, or your "
            "run volume when there is none — this scenario has neither yet",
            "Fatigue needs runs with mouse telemetry; volume falls back to "
            "the run timestamps in this scenario's profile history.")
    today = _runs_today(profile.history)
    word, tone = _volume_word(today)
    plural = "" if today == 1 else "s"
    # Say WHY there is no fatigue reading. Citing a run countdown while
    # telemetry is switched off promises a number that can never arrive.
    why = (f"fatigue needs {needed} runs with telemetry this session "
           f"({runs} so far)" if telemetry_on else
           "fatigue needs mouse telemetry, which is switched off")
    return Hero(
        f"{today} run{plural}", word,
        f"because {today} run{plural} logged today in this scenario — {why}",
        tone,
        "Runs stamped with today's date in THIS scenario's profile history, "
        "so a day spread across several scenarios counts only the selected "
        "one. Buckets: 1-14 light, 15-39 steady, 40+ heavy — a session-length "
        "cue, not a fatigue measurement.")


def localized_hero(metric: str, hero: Hero) -> Hero:
    """Translate a dashboard reading at the display boundary.

    The calculation helpers intentionally keep their stable English output for
    CLI/tests; only the GUI card receives the localized copy.
    """
    words = {
        "no runs yet": "暂无记录", "too few runs": "数据不足",
        "no baseline": "暂无基线", "cold start": "冷启动",
        "learning": "学习中", "calibrating": "校准中", "dialed in": "已就绪",
        "climbing": "上升", "dipping": "下降", "holding": "稳定",
        "idle": "未训练", "light": "轻量", "steady": "适中", "heavy": "较高",
        "fresh": "状态良好", "declining": "正在下降", "fatigued": "已疲劳",
    }
    word = words.get(hero.word, hero.word)

    if metric == "readiness":
        if hero.value == "—":
            because = "该场景还没有历史记录；完成第一局后开始计算准备度"
        else:
            detail = hero.because.removeprefix("because ")
            detail = re.sub(r"runs (\d+)/(\d+)", r"基线局数 \1/\2", detail)
            detail = re.sub(
                r"regions (\d+)/(\d+) mapped \((\d+)\+ observations each\)",
                r"已映射区域 \1/\2（每区至少 \3 次观测）", detail)
            detail = detail.replace("bias evidence collected", "方向偏差证据已收集")
            detail = re.sub(
                r"bias evidence (\d+)/(\d+) measurements",
                r"方向偏差证据 \1/\2 次", detail)
            because = "依据：" + detail
        tip = ("准备度由三部分组成：稳定的准确率基线占 50%，有足够证据的墙面区域占 35%，"
               "方向偏差证据占 15%。数值越高，个性化调整越可靠。")
    elif metric == "form":
        if hero.word == "too few runs":
            count = re.search(r"this scenario has (\d+)", hero.because)
            because = (f"至少需要 {FORM_MIN_RUNS} 局才能比较近期表现；当前只有 "
                       f"{count.group(1) if count else 0} 局")
        elif hero.word == "no baseline":
            because = "准确率的指数移动平均仍为 0，目前没有可供比较的个人基线"
        else:
            match = re.search(
                r"last (\d+) runs average ([\d.]+%) against a ([\d.]+%)", hero.because)
            because = (f"依据：最近 {match.group(1)} 局平均准确率 {match.group(2)}，"
                       f"个人基线为 {match.group(3)}"
                       if match else "依据：近期准确率与个人长期基线的差值")
        tip = ("近期状态 = 最近几局的平均准确率 − 个人准确率 EWMA 基线，单位为百分点。"
               "下降也可能是上一轮提高了难度，并不一定代表能力退步。")
    else:
        value_match = re.match(r"(\d+) runs?", hero.value)
        value = f"{value_match.group(1)} 局" if value_match else hero.value
        if hero.value == "—":
            because = "该场景尚无记录，暂时无法判断本次训练的疲劳趋势或训练量"
        elif hero.value.endswith("%"):
            drift = re.search(
                r"flick quality is (.+) across (\d+) telemetry runs", hero.because)
            drift_text = drift.group(1) if drift else ""
            drift_text = re.sub(r"degrading ([\d.]+%) per run", r"每局恶化 \1", drift_text)
            drift_text = re.sub(r"improving ([\d.]+%) per run", r"每局改善 \1", drift_text)
            drift_text = drift_text.replace("flat", "基本持平")
            because = (f"依据：本次训练 {drift.group(2)} 局遥测中，甩枪质量{drift_text}"
                       if drift else "依据：本次训练的甩枪质量趋势")
        else:
            today = value_match.group(1) if value_match else "0"
            because = f"今天在该场景完成了 {today} 局；遥测样本足够后会改为显示疲劳趋势"
        tip = ("有足够鼠标遥测时，训练负荷显示本次训练中甩枪质量的疲劳趋势；"
               "证据不足时退回显示今天在当前场景的训练局数。")
        translated_message = {
            "Flick quality has dropped steadily this session — a 10-15 minute break will likely gain you more than grinding on.":
                "本次训练的甩枪质量持续下降；休息 10–15 分钟通常比继续硬练更有效。",
            "Flick quality is trending down — consider a short break soon.":
                "甩枪质量正在下降，建议近期安排一次短暂休息。",
        }
        tip = translated_message.get(hero.tip, tip)
        return Hero(value, word, because, hero.tone, tip)
    return Hero(hero.value, word, because, hero.tone, tip)


def dashboard_message_zh(message: str) -> str:
    """Localize the user-visible session log without changing CLI messages."""
    msg = str(message)
    exact = {
        "pick a scenario first": "请先选择一个场景",
        "stopped": "已停止分析",
        "mouse telemetry: recording (Raw Input)": "鼠标遥测：正在通过 Raw Input 录制",
        "mouse telemetry: unavailable on this OS — skipping": "鼠标遥测：当前系统不支持，已跳过",
        "clip capture: recording (ring buffer)": "录像片段：正在使用循环缓冲录制",
        "clip capture: dxcam/opencv missing — pip install kovadapt[clips]":
            "录像片段：缺少 dxcam/opencv；请安装 kovadapt[clips]",
        "KovaaK's is already running": "KovaaK's 已经在运行",
        "launching KovaaK's through Steam…": "正在通过 Steam 启动 KovaaK's…",
        "launching KovaaK's requires Windows": "启动 KovaaK's 需要 Windows",
    }
    if msg in exact:
        return exact[msg]
    if msg.startswith("switching from "):
        match = re.match(r"switching from (.+) to (.+)…", msg)
        return f"正在从 {match.group(1)} 切换到 {match.group(2)}…" if match else msg
    if msg.startswith("scenario file not found:"):
        return "找不到场景文件：" + msg.split(":", 1)[1].strip()
    if msg.startswith("could not create adaptive variant:"):
        return "无法创建自适应版本：" + msg.split(":", 1)[1].strip()
    if msg.startswith("watcher error:"):
        return "后台分析发生错误：" + msg.split(":", 1)[1].strip()
    if msg.startswith("could not reach Steam"):
        detail = _after_parenthetical(msg)
        return "无法连接 Steam；请确认 Steam 已安装并正在运行" + detail
    if msg.startswith("jumping to "):
        return "正在切换到场景 " + msg.removeprefix("jumping to ")
    if msg.startswith("launching KovaaK's into "):
        return "正在启动 KovaaK's 并进入场景 " + msg.removeprefix("launching KovaaK's into ")
    if msg.startswith("no adaptive variant yet for "):
        name = msg.removeprefix("no adaptive variant yet for ").split(" —", 1)[0]
        return f"{name} 尚无自适应版本；请先开始分析"
    if msg.startswith("could not write playlist:"):
        return "无法写入播放列表：" + msg.split(":", 1)[1].strip()
    if msg.startswith("KovaaK's launching —"):
        if "staged to resume" in msg:
            return ("正在启动 KovaaK's；kovadapt 播放列表已尝试设为自动恢复（实验功能）。"
                    "若未自动进入，请打开 Playlists → kovadapt adaptive")
        return ("正在启动 KovaaK's；进入游戏后打开 Playlists → kovadapt adaptive，"
                "或浏览本地场景开始训练")
    if msg.startswith("created "):
        name = msg.split(" —", 1)[0].removeprefix("created ")
        return f"已创建 {name}；在 KovaaK's 中开始训练后，每局结束都会继续适配"
    if msg.startswith("watching "):
        match = re.match(r"watching (.+) for (.+) runs", msg)
        return (f"正在监测 {match.group(1)} 中的 {match.group(2)} 训练记录"
                if match else "正在监测新的训练记录")
    if msg.startswith("error processing "):
        return "处理训练记录时出错：" + msg.removeprefix("error processing ")
    if msg.startswith("clip capture: unavailable"):
        detail = _after_parenthetical(msg)
        return "录像片段不可用；将继续训练但不录制片段" + detail
    if msg.startswith("neural scorer: checkpoint loaded"):
        return "神经评分器：已载入检查点，将把甩枪质量写入报告"
    if msg.lstrip().startswith("note: shadow transition not logged"):
        return "提示：未写入影子策略训练记录；自适应不受影响，但 ML 训练集不会累积"
    if msg.lstrip().startswith("fatigue:"):
        if "dropped steadily" in msg:
            return "疲劳：本次训练甩枪质量持续下降，建议休息 10–15 分钟"
        return "疲劳：甩枪质量正在下降，建议近期短暂休息"
    if msg.lstrip().startswith("neural scorer error:"):
        return "神经评分器出错：" + msg.split(":", 1)[1].strip()
    if msg.lstrip().startswith("archetype:"):
        body = msg.split(":", 1)[1].strip()
        for old, new in (("clicking", "点击"), ("tracking", "跟枪"),
                         ("switching", "目标切换")):
            body = body.replace(old, new)
        body = body.replace("this run disagrees with the guess made before it was played",
                            "本局数据修正了训练前的类型推测")
        return "训练类型：" + body
    if msg.lstrip().startswith("directional bias:"):
        return "方向偏差：此场景允许玩家移动，左右甩枪代价会受到反向走位影响，因此不在这里测量"
    if msg.lstrip().startswith("note: region "):
        region = _first_group(r"region (r\d+c\d+)", msg, "所选区域")
        return f"提示：{region} 没有目标生成点，未应用重点权重，也不会给该区域记入训练收益"
    if re.match(r"\[\d\d:\d\d:\d\d\] run #", msg):
        match = re.match(r"(\[[^]]+\]) run #(\d+) acc=([\d.]+%) score=([\d.]+) -> (.+)", msg)
        if match:
            return (f"{match.group(1)} 第 {match.group(2)} 局：准确率 {match.group(3)}，"
                    f"分数 {match.group(4)} → {match.group(5)}")
    if msg.lstrip().startswith("analysis:"):
        return "复盘报告已生成；详细结果请查看“复盘分析”页面"
    return msg


def _first_group(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else default


def _after_parenthetical(text: str) -> str:
    match = re.search(r"\(([^)]+)\)", text)
    return f"（系统原始错误：{match.group(1)}）" if match else ""


def _mono_css(px: int) -> str:
    """`font-family`/`font-size` for a widget-level sheet.

    theme.py's app QSS opens with `* { font-family: "Segoe UI"; font-size:
    13px }`, and a QSS font property beats QWidget.setFont() — so structural
    mono type has to be restated here or it silently renders as 13px body
    text (the same reason overlay.py pairs setFont with a font-size sheet).
    """
    return f'font-family: "{theme.mono_family()}"; font-size: {px}px;'


class HeroStat(QFrame):
    """One hero card: eyebrow, big mono numeral, state word, because-clause.

    Colors are resolved in `_render()` from theme.current(), and `restyle()`
    is just another render — the card caches its `Hero`, never a palette.
    """

    # theme.mono() snaps to CELL_SIZES (max 24); hero scale is exactly 2x the
    # largest, which keeps Cascadia Mono's advance and row pitch on integer
    # pixels — the whole reason those sizes were measured in the first place.
    NUMERAL_PX = theme.CELL_SIZES[-1] * 2

    def __init__(self, name: str, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self._hero = Hero.unknown("", "")

        # setFont carries what QSS cannot (capitalization, letter spacing) and
        # gives the label honest metrics; the size and family have to come
        # from a widget-level sheet as well, because theme.py's app-wide
        # `* { font-family: "Segoe UI"; font-size: 13px }` outranks setFont —
        # without the sheet the hero numeral rendered at body size.
        eyebrow = theme.mono(12)
        eyebrow.setCapitalization(QFont.AllUppercase)
        eyebrow.setLetterSpacing(QFont.AbsoluteSpacing, 1.4)
        self.name = QLabel(name)
        self.name.setFont(eyebrow)
        self.name.setProperty("dim", True)      # color still cascades from QSS
        self.name.setStyleSheet(_mono_css(12))

        numeral = theme.mono(theme.CELL_SIZES[-1])
        numeral.setPixelSize(self.NUMERAL_PX)
        self.value = QLabel("—")
        self.value.setFont(numeral)

        self.word = QLabel("")
        self.word.setWordWrap(True)
        self.because = QLabel("")
        self.because.setWordWrap(True)
        self.because.setProperty("dim", True)

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.value, 0, Qt.AlignBottom)
        row.addWidget(self.word, 1, Qt.AlignBottom)
        # AlignBottom aligns bottom EDGES, and a label's bottom edge is one
        # DESCENT below its baseline — so the state word hung well below the
        # 48px numeral it qualifies. The gap is largest here of anywhere in
        # the app, because the numeral is the biggest text there is.
        theme.align_baselines(self.word, numeral, theme.body_font())

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(4)
        lay.addWidget(self.name)
        lay.addLayout(row)
        lay.addWidget(self.because, 1)

    # ------------------------------------------------------------------
    def show_hero(self, hero: Hero) -> None:
        self._hero = hero
        self._render()

    def restyle(self, *_pal) -> None:
        self._render()

    def _render(self) -> None:
        pal = theme.current()
        h = self._hero
        self.value.setText(h.value)
        # An unresolved reading stays dim: a dash in accent would read as a
        # value the model actually has.
        known = h.value != "—"
        self.value.setStyleSheet(
            f"{_mono_css(self.NUMERAL_PX)} "
            f"color: {pal.accent if known else pal.fg_dim};")
        tone = {"good": pal.good, "warn": pal.warn, "bad": pal.bad,
                "accent": pal.accent}.get(h.tone, pal.fg_dim)
        self.word.setText(h.word)
        self.word.setStyleSheet(f"color: {tone};")
        self.because.setText(h.because)
        tip = h.tip or h.because
        # Qt does not inherit tooltips, so the whole card has to carry it or
        # hovering the numeral itself explains nothing.
        for w in (self, self.name, self.value, self.word, self.because):
            w.setToolTip(tip)


class Dashboard(QWidget):
    report_ready = Signal(object)   # re-emitted RunReport for the analysis tab

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.worker: WatcherWorker | None = None
        self._watching: str = ""
        self._stopping = False
        self._pending: tuple[str, str] | None = None   # (scenario, "play"|"watch")
        self._last_log = ""
        self._last_profile: PlayerProfile | None = None
        self._install: launcher.InstallStatus | None = None
        self._fatigue: dict = {}          # newest report's FatigueState, if any
        self._shown: str | None = None    # scenario the heroes currently read
        self.overlay = OverlayWindow(settings)

        hint = HintBar(settings, (
            "选择场景后点击<b>开始自适应训练</b>。kovadapt 会监测每一局，"
            "并在局间重新生成 <b>[Adaptive]</b> 版本。下方三个指标分别表示："
            "<b>准备度</b>（模型校准程度）、<b>近期状态</b>（相对个人基线的近期"
            "准确率）和<b>训练负荷</b>（本次训练的疲劳程度）。训练开始后会"
            "自动记录鼠标遥测，界面以 <b>REC</b> 圆点提示。"))

        # ------------------------------------------------- the three heroes
        self.heroes: dict[str, HeroStat] = {}
        hero_row = QHBoxLayout()
        hero_row.setSpacing(14)
        for key, name in (("readiness", "Readiness"), ("form", "Form"),
                          ("load", "Load")):
            card = HeroStat(tr(name))
            self.heroes[key] = card
            hero_row.addWidget(card, 1)

        # ------------------------------------------------- the one trend
        self.trend = viz.AsciiTrend(title="每局准确率 · 当前场景",
                                    fmt="{:.0%}")
        self.trend_caption = QLabel(
            "左侧为较早的训练，右侧为最新一局。自适应系统会调整目标大小，"
            "让准确率维持在适合当前训练类型的区间，因此曲线不需要一直冲向 100%。")
        self.trend_caption.setWordWrap(True)
        self.trend_caption.setProperty("dim", True)
        # A sparkline reads at a glance; past ~220px the extra height is just
        # a taller gap between points and it starts to dominate the page.
        self.trend.setMaximumHeight(220)
        self.trend_w = QWidget()
        tv = QVBoxLayout(self.trend_w)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(6)
        tv.addWidget(self.trend)
        tv.addWidget(self.trend_caption)

        # ---------------------------------------------------- play controls
        self.install_lbl = QLabel("正在检查安装状态…")
        self.install_lbl.setProperty("dim", True)
        self.launch_btn = QPushButton(tr("Launch KovaaK's"))
        self.launch_btn.clicked.connect(self._launch_game)
        self.play_btn = QPushButton(tr("▶  Play adaptive task"))
        self.play_btn.setProperty("accent", True)
        self.play_btn.setToolTip(
            "开始记录，加入自适应播放列表并启动 KovaaK's。进入游戏后打开 "
            "Playlists → kovadapt adaptive 开始训练")
        self.play_btn.clicked.connect(self.play)

        self.scenario = QComboBox()
        self.scenario.setEditable(True)
        self.refresh_btn = QPushButton(tr("Refresh"))
        self.refresh_btn.clicked.connect(self.refresh_scenarios)
        self.start_btn = QPushButton(tr("Start adapting"))
        self.start_btn.clicked.connect(self.toggle)

        # full-width Play panel: the column's lead panel, generous rows
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        row1.addWidget(QLabel(tr("Scenario:")))
        row1.addWidget(self.scenario, 1)
        row1.addWidget(self.refresh_btn)
        row1.addWidget(self.start_btn)
        self.rec_lbl = QLabel("")
        self.rec_lbl.setTextFormat(Qt.RichText)
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        row2.addWidget(self.install_lbl, 1)
        row2.addWidget(self.rec_lbl)
        row2.addWidget(self.launch_btn)
        row2.addWidget(self.play_btn)
        play_box = QGroupBox(tr("Play"))
        pv = QVBoxLayout(play_box)
        pv.setContentsMargins(14, 12, 14, 14)
        pv.setSpacing(12)
        pv.addLayout(row1)
        pv.addLayout(row2)

        # -------------------------------------------------- overlay controls
        self.ov_toggle = QPushButton(tr("Overlay"))
        self.ov_toggle.setCheckable(True)
        self.ov_toggle.setToolTip(
            "在游戏画面上方显示训练状态卡片（仅限无边框或窗口模式）")
        self.ov_toggle.toggled.connect(self._toggle_overlay)
        self.ov_unlock = QPushButton(tr("Unlock"))
        self.ov_unlock.setCheckable(True)
        self.ov_unlock.setToolTip("解锁后可以拖动浮窗；锁定后鼠标点击会穿透浮窗")
        self.ov_unlock.toggled.connect(self._unlock_overlay)
        self.ov_opacity = CatSlider(30, 100)
        self.ov_opacity.setValue(int(settings.overlay_opacity * 100))
        self.ov_opacity.setToolTip("调整浮窗透明度")
        self.ov_opacity.valueChanged.connect(
            lambda v: self.overlay.set_opacity(v / 100))
        # Debounced persist: keyboard/wheel changes never fire sliderReleased.
        self._opacity_save = QTimer(self)
        self._opacity_save.setSingleShot(True)
        self._opacity_save.setInterval(800)
        self._opacity_save.timeout.connect(self._save_settings)
        self.ov_opacity.valueChanged.connect(
            lambda _v: self._opacity_save.start())
        self.ov_auto = QCheckBox(tr("Show when a session starts"))
        self.ov_auto.setChecked(settings.overlay_autoshow)
        self.ov_auto.toggled.connect(self._set_autoshow)

        # overlay controls: their own full-width row in the column
        ov_box = QGroupBox(tr("Overlay"))
        ov = QHBoxLayout(ov_box)
        ov.setContentsMargins(14, 12, 14, 14)
        ov.setSpacing(12)
        ov.addWidget(self.ov_toggle)
        ov.addWidget(self.ov_unlock)
        ov.addWidget(self.ov_opacity, 1)
        ov.addWidget(self.ov_auto)

        # ------------------------------------------------- log, behind a lid
        self.log_btn = QPushButton(LOG_LABEL)
        self.log_btn.setCheckable(True)
        self.log_btn.setProperty("flat", True)
        self.log_btn.setFont(theme.mono(12))
        self.log_btn.setStyleSheet(_mono_css(12))   # see _mono_css: QSS wins
        self.log_btn.setToolTip(
            "训练日志：监测消息、自适应版本生成结果和游戏启动结果")
        self.log_btn.toggled.connect(self._toggle_log)
        log_row = QHBoxLayout()
        log_row.addWidget(self.log_btn)
        log_row.addStretch(1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(150)
        self.log.hide()

        lay = QVBoxLayout(self)
        # ZERO, explicitly. Every section view inherited Qt's ~9px default
        # layout margin, while the section's own H1, its divider rule and
        # every panel sit flush to shell._Section's column — so bare page
        # text was the only thing indented, and lined up with nothing on the
        # screen. The column IS the measure; panels pad their own contents.
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        lay.addWidget(hint)
        lay.addLayout(hero_row)
        # Slack goes to a trailing stretch, not to a panel. Handing it to the
        # trend made a sparkline 1400px tall on a tall window — the same
        # failure as the hint bar that once filled a whole section. Every
        # panel here has a natural height; only the empty space below should
        # grow.
        lay.addWidget(self.trend_w)
        lay.addWidget(play_box)
        lay.addWidget(ov_box)
        lay.addLayout(log_row)
        lay.addWidget(self.log)
        lay.addStretch(1)

        self.refresh_scenarios()
        # Show what the model already knows, on launch and whenever the pick
        # changes. Without these the heroes only ever filled when a session
        # started or a report landed, so the home screen greeted every launch
        # with em-dashes no matter how much history the scenario had.
        self.scenario.currentTextChanged.connect(
            lambda _t: self.refresh_profile(self._picked_scenario()))
        self.refresh_profile(self._picked_scenario())
        self.restyle(theme.current())
        self._refresh_install()

    @property
    def last_profile(self) -> PlayerProfile | None:
        """Freshest loaded profile (updated before report_ready re-emits)."""
        return self._last_profile

    # ------------------------------------------------------------- theming
    def restyle(self, pal=None) -> None:
        pal = pal or theme.current()
        for card in self.heroes.values():
            card.restyle(pal)
        self.trend.restyle()            # reads theme.current() at paint time
        self._render_install()
        # The REC dot bakes a palette color into rich text; without this it
        # kept the old theme's color across a switch.
        self._render_rec(self.worker is not None)
        self.overlay.restyle()

    # ------------------------------------------------------------- install
    def _refresh_install(self) -> None:
        self._install = launcher.check_install(self.s)
        self._render_install()

    def _render_rec(self, watching: bool) -> None:
        """Make the automatic telemetry recorder visible: it starts with
        every session — nobody should have to wonder how to turn it on."""
        pal = theme.current()
        if not watching:
            self.rec_lbl.setText("")
            return
        if self.s.telemetry_enabled:
            self.rec_lbl.setText(
                f"<span style='color:{pal.bad}'>●</span> 正在录制鼠标遥测")
            self.rec_lbl.setToolTip(
                "本次训练已自动开始 Raw Input 录制；每局报告都会生成甩枪分析")
        else:
            self.rec_lbl.setText(
                f"<span style='color:{pal.warn}'>○</span> 鼠标遥测已关闭")
            self.rec_lbl.setToolTip(
                "请在“自适应设置”中启用“分析时记录原始鼠标输入”，以获得甩枪分析和区域证据")

    def _render_install(self) -> None:
        st = self._install
        if st is None:
            return
        pal = theme.current()
        color = pal.good if st.ok else pal.bad
        if not st.root_found:
            description = "未找到 KovaaK's 安装；请设置 KOVAAKS_ROOT 或检查路径设置"
        else:
            bits = ["已找到 KovaaK's"]
            bits.append("Steam 清单正常" if st.manifest_found else "缺少 Steam 安装清单")
            if not st.steam_found:
                bits.append("未找到 Steam 客户端")
            if st.game_running:
                bits.append("游戏正在运行")
            description = " · ".join(bits)
        self.install_lbl.setText(
            f"<span style='color:{color}'>●</span> {description}")
        self.install_lbl.setTextFormat(Qt.RichText)
        self.launch_btn.setEnabled(st.ok)
        self.play_btn.setEnabled(st.ok)

    # --------------------------------------------------------------- play
    def _launch_game(self) -> None:
        self._refresh_install()
        msg, _ok = launcher.launch_game()
        self.append_log(msg)

    def _picked_scenario(self) -> str:
        """Current selection, normalized: typing the adaptive name directly
        must not create '[Adaptive] [Adaptive]' compounding variants."""
        return self.scenario.currentText().strip().removesuffix(ADAPTIVE_SUFFIX)

    def play_scenario(self, name: str) -> None:
        """Entry point for the scenario browser: select + play."""
        self.scenario.setCurrentText(name)
        self.play()

    def watch_scenario(self, name: str) -> None:
        if self.worker is not None and self._watching != name:
            self._switch_to(name, "watch")
            return
        self.scenario.setCurrentText(name)
        if self.worker is None:
            self.toggle()

    def _switch_to(self, name: str, kind: str) -> None:
        """Stop the running session and start `name` once it is down —
        changing scenarios must never require a manual stop first."""
        self._pending = (name, kind)
        self.append_log(f"switching from {self._watching!r} to {name!r}…")
        if not self._stopping:
            self.toggle()               # the stop branch

    def play(self) -> None:
        """Watch + queue playlist + deep-link into the adaptive scenario."""
        name = self._picked_scenario()
        if not name:
            self.append_log("pick a scenario first")
            return
        if self._stopping:
            self._pending = (name, "play")   # start this once the stop lands
            return
        if self.worker is not None and self._watching != name:
            self._switch_to(name, "play")
            return
        if self.worker is None and not self._start_watch(name):
            return
        self._refresh_install()
        msg, _ok = launcher.play_adaptive(self.s, name)
        self.append_log(msg)

    # ------------------------------------------------------------- watching
    def toggle(self) -> None:
        if self.worker is not None:
            self._stopping = True
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.start_btn.setText(tr("Stopping…"))
            self.play_btn.setEnabled(False)
            return
        name = self._picked_scenario()
        if not name:
            self.append_log("pick a scenario first")
            return
        self._start_watch(name)

    def _start_watch(self, name: str) -> bool:
        """Create the worker, bootstrap the adaptive .sce synchronously (so
        Play can deep-link immediately), and start watching."""
        w = WatcherWorker(self.s, name, parent=self)
        if not w.watcher.base_sce_path().is_file():
            self.append_log(f"scenario file not found: {w.watcher.base_sce_path()}")
            w.deleteLater()
            return False
        # Connect before bootstrap so its log lines actually land in the log.
        w.message.connect(self.append_log)
        w.report_ready.connect(self._on_report)
        w.stopped.connect(self._on_stopped)
        w.finished.connect(w.deleteLater)   # don't retain dead QThreads
        if not w.watcher.adaptive_sce_path().is_file():
            try:
                w.watcher.bootstrap()
            except Exception as exc:
                self.append_log(f"could not create adaptive variant: {exc}")
                w.deleteLater()
                return False
        self.worker = w
        self._watching = name
        w.start()
        self.start_btn.setText(tr("Stop"))
        self.scenario.setEnabled(False)
        self._render_rec(True)
        # The watcher's fatigue tracker is session-scoped and restarts here;
        # LOAD must not keep citing the previous session's trend.
        self._fatigue = {}
        self.refresh_profile(name)
        self.overlay.start_session(name + ADAPTIVE_SUFFIX)
        if self.s.overlay_autoshow and not self.ov_toggle.isChecked():
            self.ov_toggle.setChecked(True)
        return True

    def _on_stopped(self) -> None:
        self.worker = None
        self._watching = ""
        self._stopping = False
        self.start_btn.setEnabled(True)
        self.start_btn.setText(tr("Start adapting"))
        self.scenario.setEnabled(True)
        self._render_install()          # re-enable Play per install status
        self._render_rec(False)
        self.overlay.stop_session()
        self.append_log("stopped")
        if self._pending is not None:
            name, kind = self._pending
            self._pending = None
            self.scenario.setCurrentText(name)
            if kind == "play":
                self.play()
            else:
                self.toggle()

    def _on_report(self, rep) -> None:
        # Stash the fatigue reading BEFORE the reload: LOAD reads it, and
        # refresh_profile is what repaints the heroes.
        self._fatigue = dict(getattr(rep, "fatigue", None) or {})
        self.refresh_profile(self._watching or self._picked_scenario())
        self.overlay.on_report(rep, self._last_profile)
        self.report_ready.emit(rep)

    # -------------------------------------------------------------- overlay
    def _toggle_overlay(self, on: bool) -> None:
        if on:
            self.overlay.show_overlay()
        else:
            if self.ov_unlock.isChecked():
                self.ov_unlock.setChecked(False)
            self.overlay.hide()

    def _unlock_overlay(self, on: bool) -> None:
        if on and not self.ov_toggle.isChecked():
            self.ov_toggle.setChecked(True)
        self.overlay.set_unlocked(on)
        self.ov_unlock.setText(tr("Lock") if on else tr("Unlock"))

    def _set_autoshow(self, on: bool) -> None:
        self.s.overlay_autoshow = on
        self._save_settings()

    def _save_settings(self) -> None:
        try:
            self.s.save()
        except OSError:
            pass

    def shutdown(self) -> None:
        self.overlay.close()

    # ------------------------------------------------------------------
    def refresh_scenarios(self) -> None:
        cur = self.scenario.currentText()
        # Rebuilding the list is not a user pick. clear() emits
        # currentTextChanged("") and setCurrentText() emits it again, and the
        # picker is wired to refresh_profile — so an unguarded Refresh ran
        # refresh_profile("") mid-session, which sees a changed scenario and
        # discards the session's fatigue reading. Pressing Refresh must not
        # erase LOAD.
        blocked = self.scenario.blockSignals(True)
        try:
            self.scenario.clear()
            if self.s.scenarios_dir.is_dir():
                names = sorted(
                    p.stem for p in self.s.scenarios_dir.glob("*.sce")
                    if not p.stem.endswith(ADAPTIVE_SUFFIX)
                )
                self.scenario.addItems(names)
            if cur:
                self.scenario.setCurrentText(cur)
        finally:
            self.scenario.blockSignals(blocked)
        # Only a genuine change of pick reloads (and legitimately clears it).
        if self._picked_scenario() != self._shown:
            self.refresh_profile(self._picked_scenario())

    def _toggle_log(self, on: bool) -> None:
        self.log.setVisible(on)
        self.log_btn.setText(LOG_LABEL)     # opening clears the unread mark

    def append_log(self, line: str) -> None:
        line = dashboard_message_zh(line)
        if line == self._last_log:      # never spam identical lines
            return
        self._last_log = line
        self.log.appendPlainText(line)
        if not self.log_btn.isChecked():
            # The log is collapsed by default, so failures ("scenario file not
            # found") would otherwise land completely silently.
            self.log_btn.setText(LOG_UNREAD)

    def refresh_profile(self, base_name: str) -> None:
        if base_name != self._shown:
            # A fatigue reading belongs to the session that produced it —
            # carrying it onto another scenario's profile would make LOAD
            # cite runs that never happened there.
            self._fatigue = {}
            self._shown = base_name
        prof = PlayerProfile.load(base_name + ADAPTIVE_SUFFIX, self.s.profile_path)
        self._last_profile = prof
        self.heroes["readiness"].show_hero(
            localized_hero(
                "readiness",
                readiness_hero(prof, self.s.region_cols * self.s.region_rows)))
        self.heroes["form"].show_hero(
            localized_hero("form", form_hero(prof, self.s.ewma_half_life)))
        self.heroes["load"].show_hero(
            localized_hero(
                "load",
                load_hero(prof, self._fatigue, self.s.fatigue_min_runs,
                          telemetry_on=self.s.telemetry_enabled)))
        accs = [float(h.get("accuracy", 0.0)) for h in prof.history[-60:]]
        if len(accs) >= 2:
            # Cite the real run numbers: the list is sliced, so without this
            # the axis called run 78 "run 1" on a long profile.
            self.trend.set_data(accs, tag=f"{accs[-1]:.0%}",
                                first_run=len(prof.history) - len(accs) + 1)
        else:
            self.trend.clear()      # AsciiTrend renders "not enough runs yet"
