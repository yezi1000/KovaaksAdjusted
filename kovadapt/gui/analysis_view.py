"""Post-run analysis tab: the run's four headline numbers, then the charts,
then the Coach.

Reading order is the design. A KPI strip readable in a second leads; the
character-art charts (gui/viz.py) follow side by side, each carrying a
TAKEAWAY title that states what its data shows rather than what the chart
is; the Coach lands last, folded to its two most severe cards with the rest
one click away — nothing is dropped, because the citations are the product.

Every takeaway is derived from the values actually on screen and falls back
to the neutral descriptor the moment the data stops supporting a claim: a
title that overstates is worse than no title at all.

pyqtgraph remains only inside TrajectoryReplay's canvas."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# The KPI reads reuse the Coach's own cutoffs and its region wording rather
# than restating them: a tile reading "clean" above a card that says
# "overshooting, then repairing" would be one body of evidence contradicting
# itself, and two spellings of the same region would read as two findings.
from ..analysis.insights import (
    _CORRECTIONS_CHAIN,
    _CORRECTIONS_CLEAN,
    _MIN_FLICKS,
    _OVERSHOOT_HIGH,
    Insight,
    _region_words,
    generate_insights,
)
from ..analysis.movement import (MIN_FLICK_DEG, apply_shot_outcomes,
                                 movement_heatmap, segment_flicks)
from ..analysis.report import (
    RunReport,
    apply_flick_metrics,
    input_degraded,
)
from ..analysis.sens import min_flick_counts
from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile
from ..telemetry.trace import MouseTrace
from . import theme, viz
from .i18n import tr
from .onboarding import HintBar
from .replay import TrajectoryReplay

# Neutral chart descriptors — what the chart IS. Used whenever the run's own
# numbers do not clear the floors below.
_BIAS_TITLE = "flick quality by direction · lower is better"
_TRAVEL_TITLE = "aim travel around engagements"
_DEFICIT_TITLE = "weakness by wall region · brighter = weaker"
_TREND_TITLE = "accuracy over runs"

# Captions say how to READ the chart; the title carries what it says. Both
# stay short — the page used to open with ~700 characters of caption prose.
#
# What the chart IS, with no clause about ink that may not be on it. This is
# also the cold-start caption: before any run is loaded the panel reads
# "waiting for flick data", and a caption pointing at a red bar there sends
# the reader hunting for a colour no run has produced yet.
_BIAS_CAPTION_BASE = ("Cost per flick = overshoot + 0.15 x corrective "
                      "submovements, one bar per direction.")
_BIAS_CAPTION = f"{_BIAS_CAPTION_BASE} The red bar is this run's worst."
# The smallest flick cost whose bar is allowed to fill the track. Cost is
# overshoot as a FRACTION OF THE FLICK'S OWN AMPLITUDE plus 0.15 per
# corrective submovement, so 0.05 is "overshot by five percent" — small.
# Calibrated against the run reports on this machine (a thin sample: 5 runs
# carrying bias, 15 bars): the per-run maximum ran 0.072 at the 1st
# percentile and 0.153 at the median, so this floor rescales none of them and
# only ever bites on data that is already noise.
_BIAS_COST_FLOOR = 0.05

# On a degraded run nothing is red — the worst-bar highlight answers to the
# same input-health gate as the title and the footer ratio — so the caption
# must stop pointing at a colour that is not on screen.
_BIAS_CAPTION_DEGRADED = (f"{_BIAS_CAPTION_BASE} No direction is marked worst: "
                          "this run's input timing is too noisy to rank them.")
_BIAS_CAPTION_NO_COST = (f"{_BIAS_CAPTION_BASE} No direction is marked worst: "
                         "nothing this run cost enough to rank.")
# "Nothing cost enough to rank" is a MEASUREMENT, and a run with no flicks
# has not made one — the chart beside this caption reads "waiting for flick
# data" on exactly that run. Two sentences about the same absence, one of
# them claiming more than it can.
_MOMENTS_EMPTY = "Notable moments appear here after a run."
_BIAS_CAPTION_NO_FLICKS = (f"{_BIAS_CAPTION_BASE} No flicks were recorded for "
                           "this run, so there is nothing to rank.")


def _bias_caption(vals: list[float], ns: list[int], degraded: bool,
                  moving_frame: bool = False) -> str:
    """How to read the bars — derived from the same numbers they draw.

    This was a constant ending "the red bar is this run's worst", which is a
    sentence about a colour. Three different runs put no red on that panel at
    all — a degraded run, a run with no flicks, and a run whose every bar
    rounds to 0.00 — and on each of them the caption sent the reader looking
    for ink that is not there.
    """
    if moving_frame:
        return (f"{_BIAS_CAPTION_BASE} No direction is marked worst: this "
                "scenario lets you move, so a side-by-side comparison would "
                "be measuring your strafing.")
    if degraded:
        return _BIAS_CAPTION_DEGRADED
    if sum(ns) == 0:
        return _BIAS_CAPTION_NO_FLICKS
    if all(viz.prints_zero(v) for v in vals):
        return _BIAS_CAPTION_NO_COST
    return _BIAS_CAPTION
_TRAVEL_CAPTION = ("Where the crosshair spent its time around each engagement — "
                   "denser glyphs = more time. Hover a zone for its value.")
_DEFICIT_CAPTION = ("Weakness per wall region as z-scores against this run's own "
                    "average, on the r{row}c{col} grid the engine targets. Hover "
                    "a zone; outlined zones were never measured.")
_TREND_CAPTION = ("Accuracy per run for this scenario — the loop should bend it "
                  "upward without ever pinning it at 100%.")


def analysis_zh(text: str) -> str:
    """Translate analysis display text without changing analytical outputs."""
    exact = {
        _BIAS_TITLE: "各方向甩枪质量 · 越低越好",
        _TRAVEL_TITLE: "交战前后准星移动分布",
        _DEFICIT_TITLE: "墙面区域弱项 · 越亮表示越弱",
        _TREND_TITLE: "准确率变化趋势",
        _BIAS_CAPTION_BASE: "单次甩枪代价 = 过冲比例 + 0.15 × 修正次数；每个方向一条。",
        _BIAS_CAPTION: "单次甩枪代价 = 过冲比例 + 0.15 × 修正次数；红色表示本局代价最高的方向。",
        _BIAS_CAPTION_DEGRADED: "输入时序噪声过大，本局不对各方向进行强弱排名。",
        _BIAS_CAPTION_NO_COST: "本局各方向的代价都低于可判断阈值，因此不标记最差方向。",
        _BIAS_CAPTION_NO_FLICKS: "本局没有记录到甩枪数据，暂时无法比较方向。",
        _TRAVEL_CAPTION: ("显示每次交战前后准星停留的位置；字符越密表示在该区域停留越久。"
                          "这个分布以每次射击点为中心重新对齐，用于观察准星习惯性停留在哪一侧；"
                          "可悬停查看每个区域的具体数值。"),
        _DEFICIT_CAPTION: ("显示各墙面区域相对于本局平均水平的弱项程度（z 分数）。"
                           "字符越亮表示该区域的甩枪代价越高；描边区域尚未获得有效测量。"
                           "区域编号与自适应引擎使用的 r{row}c{col} 网格一致，可悬停查看详情。"),
        _TREND_CAPTION: "当前场景每局准确率；目标是随训练改善，而不是长期固定在 100%。",
        _MOMENTS_EMPTY: "完成一局后，值得复盘的关键片段会显示在这里。",
        "hit rate": "命中率", "kills": "击杀", "flicks": "次甩枪",
        "no-telemetry": "无遥测", "kills/s": "击杀/秒",
        "not-measurable": "无法测量", "above-band": "高于区间",
        "below-band": "低于区间", "in-band": "位于区间内",
        "faster": "更快", "slower": "更慢", "steady": "稳定",
        "no-baseline": "暂无基线", "noisy-input": "输入噪声较大",
        "thin-data": "样本不足", "repaired": "过冲后反复修正",
        "clean": "干净", "mixed": "表现混合", "ms": "毫秒",
    }
    if text in exact:
        return exact[text]
    patterns = (
        (r"(\d+) flicks", r"\1 次甩枪"),
        (r"this scenario lets you move — sides are not comparable here", "该场景允许角色移动，左右方向不宜直接比较"),
        (r"input timing too noisy to compare directions this run", "本局输入时序噪声过大，无法比较方向"),
        (r"only (\d+) left / (\d+) right flicks — too few to call a side", r"仅记录到左侧 \1 次、右侧 \2 次甩枪，样本不足"),
        (r"vertical flicks cost most — ([\d.]+) vs ([\d.]+) horizontal", r"垂直甩枪代价最高：\1，对比水平方向 \2"),
        (r"no overshoot or correction cost in either direction", "各方向均未出现明显过冲或修正代价"),
        (r"no measurable cost in either direction this run", "本局各方向都没有可测量的甩枪代价"),
        (r"only your left flicks carry any cost — (.+)", r"只有左侧甩枪出现明显代价：\1"),
        (r"only your right flicks carry any cost — (.+)", r"只有右侧甩枪出现明显代价：\1"),
        (r"your left flicks cost ([\d.]+)x more than your right — (.+)",
         r"左侧甩枪代价是右侧的 \1 倍：\2"),
        (r"your right flicks cost ([\d.]+)x more than your left — (.+)",
         r"右侧甩枪代价是左侧的 \1 倍：\2"),
        (r"left and right flicks are even — (.+)", r"左右甩枪代价接近：\1"),
        (r"input timing too noisy to rank zones this run", "本局输入时序噪声过大，无法对区域排序"),
        (r"no zone stands out — all within (.+)", r"没有明显弱区：全部位于平均值附近（\1）"),
        (r"no zone is weaker than your average — (.+) is your strongest, (.+)",
         r"没有区域弱于个人平均；最强区域为 \1（\2）"),
        (r"weakest zone this run: (.+), ([+-].+)", r"本局最弱区域：\1，高于平均值 \2"),
        (r"aim travel is balanced left/right of your shots", "准星在射击点左右两侧的移动较均衡"),
        (r"aim travel leans left — (.+)", r"准星移动偏左：\1"),
        (r"aim travel leans right — (.+)", r"准星移动偏右：\1"),
        (r"accuracy over (\d+) runs — too few to call a direction", r"共 \1 局，暂不足以判断准确率趋势"),
        (r"accuracy flat near (.+) across (\d+) runs", r"最近 \2 局准确率稳定在 \1 附近"),
        (r"accuracy up (\d+) points over (\d+) runs — (.+)", r"最近 \2 局准确率上升 \1 个百分点：\3"),
        (r"accuracy down (\d+) points over (\d+) runs — (.+)", r"最近 \2 局准确率下降 \1 个百分点：\3"),
        (r"(.+) No direction is marked worst: this scenario lets you move, so a side-by-side comparison would be measuring your strafing\.",
         r"\1 该场景允许角色移动，左右比较会同时受到走位影响，因此不标记最差方向。"),
    )
    for pattern, replacement in patterns:
        if re.fullmatch(pattern, text):
            out = re.sub(pattern, replacement, text)
            region_words = {
                "lower left": "左下", "lower center": "下方中央",
                "lower right": "右下", "middle left": "左侧中部",
                "middle right": "右侧中部", "upper left": "左上",
                "upper center": "上方中央", "upper right": "右上",
                "center": "中央", " SD": " 个标准差",
            }
            for old, new in region_words.items():
                out = out.replace(old, new)
            return out
    return text


def report_summary_zh(rep: RunReport) -> str:
    """Concise Chinese summary derived from the same report fields."""
    lines = [f"准确率 {rep.accuracy:.0%}，击杀 {rep.kills}，节奏 {rep.kps:.2f} 次/秒。"]
    if not rep.n_flicks:
        lines.append("本局没有鼠标遥测；开始分析并完成一局后才能进行移动与甩枪复盘。")
        return " ".join(lines)
    if input_degraded(rep):
        ih = rep.input_health or {}
        jitter = float(ih.get("jitter_ms", 0.0) or 0.0)
        polling = float(ih.get("polling_hz_est", 0.0) or 0.0)
        lines.append(
            f"输入时序质量不足（估算回报率 {polling:.0f} Hz、抖动 {jitter:.1f} 毫秒），"
            "因此本局不会给出过冲和方向偏差结论。")
    else:
        bias = float((rep.bias or {}).get("bias_score", 0.0) or 0.0)
        if abs(bias) > 0.15:
            lines.append(f"本局{'左侧' if bias > 0 else '右侧'}甩枪的代价明显更高。")
        else:
            lines.append("本局左右甩枪表现较均衡。")
        if rep.overshoot_rate > 0.25:
            lines.append(f"{rep.overshoot_rate:.0%} 的甩枪出现过冲，可结合下方修正次数判断原因。")
    if rep.mean_flick_ms > 0:
        lines.append(f"平均甩枪时间 {rep.mean_flick_ms:.0f} 毫秒。")
    phases = rep.click_phases or {}
    labeled = int(phases.get("labeled", 0) or 0)
    misses = int(phases.get("misses", 0) or 0)
    raw_misses = int(phases.get("uncorrected_misses", 0) or 0)
    if labeled:
        lines.append(
            f"已把 {labeled} 次点击与逐目标结果对齐；其中 {misses} 次未命中，"
            f"{raw_misses} 次是在没有检测到修正动作时直接击发。")
    return " ".join(lines)


def moment_text_zh(moment: dict) -> str:
    """Translate a persisted notable-moment sentence without rewriting JSON."""
    text = str(moment.get("text", ""))
    dirs = {
        "right": "向右", "up-right": "右上", "up": "向上",
        "up-left": "左上", "left": "向左", "down-left": "左下",
        "down": "向下", "down-right": "右下",
    }
    patterns = (
        (r"Overshot a ([a-z-]+) flick by (\d+)% of its distance, then corrected (\d+)x before shooting\.",
         lambda m: (f"一次{dirs.get(m.group(1), m.group(1))}甩枪过冲了移动距离的 "
                    f"{m.group(2)}%，射击前又修正了 {m.group(3)} 次。")),
        (r"Hesitated on a ([a-z-]+) target: (\d+) micro-corrections over (\d+)ms before committing\.",
         lambda m: (f"处理{dirs.get(m.group(1), m.group(1))}目标时出现犹豫：在 "
                    f"{m.group(3)} 毫秒内进行了 {m.group(2)} 次微调后才击发。")),
        (r"Slow ([a-z-]+) acquisition: (\d+)ms for a (\d+)-count flick \(bottom 10% of this run's pace\)\.",
         lambda m: (f"{dirs.get(m.group(1), m.group(1))}目标获取偏慢："
                    f"{m.group(3)} 计数的甩枪耗时 {m.group(2)} 毫秒，"
                    "属于本局最慢的 10%。")),
        (r"Reference: a clean (\d+)-count ([a-z-]+) flick — (\d+)ms, no overshoot\. This is your benchmark\.",
         lambda m: (f"参考动作：一次干净的 {m.group(1)} 计数"
                    f"{dirs.get(m.group(2), m.group(2))}甩枪，耗时 {m.group(3)} 毫秒，"
                    "没有过冲；可将它作为本局基准。")),
        (r"Missed a ([a-z-]+) flick without a corrective submovement before firing\.",
         lambda m: (f"一次{dirs.get(m.group(1), m.group(1))}甩枪在没有进行修正动作时"
                    "直接击发并未命中。")),
    )
    for pattern, render in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            return render(match)
    kinds = {
        "overshoot": "过冲片段", "hesitation": "犹豫与连续修正片段",
        "slow_flick": "较慢甩枪片段", "clean_flick": "干净甩枪参考片段",
        "unconfirmed_miss": "无修正直接点空片段",
    }
    # Older reports may contain free-form sentences that predate the known
    # templates.  Preserve those details instead of replacing them with only
    # a generic kind label; use the label solely when the report has no text.
    return text or kinds.get(str(moment.get("kind", "")), "关键片段")

# Claim floors. A takeaway has to clear one of these or the chart keeps its
# neutral title; each is kovadapt's own editorial calibration except the
# per-side flick count, which is analysis.directional_bias's own gate.
_EVEN_RATIO = 1.25       # side-vs-side cost ratio that stops reading as "even"
_MIN_SIDE_FLICKS = 3     # per-side floor (matches analysis.directional_bias)
_TREND_MIN_RUNS = 6      # runs before a direction is called on the sparkline
_TREND_STEP = 0.02       # accuracy points that count as a move, not noise
_LOPSIDED_SHARE = 0.55   # occupancy share that stops reading as balanced
_PACE_STEP = 0.05        # pace change vs the EWMA that counts as faster/slower

# Coach folding: severity order, then how many cards stay unfolded.
_SEVERITY_RANK = {"warning": 0, "attention": 1, "info": 2}
_COACH_FOLD = 2


def _severity_color(severity: str, pal) -> str:
    """The dot beside a Coach card, derived from the card's RANK above.

    It was `{"warning": pal.warn, "attention": pal.bad}.get(sev, pal.good)`,
    which runs backwards against the very ordering it sits next to: `warning`
    is rank 0, sorted first and always left unfolded, and it got amber, while
    the less severe `attention` got red. With two warnings the default folded
    Coach showed two amber dots and no red at all — the only red card was
    behind "show all".

    And `info` fell through to pal.good, so a card that merely states
    something wore the all-clear green. It is neutral now: the ramp is
    ordinal and info is not on it.
    """
    return (pal.bad, pal.warn, pal.fg_dim)[
        min(_SEVERITY_RANK.get(severity, 2), 2)]


def localized_insight(ins: Insight) -> tuple[str, str, str, str, str]:
    """Chinese card copy keyed by the stable knowledge-base diagnostic id."""
    copy = {
        "dx-input-health": (
            "输入质量正在影响分析",
            "鼠标回报率或输入时序抖动不足以可靠识别细小修正动作，因此本局会暂停相关诊断。",
            "先检查鼠标回报率、USB 连接和后台程序，再重新录制一局。"),
        "dx-acc-above-band": (
            "近期准确率持续高于训练区间",
            "连续多局准确率过高通常表示任务已经过于舒适，继续保持当前难度带来的学习信息有限。",
            "保持动作质量，同时主动提高节奏，或让自适应系统逐步缩小目标。"),
        "dx-acc-below-band": (
            "近期准确率低于训练区间下限",
            "连续多局低于下限表示当前难度可能超过可稳定练习的范围，容易固化失控动作。",
            "先降低节奏并恢复干净命中；自适应系统也会逐步放大目标。"),
        "dx-overshoot-control": (
            "过冲后进行了多次修正",
            "甩枪经常越过目标，随后又通过连续小动作回拉，说明主要问题更接近制动与控制。",
            "练习一次甩枪后只做一次小幅修正，减少反复拉扯。"),
        "dx-static-unconfirmed-miss": (
            "需要修正时却直接击发",
            "逐目标结果表明，多次未命中发生在首次甩枪之后、尚未检测到修正动作之前；"
            "这与一次到位的直接命中不同。",
            "先做快速但可控的首次定位，接近目标时减速；不确定时完成一次小幅修正并确认后再点击，"
            "稳定命中后再逐步提高节奏。"),
        "dx-overshoot-strategic": (
            "过冲但几乎不修正，可能是主动速度策略",
            "在准确率仍位于区间内时，过冲而不反复回拉可能来自速度任务中的扫过式击发。",
            "先保持当前策略，继续观察跨局准确率和修正次数，不必仅因过冲立即降速。"),
        "dx-tracking-jitter": (
            "跟枪轨迹存在抖动",
            "跟枪过程中出现较多修正子动作，准星可能在目标两侧频繁来回调整。",
            "降低无效小修正，练习更连续、平滑的跟随动作。"),
        "dx-switch-corrections": (
            "目标切换需要额外修正",
            "首次甩枪没有直接落到新目标，后续修正增加了每次目标获取的时间成本。",
            "把重点放在首次落点，先追求一次到位，再提高切换速度。"),
        "dx-bias": (
            "左右方向存在持续差异",
            "多局数据表明某一侧甩枪的过冲与修正代价长期更高，而不只是单局波动。",
            "适当增加较弱一侧的训练量；自适应场景已会向该侧增加移动时间。"),
        "dx-region-deficit": (
            "跨局数据发现稳定弱区",
            "墙面上的某个区域在多局中持续弱于个人平均水平，模型会把更多生成权重分配到那里。",
            "保持正常训练，让模型继续验证该区域；弱项改善后旧证据会逐渐衰减。"),
        "dx-fatigue": (
            "本次训练出现疲劳趋势",
            "过冲和甩枪时间在本次训练中同时恶化，继续硬练的收益可能已经下降。",
            "安排一次短暂休息，回来后比较新的几局是否恢复。"),
        "dx-fitts-progress": (
            "总分持平，但运动效率仍在改善",
            "分数没有明显变化时，击杀节奏上升或每比特动作时间下降仍代表真实进步。",
            "继续以多局平均趋势评估进步，不要只看单次最高分。"),
        "p-sensitivity-doctrine": (
            "你的灵敏度：同时查看正反两面的证据",
            "高、低灵敏度各有代价，当前证据并不足以支持单向结论；可用范围通常比单个推荐值更宽。",
            "不强制建议任何方向。若要试改，只改变一个变量，并用多局平均结果进行判断。"),
    }
    title, body, prescription = copy.get(
        ins.id,
        (ins.title, "当前报告触发了知识库中的这条规则。", "结合多局趋势验证后再调整训练。"))
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?%?(?:\s?(?:ms|Hz|runs?|kills/s))?", ins.reasoning)
    reasoning = "触发依据：当前报告达到该规则的判断条件"
    if nums:
        reasoning += "；相关数值包括 " + "、".join(nums[:8])
    reasoning += "。"
    confidence = ("高置信度" if "high" in ins.confidence.lower() else
                  "中等置信度" if "medium" in ins.confidence.lower() else
                  "参考结论")
    return title, body, prescription, reasoning, confidence


class _InsightCard(QFrame):
    """One coach insight: severity dot, title, sourced body + prescription,
    and the reasoning/citations chain (the cite-everything rule made visible)."""

    def __init__(self, ins: Insight, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self.insight = ins       # the card's evidence, still queryable after build
        pal = theme.current()
        color = _severity_color(ins.severity, pal)
        title, body_text, prescription, reasoning, confidence = localized_insight(ins)
        head = QLabel(f"<span style='color:{color}'>●</span>  <b>{title}</b>"
                      f"  <span style='color:{pal.fg_dim}'>{confidence}</span>")
        head.setTextFormat(Qt.RichText)
        body = QLabel(f"{body_text}<br><b>训练建议：</b> {prescription}")
        body.setTextFormat(Qt.RichText)
        body.setWordWrap(True)
        why = QLabel(reasoning)
        why.setWordWrap(True)
        why.setProperty("dim", True)
        cites = QLabel(f"{len(ins.sources)} 个来源")
        cites.setProperty("dim", True)
        cites.setToolTip("\n".join(ins.sources))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(3)
        lay.addWidget(head)
        lay.addWidget(body)
        lay.addWidget(why)
        lay.addWidget(cites)


def _mono_css(px: int) -> str:
    """Family + size for mono text, as a widget-level sheet.

    theme.py's app-wide `* { font-family: "Segoe UI"; font-size: 13px }`
    outranks setFont(), so a numeral styled only with setFont renders at body
    size in the running app — and looks right in tests, where no app QSS is
    installed. `px` must stay on theme.CELL_SIZES."""
    return f'font-family: "{theme.mono_family()}"; font-size: {px}px;'


class _KpiTile(QFrame):
    """One headline number: caption, mono value + unit, and a one-word read.

    The read is always a comparison against a stated baseline, and the tile's
    tooltip carries that baseline with the live numbers — the strip is the
    first thing on the page, so it has to be as citable as a Coach card."""

    def __init__(self, caption: str, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self._tone = "dim"
        self.cap = QLabel(caption.upper())
        self.value = QLabel("—")
        self.unit = QLabel("")
        self.read = QLabel("")

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.value, 0, Qt.AlignBottom)
        row.addWidget(self.unit, 0, Qt.AlignBottom)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(2)
        lay.addWidget(self.cap)
        lay.addLayout(row)
        lay.addWidget(self.read)
        self.restyle()

    def set_value(self, value: str, unit: str, read: str, tone: str,
                  why: str) -> None:
        """tone is a palette role name ('good' | 'warn' | 'bad' | 'dim');
        `why` is the tooltip that must justify the read."""
        self.value.setText(value)
        self.unit.setText(analysis_zh(unit))
        self.read.setText(analysis_zh(read))
        self._tone = tone
        self.setToolTip(why)
        self.restyle()

    def restyle(self, *_pal) -> None:
        pal = theme.current()
        tone = {"good": pal.good, "warn": pal.warn,
                "bad": pal.bad}.get(self._tone, pal.fg_dim)
        self.cap.setStyleSheet(
            f"color: {pal.fg_dim}; letter-spacing: 0.8px; {_mono_css(12)}")
        self.value.setStyleSheet(
            f"color: {pal.fg}; font-weight: 700; {_mono_css(24)}")
        self.unit.setStyleSheet(f"color: {pal.fg_dim}; {_mono_css(12)}")
        self.read.setStyleSheet(f"color: {tone}; {_mono_css(12)}")
        # setFont as well as the sheet: the sheet is what survives theme.py's
        # app-wide font rule, setFont is what gives the labels honest metrics
        # (and theme.mono snaps to the sizes Cascadia Mono renders crisply at).
        self.cap.setFont(theme.mono(12))
        self.value.setFont(theme.mono(24, bold=True))
        self.unit.setFont(theme.mono(12))
        self.read.setFont(theme.mono(12))
        # AlignBottom aligns bottom EDGES; a 12px unit beside a 24px numeral
        # therefore hung 3px below the number it belongs to.
        theme.align_baselines(self.unit, theme.mono(24, bold=True),
                              theme.mono(12))


# ---------------------------------------------------------------- takeaways
def _bias_title(vals: list[float], ns: list[int],
                degraded: bool = False, moving_frame: bool = False) -> str:
    """Headline for the direction bars: which side actually costs more.

    `vals`/`ns` are the plotted [left, vertical, right] costs and flick
    counts — the claim comes from the same numbers the bars draw, and only
    once both sides carry the flicks analysis.directional_bias needs before
    it will score them at all.

    `degraded` is the shared input-health gate: these costs are overshoot
    plus corrections, i.e. pure flick microstructure, so a run too noisy to
    diagnose must not get a directional verdict here either — that was a
    "your left flicks cost 3.2x more than your right" headline sitting
    directly above a tile reading "noisy-input"."""
    left, vert, right = vals
    n_left, n_vert, n_right = ns
    if sum(ns) == 0:
        return _BIAS_TITLE
    if moving_frame:
        # The costs are split by which way the CROSSHAIR travelled, and a
        # strafing player counter-moves constantly just to hold a target. On
        # a scenario that lets the player move, a left/right verdict measures
        # which way they strafed as much as which way they aim — so there is
        # no side to call, and the bars stay as data without a claim.
        return "this scenario lets you move — sides are not comparable here"
    if degraded:
        return "input timing too noisy to compare directions this run"
    if n_left < _MIN_SIDE_FLICKS or n_right < _MIN_SIDE_FLICKS:
        return f"only {n_left} left / {n_right} right flicks — too few to call a side"
    hi, lo = max(left, right), min(left, right)
    weak = "left" if left >= right else "right"
    other = "right" if weak == "left" else "left"
    # vert > 0 guard: a run with no cost anywhere satisfies vert >= 1.25 * hi
    # arithmetically, and would headline "vertical flicks cost most — 0.00".
    if vert > 0 and n_vert >= _MIN_SIDE_FLICKS and vert >= _EVEN_RATIO * hi:
        return f"vertical flicks cost most — {vert:.2f} vs {hi:.2f} horizontal"
    if lo <= 0:
        if hi <= 0:
            return "no overshoot or correction cost in either direction"
        return f"only your {weak} flicks carry any cost — {hi:.2f} vs 0.00"
    # Both sides printing 0.00 is not a 1.3x finding. The ratio between two
    # numbers that both round away to nothing is arithmetic on noise, and the
    # sentence carried its own refutation: "your left flicks cost 1.3x more
    # than your right — 0.00 vs 0.00", over a footer already saying there was
    # no cost to compare. viz.prints_zero is the shared rule so the headline
    # and the chart under it cannot disagree about what a zero is.
    if viz.prints_zero(hi) and viz.prints_zero(lo):
        return "no measurable cost in either direction this run"
    ratio = hi / lo
    if ratio >= _EVEN_RATIO:
        return (f"your {weak} flicks cost {ratio:.1f}x more than your {other} "
                f"— {hi:.2f} vs {lo:.2f}")
    return f"left and right flicks are even — {left:.2f} vs {right:.2f}"


def _bias_claim(vals: list[float], ns: list[int],
                degraded: bool) -> tuple[tuple[int, int] | None, int | None]:
    """(pair the panel compares, bar to mark red) — the SAME branches
    `_bias_title` takes, so the sentence and the picture cannot diverge.

    They did diverge. viz derived the red bar as the global argmax while the
    title deliberately ignores a vertical bar unless it dominates by 1.25x, so
    on a real report (left 0.104, vertical 0.115, right 0.042) the headline
    said "your left flicks cost 2.5x more than your right", the red bar and
    the numeral sat on VERTICAL, the footer cited "vertical 0.12 / left 0.10 =
    1.11x" under that 2.5x headline, and the caption told the reader the red
    bar was this run's worst. One panel, four surfaces, two answers.

    Returning None for both is a real answer: it means this run supports no
    ranking, and then nothing is highlighted and no ratio is printed.

    Indices are into [left, vertical, right].
    """
    left, vert, right = vals
    n_left, n_vert, n_right = ns
    if sum(ns) == 0 or degraded:
        return None, None
    if n_left < _MIN_SIDE_FLICKS or n_right < _MIN_SIDE_FLICKS:
        return None, None
    hi_i, lo_i = (0, 2) if left >= right else (2, 0)
    hi, lo = vals[hi_i], vals[lo_i]
    if vert > 0 and n_vert >= _MIN_SIDE_FLICKS and vert >= _EVEN_RATIO * hi:
        return (1, hi_i), 1                       # vertical dominates
    if lo <= 0:
        return (None, None) if hi <= 0 else ((hi_i, lo_i), hi_i)
    if viz.prints_zero(hi) and viz.prints_zero(lo):
        return None, None
    if hi / lo >= _EVEN_RATIO:
        return (hi_i, lo_i), hi_i
    # "left and right are even" — a real finding, with a pair to cite and no
    # worst bar to mark.
    return (hi_i, lo_i), None


def _deficit_title(deficits: dict[str, float], settings: Settings | None,
                   degraded: bool = False) -> str:
    """Headline for the region map: the weakest zone, when one really is.

    Below viz.NOISE_FLOOR the map itself renders flat (that constant is what
    stops noise being stretched across the ramp), so naming a "weakest" zone
    there would claim a finding the picture deliberately refuses to draw.

    `degraded` is the SAME input-health gate the bias panel answers to, and
    this panel is the one microstructure surface that never asked for it. A
    region deficit is `overshoot + 0.15*corrections + 0.25*slowness` z-scored
    (analysis/movement.py) — pure flick microstructure — so on a noisy run
    this title read "WEAKEST ZONE THIS RUN: CENTER, +3.60 SD ABOVE AVERAGE"
    on the same baseline, 700px to the right of "INPUT TIMING TOO NOISY TO
    COMPARE DIRECTIONS THIS RUN". Three of the five real reports on this
    machine land there.
    """
    if degraded:
        return "input timing too noisy to rank zones this run"
    if not deficits:
        return _DEFICIT_TITLE
    key, z = max(deficits.items(), key=lambda kv: kv[1])
    # "All within N SD" has to be tested on the largest ABSOLUTE deviation,
    # the same quantity the map colours. Testing only the maximum let a
    # strongly negative zone (a genuine strength, drawn cool and saturated)
    # sit under a title claiming nothing deviated at all.
    peak = max(abs(v) for v in deficits.values())
    if peak < viz.NOISE_FLOOR:
        return (f"no zone stands out — all within "
                f"{viz.NOISE_FLOOR:.1f} SD of average")
    if z < viz.NOISE_FLOOR:
        # Nothing is weak; the spread that colours the map is on the strong
        # side, so report THAT rather than a weakness the run does not show.
        skey, sz = min(deficits.items(), key=lambda kv: kv[1])
        swhere = _region_words(skey, settings) if settings is not None else skey
        return (f"no zone is weaker than your average — {swhere} is your "
                f"strongest, {sz:+.2f} SD")
    where = _region_words(key, settings) if settings is not None else key
    return f"weakest zone this run: {where}, {z:+.2f} SD above average"


def _travel_title(heat: np.ndarray) -> str:
    """Headline for the occupancy map: which side the crosshair lives on.

    movement_heatmap recenters at every click, so axis 0 is displacement
    left/right of the last shot — the share is of resampled samples, not of
    distance travelled."""
    arr = np.asarray(heat, dtype=float)
    half = arr.shape[0] // 2
    if half == 0:
        return _TRAVEL_TITLE
    low, high = float(arr[:half].sum()), float(arr[-half:].sum())
    total = low + high
    if total <= 0:
        return _TRAVEL_TITLE
    share = max(low, high) / total
    if share < _LOPSIDED_SHARE:
        return "aim travel is balanced left/right of your shots"
    side = "left" if low > high else "right"
    return f"aim travel leans {side} — {share:.0%} of the time on that side"


def _trend_title(accs: list[float]) -> str:
    """Headline for the sparkline: where accuracy is actually going.

    Halves, not endpoints: a single hot or cold run at either end must not
    become a direction. Under _TREND_MIN_RUNS runs no direction is claimed
    at all."""
    n = len(accs)
    if n < 2:
        return _TREND_TITLE
    half = max(n // 2, 1)
    first, second = float(np.mean(accs[:half])), float(np.mean(accs[half:]))
    if n < _TREND_MIN_RUNS:
        return f"accuracy over {n} runs — too few to call a direction"
    delta = second - first
    if abs(delta) < _TREND_STEP:
        return f"accuracy flat near {second:.0%} across {n} runs"
    verb = "up" if delta > 0 else "down"
    return (f"accuracy {verb} {abs(delta) * 100:.0f} points over {n} runs "
            f"— {first:.0%} → {second:.0%}")


def _kind_color(kind: str) -> str:
    pal = theme.current()
    return {"overshoot": pal.bad, "hesitation": pal.bad,
            "unconfirmed_miss": pal.bad,
            "slow_flick": pal.warn, "clean_flick": pal.good}.get(kind, pal.accent)


def _clips_available() -> bool:
    """Whether the [clips] extra (dxcam/opencv) is importable. Lazy on purpose:
    kovadapt.capture must never be pulled in at module import."""
    try:
        from ..capture.clips import CLIPS_AVAILABLE
    except ImportError:
        return False
    return CLIPS_AVAILABLE


def _caption(text: str) -> QLabel:
    """Dim, word-wrapped how-to-read-this caption shown under a plot."""
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setProperty("dim", True)
    return lab


class AnalysisView(QWidget):
    def __init__(self, settings: Settings | None = None, parent=None) -> None:
        super().__init__(parent)
        self.report: RunReport | None = None
        self.trace: MouseTrace | None = None
        self.flicks: list = []
        self._trace_unreadable = False
        self._settings = settings
        self._last_insights: tuple[RunReport, PlayerProfile] | None = None
        self._coach_cards: list[_InsightCard] = []
        self.coach_more: QPushButton | None = None
        self._coach_open = False
        # Set before any showEvent can fire: the caption-levelling pass is
        # scheduled from showEvent as well as resizeEvent, and a missing guard
        # here is an AttributeError raised from inside a Qt event handler,
        # which takes the process down rather than the widget.
        self._levelling = False

        # header
        self.title = QLabel(tr("No run analyzed yet"))
        self.title.setProperty("headline", True)
        self.summary = QLabel(tr("Finish a run while watching (or open a saved report)."))
        self.summary.setWordWrap(True)
        # Secondary now: the strip below carries the numbers this line opens
        # with, so it reads as the caption to the headline rather than a
        # second copy of the run.
        self.summary.setProperty("dim", True)
        open_btn = QPushButton(tr("Open report…"))
        open_btn.clicked.connect(self._open_dialog)

        head = QHBoxLayout()
        head_col = QVBoxLayout()
        head_col.addWidget(self.title)
        head_col.addWidget(self.summary)
        head.addLayout(head_col, 1)
        head.addWidget(open_btn, 0, Qt.AlignTop)

        # ---- KPI strip: the four numbers that answer "how did that go?"
        self.kpi_strip = QWidget()
        self.kpi_strip.setObjectName("tabPage")     # transparent; backdrop shows
        kpi_lay = QHBoxLayout(self.kpi_strip)
        kpi_lay.setContentsMargins(0, 0, 0, 0)
        kpi_lay.setSpacing(12)
        self.kpis: dict[str, _KpiTile] = {}
        for key, cap in (("accuracy", "accuracy"), ("kills", "kills"),
                         ("pace", "pace"), ("flick", "mean flick")):
            tile = _KpiTile(tr(cap))
            self.kpis[key] = tile
            kpi_lay.addWidget(tile, 1)

        # ---- charts: side by side, each with a takeaway title + short caption
        self.bias_bars = viz.AsciiBars(title=analysis_zh(_BIAS_TITLE))
        self.bias_caption = _caption(analysis_zh(_BIAS_CAPTION_BASE))
        self.heat_map = viz.AsciiHeatmap(title=analysis_zh(_TRAVEL_TITLE))
        self.heat_caption = _caption(analysis_zh(_TRAVEL_CAPTION))
        self.trend_spark = viz.AsciiTrend(title=analysis_zh(_TREND_TITLE), fmt="{:.0%}")
        self.trend_caption = _caption(analysis_zh(_TREND_CAPTION))

        bias_w = QWidget()
        bv = QVBoxLayout(bias_w)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(self.bias_bars, 1)
        bv.addWidget(self.bias_caption)
        heat_w = QWidget()
        hv = QVBoxLayout(heat_w)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.addWidget(self.heat_map, 1)
        hv.addWidget(self.heat_caption)
        self.trend_w = QWidget()
        tv = QVBoxLayout(self.trend_w)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.addWidget(self.trend_spark, 1)
        tv.addWidget(self.trend_caption)
        self.trend_w.hide()                    # appears once the profile has history

        # The section is 1400px wide (shell.COLUMN_WIDTHS "wide"): the two
        # per-run charts sit next to each other instead of stacking a narrow
        # column and pushing everything below the fold.
        self.charts = QSplitter(Qt.Horizontal)
        self.charts.addWidget(bias_w)
        self.charts.addWidget(heat_w)
        self.charts.setSizes([700, 700])
        self.charts.setMinimumHeight(300)

        # ---- notable moments + replay, also side by side
        self.moments = QListWidget()
        # Moment text is a full sentence, and the panel is ~330px wide: elided
        # to one line it read "Overshot a right flick by 36% of its distance,
        # then corrected 1x before" and pushed a horizontal scrollbar under the
        # list. Wrapping is what makes the sentence readable; ElideNone stops
        # Qt truncating instead of wrapping, and the off switch on the
        # horizontal bar keeps the wrap authoritative.
        self.moments.setWordWrap(True)
        self.moments.setTextElideMode(Qt.ElideNone)
        self.moments.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.moments.setResizeMode(QListWidget.Adjust)
        self.moments.currentRowChanged.connect(self._select_moment)
        self.full_btn = QPushButton(tr("Whole run"))
        self.full_btn.setEnabled(False)
        self.full_btn.setToolTip("回放整局鼠标轨迹，而不是单个关键片段")
        self.full_btn.clicked.connect(self._show_full_run)
        self.clip_btn = QPushButton(tr("Play video clip"))
        self.clip_btn.setEnabled(False)
        self.clip_btn.clicked.connect(self._play_clip)
        self.clip_hint = QLabel("")           # why clips are off, when they are
        self.clip_hint.setWordWrap(True)
        self.clip_hint.setProperty("dim", True)
        self.replay = TrajectoryReplay()
        # Cold start says what it is. On every launch this page opened with a
        # 246,448px moments list and a 531,912px replay canvas that were each
        # 100.000% one flat colour, zero ink, while the two chart panels
        # beside them explained themselves perfectly well. The sentence
        # already existed — replay.clear() takes one — it was just never
        # reachable except through show_report.
        self.replay.clear("尚未加载训练数据 — 请完成一局，或从上方打开保存的报告")
        empty = QListWidgetItem(analysis_zh(_MOMENTS_EMPTY))
        empty.setFlags(Qt.NoItemFlags)          # not selectable: it is not a moment
        self.moments.addItem(empty)

        mo_box = QGroupBox(tr("Notable moments"))
        mo_lay = QVBoxLayout(mo_box)
        mo_lay.addWidget(self.moments, 1)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.full_btn)
        btn_row.addWidget(self.clip_btn, 1)
        mo_lay.addLayout(btn_row)
        mo_lay.addWidget(self.clip_hint)
        self._update_clip_state(-1)
        rep_box = QGroupBox(tr("Trajectory replay"))
        rep_lay = QVBoxLayout(rep_box)
        rep_lay.addWidget(self.replay)
        self.detail = QSplitter(Qt.Horizontal)
        self.detail.addWidget(mo_box)
        self.detail.addWidget(rep_box)
        self.detail.setSizes([440, 900])
        self.detail.setMinimumHeight(380)

        # ---- coach last: folded to its two most severe cards
        self.coach_box = QGroupBox(
            tr("Coach — every insight shows its evidence and sources"))
        self.coach_lay = QVBoxLayout(self.coach_box)
        self.coach_lay.setSpacing(10)
        self.coach_box.hide()

        # NO setHandleWidth here. The theme sizes a splitter handle to 5px and
        # FILLS it with pal.border, so widening it to 14 for "room to breathe"
        # did not add space — it added a 14px column of border colour between
        # two group boxes that already carry their own 1px frames, and ran it
        # 26px ABOVE the panels, up beside the titles where it divides nothing.
        # Breathing room belongs in the layout spacing below, which has it.
        lay = QVBoxLayout(self)
        # ZERO, explicitly. Every section view inherited Qt's ~9px default
        # layout margin, while the section's own H1, its divider rule and
        # every panel sit flush to shell._Section's column — so bare page
        # text was the only thing indented, and lined up with nothing on the
        # screen. The column IS the measure; panels pad their own contents.
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        if settings is not None:
            lay.addWidget(HintBar(settings, (
                "顶部四个数字概括当前这一局；每张图的标题会直接说明数据结论。点击"
                "关键片段可单独回放该次甩枪（绿色表示干净，红色表示过冲），也可以点击"
                "<b>整局回放</b>查看全部轨迹。训练建议中的结论均附带依据和来源，"
                "悬停来源数量即可查看。")))
        lay.addLayout(head)
        lay.addWidget(self.kpi_strip)
        lay.addWidget(self.charts)
        lay.addWidget(self.trend_w)
        lay.addWidget(self.detail, 1)
        lay.addWidget(self.coach_box)

    # ------------------------------------------------------------------
    def restyle(self, *_pal) -> None:
        # the viz widgets read theme.current() at paint time — update() is all
        for chart in (self.bias_bars, self.heat_map, self.trend_spark):
            chart.restyle()
        for tile in self.kpis.values():
            tile.restyle()
        self.replay.restyle()
        if self._last_insights is not None:
            self._fill_insights(*self._last_insights)   # cards bake colors
        if self.report is not None:
            for r in range(self.moments.count()):
                it = self.moments.item(r)
                # By ROW this painted every moment in the NEXT moment's kind
                # colour and dropped the caption's dim grey. The list is
                # severity-sorted and the clean reference flick scores 1.0,
                # so a theme switch on a degraded run rendered "this is your
                # benchmark" in the BAD colour. On this page the colour IS
                # the evidence.
                i = self._moment_index(r)
                if i < 0:
                    # the caption row: re-dim it to the NEW palette, since
                    # skipping would leave the old theme's grey behind
                    it.setForeground(QColor(theme.current().fg_dim))
                    continue
                it.setForeground(QColor(_kind_color(self.report.notable[i]["kind"])))

    # ------------------------------------------------------------------
    def set_trends(self, trends) -> None:
        """Cross-session SkillTrends from the boot worker."""
        self._trends = trends

    def resizeEvent(self, event) -> None:      # noqa: N802  (Qt override)
        super().resizeEvent(event)
        # AFTER the layout pass, not during. resizeEvent arrives before the
        # layout has resized the children, so measuring here reads the
        # PREVIOUS width and levels to the wrong number — it happened to look
        # right inside the shell, where an earlier pass had already settled
        # the widths, and was wrong on a freshly shown view.
        if not self._levelling:
            self._levelling = True
            # `self` as the context object: Qt cancels the callback if the
            # widget is destroyed first. Without it, closing the page with a
            # pass still queued calls a bound method on a freed C++ object —
            # an access violation that kills the process, not an exception.
            QTimer.singleShot(0, self, self._level_captions)

    def showEvent(self, event) -> None:        # noqa: N802  (Qt override)
        super().showEvent(event)
        if not self._levelling:
            self._levelling = True
            # `self` as the context object: Qt cancels the callback if the
            # widget is destroyed first. Without it, closing the page with a
            # pass still queued calls a bound method on a freed C++ object —
            # an access violation that kills the process, not an exception.
            QTimer.singleShot(0, self, self._level_captions)

    def _level_captions(self) -> None:
        """Keep the two side-by-side chart captions the same height.

        The charts sit in two independent columns of a splitter, each taking
        the leftover space with stretch 1 and its caption underneath. The
        captions are different lengths: below about 1490px the travel caption
        wraps to a second line while the bias caption stays on one, so it
        takes 17px more — and steals exactly that much from its own chart.
        The two canvases then have their tops aligned and their bottoms 17px
        apart, which is what makes the pair read as crooked.

        Measured: caption delta and chart delta are the same number at every
        width (17px at 1180 and 1360, 0 at 1490 and above). The audit filed
        this as chart baselines being off and had the direction backwards —
        nothing is wrong with the baselines, and a fix aimed at them would
        have moved the one thing that was already right.

        Reset before measuring, or the floor set on the last pass is what
        gets measured on this one and the captions only ever grow.
        """
        self._levelling = False
        caps = [c for c in (getattr(self, "bias_caption", None),
                            getattr(self, "heat_caption", None)) if c is not None]
        if len(caps) < 2:
            return
        for c in caps:
            c.setMinimumHeight(0)
        need = max(c.heightForWidth(c.width()) if c.width() > 0
                   else c.sizeHint().height() for c in caps)
        for c in caps:
            if c.minimumHeight() != need:
                c.setMinimumHeight(need)

    def show_report(self, rep: RunReport, trace: MouseTrace | None = None,
                    profile: PlayerProfile | None = None) -> None:
        self.report = rep
        self.trace = trace
        profile = self._resolve_profile(rep, profile)
        self._coach_open = False           # a new run opens folded
        # The trace loads and the report is re-derived BEFORE anything is
        # filled: the KPI tiles, the trend and the Coach all read `rep`, and
        # filling them from the stored numbers and then correcting `rep`
        # underneath leaves four surfaces disagreeing with the two below them.
        self._trace_unreadable = False
        if trace is None and rep.trace_file and Path(rep.trace_file).is_file():
            try:
                self.trace = MouseTrace.load(rep.trace_file)
            except Exception:
                # A .npz truncated by a crash mid-write, or a half-synced
                # cloud folder, raised BadZipFile straight out of here and
                # took the whole page down — including the stats half of the
                # report, which does not depend on the trace at all. The
                # report renders without telemetry instead, which is an
                # already-supported state.
                #
                # The file is NOT quarantined the way a corrupt profile is: a
                # profile can be rebuilt from the stats history, a recording
                # cannot be rebuilt from anything. Leave it for the user.
                self.trace = None
                self._trace_unreadable = True
        # flicks aren't serialized in the report — recompute from the trace at
        # the SAME floor the rest of the page uses, or the replay overlay marks
        # flicks the page's own numbers never counted.
        # The floor for THIS run, from the sensitivity the run itself records.
        # Falling back to the settings field is the weaker path and is what
        # made the v0.5.2 floor 1.96x wrong here; `rep.deg_per_count` carries
        # the game's own answer, and a run recorded before that field existed
        # is exactly the case the settings fallback is for.
        floor = (MIN_FLICK_DEG / rep.deg_per_count if rep.deg_per_count > 0
                 else min_flick_counts(self._settings))
        self.flicks = (segment_flicks(self.trace, min_amplitude=floor)
                       if self.trace is not None and len(self.trace) > 10 else [])
        apply_shot_outcomes(self.flicks, rep.shot_outcomes)
        # ...and when the SAVED numbers came from a different floor, that is
        # exactly what happens: the file says one thing about a run and the
        # overlay draws another. This report was written at 0.33 degrees,
        # where the overshoot ratio measures segmentation error rather than
        # aim; on the five real runs here it inverted the left/right verdict,
        # so the page opened saying "your left side is measurably weaker,
        # 2.48x" about a run today's method reads as right-side weakness.
        #
        # Same rule as summary_text_for one call below: a stored value about a
        # threshold is a cache of a judgement, and the trace is still on disk.
        # Re-derived in memory only — the file on disk is the record of what
        # was measured then, and rewriting it would destroy that.
        self._rederived = False
        if (self.flicks and rep.flick_floor_deg != MIN_FLICK_DEG
                and self._settings is not None):
            rep = replace(rep)          # never mutate the caller's report
            apply_flick_metrics(rep, self.flicks,
                                cols=self._settings.region_cols,
                                rows=self._settings.region_rows)
            rep.flick_floor_deg = MIN_FLICK_DEG
            self.report = rep
            self._rederived = True

        self._fill_kpis(rep, profile)
        self._fill_trend(profile)
        self._fill_insights(rep, profile)
        self.title.setText(f"{rep.scenario} — {rep.started_iso.replace('T', ' ')[:19]}")
        # RE-DERIVED, not the stored string. See analysis.report.
        # summary_text_for: this is the only headline on the page that was
        # persisted, so a saved report kept asserting whatever was true when
        # it was written.
        head = report_summary_zh(rep)
        if self._rederived:
            # Say it. A page that quietly disagrees with the JSON a user can
            # open in a text editor is worse than one that never corrected it:
            # the number changed, nothing on screen explains why, and the file
            # is still sitting there saying the old thing.
            head += (f" 这些数值已按当前 {MIN_FLICK_DEG:g}° 的甩枪阈值从轨迹重新计算；"
                     "保存报告使用的是旧阈值。")
        self.summary.setText(head)
        self._draw_bias(rep)
        self._draw_heat(rep)
        has_trace = self.trace is not None and len(self.trace) > 1
        self.full_btn.setEnabled(has_trace)
        # Moments FIRST: filling the list selects the top moment, which loads
        # that window into the replay. Loading the full run first as well cost
        # a whole extra path() + decimate + setData pass on every report — up
        # to ~6.8k points thrown away — and, before the ordering was fixed,
        # left the highlighted row and the replay describing different
        # segments. The full run is loaded only when nothing selected one.
        self._fill_moments(rep)
        if not has_trace:
            self.replay.clear("轨迹文件已损坏，无法读取"
                              if self._trace_unreadable else "本局没有鼠标轨迹")
        elif self.moments.currentRow() < 0:
            self.replay.load(self.trace, label="整局", flicks=self.flicks)
        self._update_clip_state(self._moment_index(self.moments.currentRow()))

    def load_report_file(self, path: Path | str) -> None:
        self.show_report(RunReport.load(path))

    # ------------------------------------------------------------------
    def _resolve_profile(self, rep: RunReport,
                         profile: PlayerProfile | None) -> PlayerProfile | None:
        """The profile this run belongs to — reloaded from disk on the
        saved-report path, where the caller has none to hand us."""
        if profile is not None or self._settings is None:
            return profile
        name = rep.scenario
        if not name.endswith(ADAPTIVE_SUFFIX):
            name += ADAPTIVE_SUFFIX
        return PlayerProfile.load(name, self._settings.profile_path)

    def _fill_kpis(self, rep: RunReport, profile: PlayerProfile | None) -> None:
        """The four headline numbers. Every read names the baseline it is a
        comparison against, and the tooltip carries that baseline with the
        live numbers — no bare adjectives."""
        arche = (profile.archetype if profile is not None else "") or "clicking"
        eff = (self._settings.for_archetype(arche)
               if self._settings is not None else None)

        # accuracy vs the archetype's band (the size controller's setpoint)
        if eff is None:
            self.kpis["accuracy"].set_value(
                f"{rep.accuracy:.0%}", "hit rate", "—", "dim",
                "尚未载入设置，因此无法确定本局应当与哪个准确率训练区间比较。")
        else:
            lo, hi = eff.target_accuracy_low, eff.target_accuracy_high
            if rep.accuracy > hi:
                read, tone = "above-band", "warn"
            elif rep.accuracy < lo:
                read, tone = "below-band", "warn"
            else:
                read, tone = "in-band", "good"
            arch_name = {"clicking": "点击", "tracking": "跟枪",
                         "switching": "目标切换"}.get(arche, arche)
            band_note = ("点击训练的 85%–95% 区间有直接资料依据"
                         if arche == "clicking" else
                         f"{arch_name}训练区间是 kovadapt 根据相同控制规律作出的外推")
            self.kpis["accuracy"].set_value(
                f"{rep.accuracy:.0%}", "hit rate", read, tone,
                f"本局命中率 {rep.accuracy:.1%}；尺寸控制器会把{arch_name}训练维持在 "
                f"{lo:.0%}–{hi:.0%} 区间（{band_note}）。")

        # kills, plus how much of the run telemetry actually saw
        n = rep.n_flicks
        self.kpis["kills"].set_value(
            str(rep.kills), "kills",
            f"{n} flicks" if n else "no-telemetry", "dim" if n else "warn",
            f"统计文件记录了 {rep.kills} 次击杀；鼠标遥测从本局识别出 {n} 次甩枪。"
            "下方图表均以这些甩枪为依据；没有遥测时只显示游戏统计。")

        # pace against the profile's own EWMA
        base = profile.ewma_kps if profile is not None else 0.0
        arche = (profile.archetype if profile is not None else "") or ""
        if rep.kills == 0:
            # PACE IS NOT MEASURABLE HERE, and saying "0.00 kills/s · no
            # baseline" was two lies at once. Tracking scenarios use
            # INVINCIBLE targets, so KovaaK's reports Kills: 0 by design and
            # kills-per-second is structurally undefined rather than slow —
            # on a real 95-scenario library that is 49 scenarios showing a
            # permanent fake zero. Worse, the no-baseline branch below then
            # explained it with "the EWMA needs a second run", which is a
            # specific and wrong reason on a profile with fifty.
            if arche == "tracking":
                why = ("此跟枪场景的目标不会死亡，因此 KovaaK's 不记录击杀，"
                       "击杀/秒不适用于这个场景；应查看准确率与甩枪质量。")
            else:
                why = ("KovaaK's 本局没有记录到击杀，因此无法计算击杀/秒；"
                       "上方准确率仍由已经完成的射击计算。")
            self.kpis["pace"].set_value("—", "kills/s", "not-measurable",
                                        "dim", why)
        # run_count > 1, not > 0: observe_run seeds every EWMA to the first
        # run's own value, so at run_count == 1 this compares the run against
        # itself and reports a confident "steady · +0%".
        elif profile is not None and profile.run_count > 1 and base > 0:
            delta = rep.kps / base - 1.0
            if delta >= _PACE_STEP:
                read, tone = "faster", "good"
            elif delta <= -_PACE_STEP:
                read, tone = "slower", "warn"
            else:
                read, tone = "steady", "dim"
            why = (f"本局 {rep.kps:.2f} 击杀/秒，对比 {profile.run_count} 局形成的"
                   f"个人 EWMA 基线 {base:.2f}（{delta:+.0%}）。kovadapt 使用 "
                   f"±{_PACE_STEP:.0%} 作为判断节奏变化的校准阈值。")
        else:
            read, tone = "no-baseline", "dim"
            runs = profile.run_count if profile is not None else 0
            why = (f"本局 {rep.kps:.2f} 击杀/秒；此场景目前只有 {runs} 局历史。"
                   "节奏 EWMA 由第一局初始化，至少完成第二局后才会成为可比较的个人基线。")
        if rep.kills:
            self.kpis["pace"].set_value(f"{rep.kps:.2f}", "kills/s",
                                        read, tone, why)

        # mean flick time, read through the Coach's own microstructure gate —
        # the SAME gate, not a copy of its cutoffs. Reading overshoot and
        # corrections without the input-health check let this tile call a run
        # "repaired" on the same screen where the Coach reported that
        # microstructure diagnoses were suppressed for it.
        degraded = input_degraded(rep)
        if degraded:
            read, tone = "noisy-input", "warn"
        elif n < _MIN_FLICKS:
            read, tone = "thin-data", "dim"
        elif (rep.overshoot_rate > _OVERSHOOT_HIGH
              and rep.mean_corrections >= _CORRECTIONS_CHAIN):
            read, tone = "repaired", "bad"
        elif (rep.overshoot_rate <= _OVERSHOOT_HIGH
              and rep.mean_corrections <= _CORRECTIONS_CLEAN):
            read, tone = "clean", "good"
        else:
            read, tone = "mixed", "warn"
        why = (f"{n} 次甩枪的平均时间为 {rep.mean_flick_ms:.0f} 毫秒；"
               f"其中 {rep.overshoot_rate:.0%} 出现过冲，每次平均包含 "
               f"{rep.mean_corrections:.1f} 个修正子动作。{_OVERSHOOT_HIGH:.0%} 过冲阈值、"
               f"{_CORRECTIONS_CHAIN:.0f} 次修正阈值和至少 {_MIN_FLICKS} 次甩枪的样本要求，"
               "与训练建议使用的微观动作判断标准一致（kovadapt 校准值）。")
        if degraded:
            why = (f"{n} 次甩枪的平均时间为 {rep.mean_flick_ms:.0f} 毫秒，但本局输入时序"
                   "噪声过大，无法可靠读取甩枪微观结构，因此页面不会给出过冲或方向结论；"
                   "甩枪用时本身仍然有效。")
        phases = rep.click_phases or {}
        if int(phases.get("labeled", 0) or 0):
            why += (
                f" 已对齐 {int(phases.get('labeled', 0))} 次逐目标点击："
                f"直接命中 {int(phases.get('direct_hits', 0))} 次，"
                f"一次修正后命中 {int(phases.get('corrected_hits', 0))} 次，"
                f"连续修正后命中 {int(phases.get('repair_chain_hits', 0))} 次，"
                f"无修正直接点空 {int(phases.get('uncorrected_misses', 0))} 次。")
        self.kpis["flick"].set_value(
            f"{rep.mean_flick_ms:.0f}" if rep.mean_flick_ms > 0 else "—", "ms",
            read, tone, why)

    # ------------------------------------------------------------------ coach
    def _fill_insights(self, rep: RunReport, profile: PlayerProfile | None) -> None:
        self._clear_coach()
        if profile is None or self._settings is None or profile.run_count == 0:
            # Drop the cached pair too: restyle() refills from it, so leaving
            # the previous run's (rep, profile) here meant any theme or accent
            # change resurrected that run's Coach cards and trend underneath
            # the CURRENT run's header — stale advice presented as live.
            self._last_insights = None
            self.coach_box.hide()
            return
        insights = generate_insights(
            rep, profile, self._settings,
            trends=getattr(self, "_trends", None))
        self._last_insights = (rep, profile)
        # Severity first, generate_insights' own actionability order within a
        # severity (sorted is stable). The two cards left showing when the
        # section is folded have to be the two that matter most.
        for ins in sorted(insights, key=lambda i: _SEVERITY_RANK.get(i.severity, 9)):
            card = _InsightCard(ins)
            self._coach_cards.append(card)
            self.coach_lay.addWidget(card)
        if len(self._coach_cards) > _COACH_FOLD:
            self.coach_more = QPushButton("")
            self.coach_more.setProperty("flat", True)
            self.coach_more.setCursor(Qt.PointingHandCursor)
            self.coach_more.clicked.connect(self._toggle_coach)
            self.coach_lay.addWidget(self.coach_more, 0, Qt.AlignLeft)
        self._apply_fold()
        self.coach_box.setVisible(bool(insights))

    def _clear_coach(self) -> None:
        while self.coach_lay.count():
            item = self.coach_lay.takeAt(0)
            if item.widget():
                # Unparent before deleting: takeAt only drops the layout item,
                # so a card left parented stays painted until the event loop
                # gets round to the deletion.
                item.widget().setParent(None)
                item.widget().deleteLater()
        self._coach_cards = []
        self.coach_more = None

    def _apply_fold(self) -> None:
        """Fold the Coach to its most severe cards. The rest are built and
        kept — their citations are the product, not decoration — and only
        hidden behind the disclosure."""
        for i, card in enumerate(self._coach_cards):
            card.setVisible(self._coach_open or i < _COACH_FOLD)
        if self.coach_more is not None:
            self.coach_more.setText(
                "收起" if self._coach_open
                else f"显示全部（{len(self._coach_cards)}）")

    def _toggle_coach(self) -> None:
        self._coach_open = not self._coach_open
        self._apply_fold()

    # ------------------------------------------------------------------
    def _fill_trend(self, profile: PlayerProfile | None) -> None:
        """Accuracy-over-runs sparkline from the profile history (hidden
        until at least two runs exist)."""
        hist = profile.history if profile is not None else []
        accs = [float(h.get("accuracy", 0.0)) for h in hist[-60:]]
        if len(accs) >= 2:
            self.trend_spark.set_title(analysis_zh(_trend_title(accs)))
            # first_run is the true 1-based run number of accs[0]. Without it
            # the widget can only say "oldest shown", because it cannot know
            # this list was sliced — on a 137-run profile it was labelling
            # run 78 as "run 1".
            self.trend_spark.set_data(accs, tag=f"{accs[-1]:.0%}",
                                      first_run=len(hist) - len(accs) + 1)
            self.trend_w.show()
        else:
            self.trend_spark.set_title(analysis_zh(_TREND_TITLE))
            self.trend_spark.clear()
            self.trend_w.hide()

    def _draw_bias(self, rep: RunReport) -> None:
        b = rep.bias or {}
        direction_keys = ["left", "vertical", "right"]
        direction_labels = ["左侧", "垂直", "右侧"]
        vals = [
            (b.get(d) or {}).get("overshoot", 0.0)
            + 0.15 * (b.get(d) or {}).get("corrections", 0.0)
            for d in direction_keys
        ]
        ns = [(b.get(d) or {}).get("n", 0) for d in direction_keys]
        degraded = input_degraded(rep)
        moving = (rep.player_frame or "") == "MOBILE"
        self.bias_bars.set_title(analysis_zh(_bias_title(vals, ns, degraded, moving)))
        # ratio_counts is the caller's explicit permission for the chart to
        # spell out a side-vs-side ratio, and only this layer can grant it:
        # viz.py cannot reach input_degraded (it would have to import analysis
        # and take a RunReport). Withheld on a degraded run, so the footer can
        # never contradict a title that says the comparison cannot be made;
        # granted otherwise, since a 1.09x gap genuinely cannot be eyeballed
        # and the sentence is the only way to read it.
        if sum(ns) == 0:
            # directional_bias([]) returns a POPULATED all-zero dict, so vals
            # is [0.0, 0.0, 0.0] — truthy, which walked straight past
            # AsciiBars' own "waiting for flick data" empty state and drew
            # three empty tracks with 0.00 beside each. The heatmap next to it
            # says "no movement data" on this same run, and the KPI tile shows
            # an em-dash; only this panel pretended to have measured something.
            self.bias_bars.set_data([], [])
        else:
            compare, worst = _bias_claim(vals, ns, degraded)
            self.bias_bars.set_data(direction_labels, vals,
                                    [f"{n} 次甩枪" for n in ns],
                                    ratio_counts=None if degraded else ns,
                                    floor=_BIAS_COST_FLOOR,
                                    compare=compare, worst=worst)
        self.bias_caption.setText(analysis_zh(_bias_caption(vals, ns, degraded, moving)))

    def _draw_heat(self, rep: RunReport | None = None) -> None:
        """Zone heatmap on the Settings.region_cols x region_rows grid:
        the run's region deficits when the report has them, else the
        movement heatmap pooled down to the same zones."""
        cols = self._settings.region_cols if self._settings is not None else 5
        rows = self._settings.region_rows if self._settings is not None else 5
        if rep is not None and rep.region_deficits:
            grid, labels = viz.region_grid(rep.region_deficits, cols, rows)
            self.heat_map.set_title(analysis_zh(
                _deficit_title(rep.region_deficits, self._settings,
                               input_degraded(rep))))
            self.heat_map.set_data(grid, labels, fmt="{:+.2f}")
            self.heat_caption.setText(analysis_zh(_DEFICIT_CAPTION))
        elif self.trace is not None and len(self.trace) >= 2:
            heat, _xe, _ye = movement_heatmap(self.trace)
            pooled = viz.pool(np.log1p(heat.T), rows, cols)  # heat.T row 0 = bottom
            labels = [[f"r{r}c{c}" for c in range(cols)] for r in range(rows)]
            self.heat_map.set_title(analysis_zh(_travel_title(heat)))
            self.heat_map.set_data(pooled, labels, fmt="{:.2f}")
            self.heat_caption.setText(analysis_zh(_TRAVEL_CAPTION))
        else:
            self.heat_map.set_title(analysis_zh(_TRAVEL_TITLE))
            self.heat_map.set_data(None)
            self.heat_caption.setText(analysis_zh(_TRAVEL_CAPTION))

    def _place_moments_placeholder(self) -> None:
        """The unselectable row that says what this list is for.

        `_fill_moments` clears the list, which destroys the one put there at
        construction — so the cold-start fix worked only until the first run,
        and any run that produced no notable moments went back to a blank
        panel. The placeholder belongs wherever the list ends up empty, not
        only in __init__.
        """
        item = QListWidgetItem(analysis_zh(_MOMENTS_EMPTY))
        item.setFlags(Qt.NoItemFlags)
        self.moments.addItem(item)

    def _fill_moments(self, rep: RunReport) -> None:
        self.moments.blockSignals(True)
        self.moments.clear()
        # Each moment narrates per-flick overshoot and correction counts, so
        # the same input-health gate applies. The clips stay listed and
        # replayable — the events are real — but the page must not quantify
        # them as if the timing behind them were trustworthy.
        if rep.notable and input_degraded(rep):
            note = QListWidgetItem(
                "· 本局输入时序噪声较大，以下过冲数值仅供参考 ·")
            note.setForeground(QColor(theme.current().fg_dim))
            note.setFlags(Qt.NoItemFlags)          # a caption, not a choice
            self.moments.addItem(note)
        for i, m in enumerate(rep.notable):
            it = QListWidgetItem(moment_text_zh(m))
            it.setForeground(QColor(_kind_color(m["kind"])))
            it.setData(Qt.UserRole, i)
            self.moments.addItem(it)
        if not self.moments.count():
            self._place_moments_placeholder()
        self.moments.blockSignals(False)
        if rep.notable:
            # First SELECTABLE row — row 0 may be the input-health caption,
            # which carries NoItemFlags and no moment index.
            first = next((r for r in range(self.moments.count())
                          if self._moment_index(r) >= 0), -1)
            self.moments.setCurrentRow(first)  # drives the replay via _select_moment

    def _moment_index(self, row: int) -> int:
        """Moment index behind a list row, or -1.

        Rows are NOT moment indices: the list can carry a non-selectable
        caption row, so every row stamps its own index in Qt.UserRole and
        that is the only mapping to trust.
        """
        item = self.moments.item(row) if row >= 0 else None
        if item is None:
            return -1
        idx = item.data(Qt.UserRole)
        return int(idx) if idx is not None else -1

    def _select_moment(self, row: int) -> None:
        idx = self._moment_index(row)
        if self.report is None or idx < 0 or idx >= len(self.report.notable):
            self._update_clip_state(-1)
            return
        m = self.report.notable[idx]
        self._update_clip_state(idx)
        if self.trace is not None and len(self.trace) > 1:
            self.replay.load(self.trace, m["t_start"], m["t_end"],
                             label={
                                 "overshoot": "过冲", "hesitation": "犹豫与连续修正",
                                 "unconfirmed_miss": "无修正直接点空",
                                 "slow_flick": "较慢甩枪", "clean_flick": "干净甩枪",
                             }.get(m["kind"], "关键片段"),
                             flicks=self.flicks)

    def _show_full_run(self) -> None:
        """Back out to the whole run. The selection is cleared first so the
        list highlight and the replay never describe different segments."""
        if self.trace is None or len(self.trace) <= 1:
            return
        self.moments.setCurrentRow(-1)        # _select_moment(-1) clears clip state
        self.replay.load(self.trace, label="整局", flicks=self.flicks)

    # ------------------------------------------------------------------
    def _clips_off_reason(self) -> str | None:
        """Why the clips feature can't produce clips at all right now, or None
        when it can (then a missing clip is just a moment without one)."""
        if self._settings is not None and not self._settings.clips_enabled:
            return "请在“自适应设置”中启用关键片段录像；之后产生的新片段才会带有录像"
        if not _clips_available():
            return "尚未安装录像依赖；请运行 pip install kovadapt[clips] 安装 dxcam/opencv"
        return None

    def _update_clip_state(self, moment_idx: int) -> None:
        """Enable the clip button when the selected moment has a clip; when it
        doesn't, the disabled button's tooltip says exactly why.

        The parameter is a MOMENT index, never a list row. It was called
        `row`, and that name was the whole bug: `clip_files` is keyed by
        moment (watcher.py enumerates rep.notable), one caller correctly
        passed an index into a parameter called row, the other passed an
        actual row, and on clean runs the two are equal — so nothing ever
        caught it.
        """
        has_clip = (self.report is not None and moment_idx >= 0
                    and str(moment_idx) in (self.report.clip_files or {}))
        self.clip_btn.setEnabled(has_clip)
        off = self._clips_off_reason()
        self.clip_btn.setToolTip(
            "" if has_clip else off or "这个关键片段没有录制录像")
        self.clip_hint.setText(off or "")
        self.clip_hint.setVisible(off is not None)

    def _play_clip(self) -> None:
        idx = self._moment_index(self.moments.currentRow())
        if self.report is None or idx < 0:
            return
        p = (self.report.clip_files or {}).get(str(idx))
        if p and Path(p).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p))))

    def _open_dialog(self) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "打开训练报告",
                                           str(Path.home() / ".kovadapt" / "reports"),
                                           "训练报告 (*.json)")
        if p:
            self.load_report_file(p)
