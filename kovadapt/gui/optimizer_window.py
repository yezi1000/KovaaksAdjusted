"""Optimizer window: the free Process Lasso alternative for KovaaK's.

A separate top-level window (dark, simple) with four sections:

    Hardware      detected CPU / GPU / RAM / refresh / Windows version
    Checkup       scan -> per-item status + Fix buttons + "Fix all safe"
    Watchdog      auto-tune every game launch; optional start with Windows
    Advice        hardware-matched launch options and settings

Scans run on a QThread (the GPU probe shells out to PowerShell, ~1s) and so
do fixes — several shell out too, and the power-plan one chains four
powercfg calls with a 15 s timeout each. Watchdog events arrive on its
polling thread and are bridged into Qt via a signal.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QObject, QRect, Qt, QThread, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..optimize.checkup import CheckResult, SystemCheckup
from ..optimize.hardware import HardwareInfo, detect_hardware
from ..optimize.recommend import (
    recommended_settings,
    skipped_launch_options,
    steam_launch_options,
)
from ..optimize.watchdog import (
    GameWatchdog,
    register_startup,
    startup_registered,
    unregister_startup,
)
from . import theme
from .i18n import tr

_STATUS_DOT = {"ok": "●", "warn": "●", "bad": "●", "info": "○", "unknown": "○"}

# How tall the watchdog log is allowed to get once it HAS something to show.
# It stays hidden until then — see the note where it is built. Four lines:
# at 110 it took so much off the top that opening it dropped the checkup
# from 197px to 81px on the smallest window the app will open. The log is a
# scrolling record and scrolls; the checkup is a list you act on.
_LOG_H = 72

_CHECK_TITLES = {
    "power_plan": "高性能电源计划",
    "core_parking": "当前电源计划已关闭 CPU 核心停放",
    "hags": "硬件加速 GPU 调度（HAGS）",
    "hags_live": "GPU 调度：注册表设置与实时驱动状态",
    "fso": "已为游戏关闭全屏优化",
    "mouse_accel": "Windows 鼠标加速度已关闭",
    "gamedvr": "Game Bar 后台录制已关闭",
    "game_mode": "Windows 游戏模式",
    "timer_res": "系统计时器精度",
    "chromium": "Chromium 后台程序",
    "game_config": "KovaaK's 配置文件健康状态",
    "game_process": "游戏优先级与 CPU 亲和性",
}

_FIX_LABELS = {
    "power_plan": "启用卓越性能电源计划",
    "fso": "为游戏关闭全屏优化",
    "mouse_accel": "关闭鼠标指针加速度",
    "gamedvr": "关闭 Game Bar 后台录制",
    "game_mode": "开启游戏模式",
    "game_config": "备份并删除（由游戏重建）",
    "game_process": "应用高优先级并释放输入核心",
}


def _first(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, re.I)
    return match.group(1) if match else default


def _check_detail_zh(result: CheckResult) -> str:
    """Chinese display copy of a check result; probe contracts stay English."""
    cid, status, src = result.check_id, result.status, result.detail
    if "not Windows" in src:
        return "当前系统不是 Windows，无法执行这项检测。"
    if "probe failed" in src:
        return "检测执行失败：" + src.split(":", 1)[-1].strip()
    if cid == "power_plan":
        if status == "ok":
            return ("卓越性能电源计划已启用。" if "Ultimate" in src
                    else "高性能电源计划已启用；当前核心不会因节能策略频繁降频。")
        return "当前使用平衡或节能电源计划，训练中可能出现核心停放、降频和帧时间尖峰。"
    if cid == "core_parking":
        pct = _first(r"(\d+)%", src, "未知")
        if status == "ok":
            return f"当前计划保证至少 {pct}% 的核心保持唤醒，训练中不会发生核心停放。"
        if status == "warn":
            return (f"当前计划只保证 {pct}% 的核心保持唤醒；核心在甩枪中途恢复可能造成帧时间尖峰。"
                    "此项可能需要管理员权限，因此不自动修改。")
        return "无法从当前电源计划读取 CPMINCORES，powercfg 输出格式无法识别。"
    if cid == "hags":
        if "driver default" in src:
            return "HAGS 使用显卡驱动默认设置；如遇卡顿，可在 Windows 图形默认设置中进行对比测试。"
        state = "开启" if "is on" in src else "关闭"
        if status == "ok":
            return f"HAGS 当前已{state}，与这张显卡的建议状态一致。"
        if "no strong guidance" in src:
            return f"HAGS 当前已{state}；对这张显卡没有足够明确的统一建议。"
        return (f"HAGS 当前已{state}，但与这张显卡的建议状态不一致。请在 Windows 图形"
                "默认设置中手动调整；需要管理员权限并重启，因此 kovadapt 不会自动切换。")
    if cid == "hags_live":
        if "caps query failed" in src:
            return "WDDM 2.7 能力查询失败，无法读取驱动当前实际使用的 HAGS 状态。"
        if "no hardware-scheduling support" in src:
            return "驱动报告不支持硬件调度，因此 HAGS 开关不会对当前显卡或驱动产生作用。"
        state = "开启" if "live on" in src or "is on" in src else "关闭"
        if status == "ok":
            return f"注册表设置与实时驱动状态一致：HAGS 当前{state}。"
        if "driver default" in src:
            return f"HAGS 当前实际{state}，由驱动默认值决定，没有注册表覆盖。"
        if status == "warn":
            return "注册表设置与驱动实时状态不一致；可能尚未重启，或驱动覆盖了该设置。"
        return f"驱动当前实际使用的 HAGS 状态为{state}。"
    if cid == "fso":
        if "game exe not found" in src:
            return "找不到游戏可执行文件，请先设置 KovaaK's 路径。"
        return ("已经为游戏可执行文件关闭全屏优化。" if status == "ok" else
                "Windows 可能在全屏时仍通过 DWM 合成器呈现游戏，从而增加显示延迟。")
    if cid == "mouse_accel":
        return ("“提高指针精确度”已关闭。KovaaK's 使用 Raw Input；此项主要保证桌面菜单和其他游戏的一致性。"
                if status == "ok" else
                "“提高指针精确度”（鼠标加速度）已开启。Raw Input 游戏会绕过它，但桌面和非 Raw Input 游戏的手感会不一致。")
    if cid == "gamedvr":
        if status == "ok":
            return "Game DVR 后台录制已关闭，不会有录制进程在后台持续占用游戏资源。"
        app = _first(r"AppCaptureEnabled=([^,]+)", src, "未知")
        store = _first(r"GameDVR_Enabled=([^ ]+)", src, "未知")
        return ("Game DVR 会在后台持续录制，可能影响帧率和帧时间稳定性。"
                f"当前值：AppCaptureEnabled={app}，GameDVR_Enabled={store}。")
    if cid == "game_mode":
        return ("游戏模式已开启；它会在游戏运行时减少 Windows 更新和驱动通知等后台干扰。"
                if status in ("ok", "info") else
                "游戏模式已关闭。Windows 11 当前建议开启，以便在游戏运行时延后后台任务。")
    if cid == "timer_res":
        cur = _first(r"currently ([\d.]+) ms", src, "未知")
        finest = _first(r"supports: ([\d.]+) ms", src, "未知")
        if "Windows 11" in src:
            return (f"当前计时器精度 {cur} 毫秒（系统最细 {finest} 毫秒）。Windows 11 按进程"
                    "管理计时器，游戏请求 1 毫秒时会单独获得，无需处理。")
        if status == "warn":
            return (f"游戏运行时计时器精度为 {cur} 毫秒；Windows 10 使用全局计时器，"
                    "高于约 1 毫秒可能影响帧节奏。此项由请求计时器的程序控制，无法自动修复。")
        return (f"当前计时器精度 {cur} 毫秒（系统最细 {finest} 毫秒）。Windows 10 使用"
                "全局计时器，建议在游戏运行时重新检查。")
    if cid == "chromium":
        if status == "ok":
            return "未发现会影响计时器的 Chromium 后台程序。"
        apps = _first(r"Running: ([^.]+)", src, "未知")
        return (f"正在运行：{apps}。Chromium 程序可能请求较粗的计时器并造成帧时间尖峰；"
                "训练时建议关闭它们或至少关闭其游戏覆盖层。为避免丢失工作，kovadapt 不会自动结束进程。")
    if cid == "game_config":
        if "not found" in src:
            return "未找到 GameUserSettings.ini；可能是首次安装或使用了自定义位置。"
        if status == "ok":
            return "GameUserSettings.ini 结构正常。"
        if "unreadable/corrupt" in src:
            return "GameUserSettings.ini 无法读取或已经损坏；这是已知的卡顿诱因。"
        return "GameUserSettings.ini 缺少预期配置段或包含异常字节，可能已经损坏。"
    if cid == "game_process":
        if "not running" in src.lower():
            return "游戏当前未运行；后台监测器会在每次启动游戏时自动应用这些设置。"
        if "access denied" in src.lower():
            return "读取游戏进程时被拒绝访问；请尝试以管理员身份运行。"
        cpu = _first(r"CPU ([0-9/]+)", src, "输入处理核心")
        return (f"游戏已设置为高优先级，并释放 CPU {cpu} 供输入处理。" if status == "ok" else
                f"游戏尚未完整应用 KovaaK's FAQ 建议的高优先级和 CPU {cpu} 释放设置。")
    return "检测结果：" + src


def _optimizer_message_zh(message: str) -> str:
    """Localize fix/watchdog outcome wrappers while preserving raw identifiers."""
    msg = str(message)
    exact = {
        "not Windows": "当前系统不是 Windows",
        "game not running": "游戏未运行",
        "access denied — run kovadapt as administrator": "访问被拒绝；请以管理员身份运行 kovadapt",
        "game exited while applying": "应用设置时游戏已经退出",
        "startup entry removed": "已移除开机启动项",
        "was not registered": "此前没有注册开机启动项",
        "nothing to do": "无需处理",
        "game exe not found": "找不到游戏可执行文件",
        "could not activate — change it in Windows Power Options":
            "无法启用该电源计划；请在 Windows 电源选项中手动修改",
        "could not change the setting": "无法修改此设置",
        "no automated fix for this item": "此项目没有自动修复方式",
    }
    if msg in exact:
        return exact[msg]
    if msg.startswith("fix failed:"):
        return "修复失败：" + msg.split(":", 1)[1].strip()
    if msg.startswith("registered:"):
        return "已注册开机启动：" + msg.split(":", 1)[1].strip()
    if "Performance plan activated" in msg:
        return msg.split()[0] + " 性能电源计划已启用"
    if msg.startswith("fullscreen optimizations disabled"):
        count = _first(r"on (\d+) exe", msg, "0")
        return f"已为 {count} 个游戏程序关闭全屏优化；设置仅对当前用户生效，可在兼容性属性中恢复"
    if msg.startswith("pointer acceleration off"):
        return "鼠标指针加速度已关闭并保存"
    if msg.startswith("background capture off"):
        return "Game Bar 后台录制已关闭；设置仅对当前用户生效，可在 Windows 游戏设置中恢复"
    if msg.startswith("Game Mode on"):
        return "Windows 游戏模式已开启；设置仅对当前用户生效"
    if msg.startswith("deleted (backup at"):
        backup = _first(r"backup at ([^)]+)", msg, "备份文件")
        return f"已删除原配置，备份位于 {backup}；启动 KovaaK's 后游戏会自动重建"
    if msg.startswith("pid "):
        pid = _first(r"pid (\d+)", msg, "未知")
        cpu = _first(r"CPU ([0-9/]+)", msg, "")
        return f"进程 {pid}：已设为高优先级" + (f"，并释放 CPU {cpu} 供输入处理" if cpu else "")
    if msg.startswith("watchdog on"):
        return "后台监测器已开启；每次启动游戏时都会自动优化"
    if msg == "watchdog off":
        return "后台监测器已关闭"
    if msg.startswith("watchdog stopped"):
        return "后台监测器已停止：缺少 psutil，请安装 kovadapt[gui]"
    if msg.startswith("game detected —"):
        core = msg.split("—", 1)[1].split("(tuned at epoch", 1)[0].strip()
        return "检测到游戏 — " + _optimizer_message_zh(core)
    if msg.startswith("game detected but not tuned:"):
        return "检测到游戏，但未能优化：" + _optimizer_message_zh(msg.split(":", 1)[1].strip())
    if msg.startswith("launching KovaaK's requires Windows"):
        return "启动 KovaaK's 需要 Windows"
    return msg


def _recommendation_zh(rec) -> tuple[str, str]:
    """Chinese advice copy keyed by the stable recommendation title."""
    title = rec.title
    if title.startswith("Cap FPS near"):
        cap = _first(r"near (\d+)", title, "合适值")
        hz = _first(r"your (\d+) Hz", title, "当前")
        return (f"将帧率上限设在约 {cap} FPS（显示器 {hz} Hz 的两倍）",
                "稳定的帧率上限比更高但波动的无限制帧率更利于瞄准。请在 KovaaK's 视频设置中限制，而不是在驱动中限制。")
    table = (
        ("Exclusive fullscreen", "使用独占全屏、原生分辨率，其余画质设为低",
         "KovaaK's 的低画质不会影响目标识别，却能保持 GPU 帧时间稳定；分辨率缩放保持 100%。"),
        ("Max the workshop cache", "把创意工坊缓存有效期调到最大（168 小时）",
         "无法关闭启动时的创意工坊数据检查，但把缓存刷新延长到一周后，大多数启动会明显更快；本地自适应场景不受影响。"),
        ("Enable NVIDIA Reflex", "在 KovaaK's 中开启 NVIDIA Reflex",
         "Reflex 会尽量清空渲染队列，是当前显卡上最有效的输入延迟优化之一。"),
        ("NVIDIA Control Panel > Low Latency", "在 NVIDIA 控制面板中把低延迟模式设为“超高”",
         "当前显卡早于游戏内 Reflex 支持；驱动的超低延迟模式可作为限制渲染队列的替代方案。"),
        ("AMD Adrenalin", "在 AMD Adrenalin 中开启 Radeon Anti-Lag",
         "这是 AMD 的渲染队列限制功能，目的与 NVIDIA Reflex 相同。"),
        ("NVIDIA Control Panel > Power management", "在 NVIDIA 控制面板中选择“最高性能优先”",
         "可避免显卡在 KovaaK's 的轻负载帧之间降频；建议只为 FPSAimTrainer.exe 单独设置，避免增加待机功耗。"),
        ("VSync off", "在游戏和驱动中都关闭垂直同步",
         "垂直同步最多会增加一帧延迟。若撕裂明显，可使用 G-Sync/FreeSync，并把上限设在刷新率以下约 3 FPS。"),
        ("HAGS ON", "开启 HAGS，把帧调度从 CPU 转移到 GPU",
         "硬件加速 GPU 调度会把调度队列交给显卡专用引擎，在 KovaaK's 的 CPU 瓶颈位置释放处理时间；修改后需要重启。"),
        ("Frame generation", "训练时关闭 NVIDIA Smooth Motion 帧生成",
         "生成帧不会采样鼠标，只增加视觉流畅度而不增加瞄准信息。KovaaK's 通常能达到很高的真实帧率，应优先使用真实帧。"),
        ("Let the GPU absorb", "有性能余量时优先提高分辨率，不要继续提高帧率上限",
         "达到帧率上限后，多余 CPU 余量不会改善手感；提高分辨率能把相对负载转移到 GPU，并比追逐无法达到的帧率更稳定。"),
        ("Keep hardware-accelerated", "保持硬件加速 GPU 调度开启",
         "较新的显卡通常能获得略稳定的延迟表现。"),
        ("Try hardware-accelerated", "尝试关闭硬件加速 GPU 调度并进行对比",
         "较旧显卡开启 HAGS 有时会增加卡顿；建议分别训练一段时间后对比结果。"),
        ("Keep a 1 ms timer", "在 Windows 10 中保持 1 毫秒计时器工具运行",
         "Windows 10 使用全局计时器精度，后台程序可能使其变粗；Windows 11 则按进程管理。"),
        ("Let the kovadapt watchdog", "让 kovadapt 后台监测器把游戏移出输入处理核心",
         "逻辑核心充足时，可以保留第一个物理核心供 Windows 处理输入；后台监测器会在每次启动游戏时自动应用。"),
    )
    for prefix, zh_title, zh_detail in table:
        if title.startswith(prefix):
            return zh_title, zh_detail
    return title, rec.detail


def _skipped_reason_zh(flag: str, reason: str) -> str:
    reasons = {
        "-high": "只在启动时设置一次优先级；kovadapt 后台监测器会在每次启动时可靠地重新应用",
        "-malloc=system": "这是 2015 年的 UE4 内存分配建议；在 KovaaK's 中没有可测收益，还可能增加内存碎片",
        "-notexturestreaming": "KovaaK's 纹理很小；只会无意义地增加显存占用",
        "-dx12": "KovaaK's 使用 DX11；强制其他渲染接口可能导致故障或性能下降",
        "-ONETHREAD": "这是调试参数，会显著降低性能",
    }
    return reasons.get(flag, reason)


def _status_color(status: str) -> str:
    """Status colour, from the semantic ramp only — never from the accent.

    `info` used to be pal.accent, and the accent is whatever the user picked
    in Settings. On live probes info is the LARGEST bucket of the twelve
    checkup rows, so nearly half the list wore it: under the rose accent every
    informational row read as an error, under mint as an all-clear. The dot
    SHAPE survives that (○ against ●), but colour is the channel a reader
    takes a verdict from, and it was reporting a preference.

    Neutral ink instead — a fact, not a judgement — and `unknown` stays dim,
    so "here is a reading" and "could not read this" stop looking alike.
    """
    pal = theme.current()
    return {"ok": pal.good, "warn": pal.warn, "bad": pal.bad,
            "info": pal.fg}.get(status, pal.fg_dim)


class _ScanWorker(QThread):
    """Hardware detection + all checkup probes off the UI thread."""

    done = Signal(object, object)   # (HardwareInfo, list[CheckResult])

    def __init__(self, kovaaks_root: str, parent=None) -> None:
        super().__init__(parent)
        self.kovaaks_root = kovaaks_root

    def run(self) -> None:
        hw = detect_hardware()
        results = SystemCheckup(self.kovaaks_root, hw).run_all()
        self.done.emit(hw, results)


class _FixWorker(QThread):
    """Checkup fixes off the UI thread, one after another.

    A fix is not "individually fast": most shell out or hit the registry, and
    _f_power alone chains four powercfg calls whose subprocess timeout is 15 s
    apiece — run inline on click that froze the window for up to a minute.
    SystemCheckup.fix() converts its own exceptions into a message, so this
    loop cannot die half way through a batch.
    """

    one_done = Signal(str, str)     # (check_id, outcome message)

    def __init__(self, checkup: SystemCheckup, check_ids: list[str],
                 parent=None) -> None:
        super().__init__(parent)
        self.checkup = checkup
        self.check_ids = list(check_ids)

    def run(self) -> None:
        for cid in self.check_ids:
            self.one_done.emit(cid, self.checkup.fix(cid))


# Worker threads that were still running at shutdown(). Qt 6 aborts the
# process when a running QThread is destroyed, and a fix can be mid registry
# write, so parking one here (out of the window's child tree, with a live
# Python reference) is safer than terminate()ing it: the interpreter is on
# its way out anyway.
_PARKED: list[QThread] = []


class _WatchdogBridge(QObject):
    event = Signal(str)


class _CheckRow(QFrame):
    """One checkup line: colored status dot, title, detail, optional Fix."""

    def __init__(self, result: CheckResult, on_fix, parent=None) -> None:
        super().__init__(parent)
        # A QFrame, and theme.py's page-background rule reaches every QWidget
        # subclass — so each of these twelve rows painted the PAGE colour on
        # top of the checkup box's own plate. It cannot be exempted globally
        # the way QCheckBox is: `QFrame` as a type selector also matches
        # QLabel, QScrollArea, QSplitter and QStackedWidget, and `.QFrame`
        # matches only a bare QFrame, which this is not.
        #
        # SCOPED to this class, which PySide publishes to QSS under its Python
        # name. A bare "background: transparent" here cascades to the children
        # and took the Fix button's accent fill down to #010102 — measured, and
        # caught by looking at the render rather than by the suite.
        self.setStyleSheet("_CheckRow { background: transparent; }")
        self.result = result
        self._dot = dot = QLabel(_STATUS_DOT.get(result.status, "○"))
        dot.setFixedWidth(18)
        self.restyle()
        title = QLabel(f"<b>{_CHECK_TITLES.get(result.check_id, result.title)}</b>")
        title.setTextFormat(Qt.RichText)
        self.detail = QLabel(_check_detail_zh(result))
        self.detail.setWordWrap(True)
        self.detail.setProperty("dim", True)

        grid = QGridLayout(self)
        grid.setContentsMargins(4, 6, 4, 6)
        grid.addWidget(dot, 0, 0, Qt.AlignTop)
        grid.addWidget(title, 0, 1)
        grid.addWidget(self.detail, 1, 1)
        if result.can_fix:
            self.fix_btn = QPushButton(
                _FIX_LABELS.get(result.check_id, result.fix_label or "修复"))
            if result.safe:
                self.fix_btn.setProperty("accent", True)
            self.fix_btn.clicked.connect(lambda: on_fix(self))
            grid.addWidget(self.fix_btn, 0, 2, 2, 1, Qt.AlignVCenter)
        grid.setColumnStretch(1, 1)

    def restyle(self) -> None:
        self._dot.setStyleSheet(
            f"color: {_status_color(self.result.status)}; font-size: 15px;")

    def mark_pending(self) -> None:
        """Queued for the fix worker: the button can't be clicked twice, and
        the row says why nothing has happened yet."""
        if hasattr(self, "fix_btn"):
            self.fix_btn.setEnabled(False)
            self.fix_btn.setText("正在应用…")

    # `SystemCheckup.apply_fix` turns any exception into "fix failed: …" and
    # hands it back through the SAME channel as a success, so the outcome
    # string is the only thing that knows which happened.
    FAILED_PREFIX = "fix failed"

    def show_outcome(self, msg: str) -> None:
        """Report what the fix actually did — including that it did not work.

        This used to set the detail text, disable the button and label it
        "Applied" unconditionally, and never touch `result.status` or the
        status dot: a pixel-exact diff of the dot before and after a real fix
        showed 0 of 360 pixels changing. So a FAILED fix rendered as a green
        "Applied" over an unchanged amber dot, and success and failure
        differed by one sentence of body text. On an app whose whole rule is
        never to claim something it has not measured, that is the wrong thing
        to put in a release.
        """
        ok = not msg.lower().startswith(self.FAILED_PREFIX)
        self.detail.setText(_optimizer_message_zh(msg))
        self.result.status = "ok" if ok else "bad"
        self._dot.setText(_STATUS_DOT.get(self.result.status, "○"))
        self.restyle()
        if hasattr(self, "fix_btn"):
            # A failure leaves the button live: the cause is usually
            # transient (a permission prompt declined, the game running) and
            # re-running is the obvious next move.
            self.fix_btn.setEnabled(not ok)
            self.fix_btn.setText("已应用" if ok else "重试")


class OptimizerWindow(QWidget):
    """Top-level optimizer window (create with parent=None)."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(None)
        self.setWindowTitle("kovadapt — 性能优化器")
        # 720 tall regardless of the panel, on a window whose main content is
        # a twelve-row list ~860px long. Take the height the screen actually
        # has, capped so this never opens taller than the desktop it is on.
        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry() if screen is not None \
            else QRect(0, 0, 1280, 800)
        self.resize(860, max(640, min(940, int(geo.height() * 0.88))))
        self.s = settings
        self.hw: HardwareInfo | None = None
        self.checkup: SystemCheckup | None = None
        self._scan: _ScanWorker | None = None
        self._fix: _FixWorker | None = None
        self._suspended: list[_CheckRow] = []   # buttons parked during a batch

        # --- hardware summary -------------------------------------------
        self.hw_label = QLabel("正在扫描硬件…")
        self.hw_label.setProperty("headline", True)
        self.hw_sub = QLabel("")
        self.hw_sub.setProperty("dim", True)
        hw_box = QGroupBox(tr("Detected hardware"))
        v = QVBoxLayout(hw_box)
        v.addWidget(self.hw_label)
        v.addWidget(self.hw_sub)

        # --- checkup ------------------------------------------------------
        self.scan_btn = QPushButton(tr("Re-scan"))
        self.scan_btn.clicked.connect(self.rescan)
        self.fix_safe_btn = QPushButton(tr("Fix all safe items"))
        self.fix_safe_btn.setProperty("accent", True)
        self.fix_safe_btn.setToolTip(
            "执行所有仅影响当前用户、可撤销且不需要管理员权限的修复。电源计划、"
            "删除损坏配置等影响较大的操作仍需单独点击对应按钮。")
        self.fix_safe_btn.clicked.connect(self.fix_all_safe)
        self.fix_safe_btn.setEnabled(False)
        head = QHBoxLayout()
        head.addWidget(self.scan_btn)
        head.addWidget(self.fix_safe_btn)
        head.addStretch(1)

        self.rows_holder = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_holder)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.addStretch(1)
        rows_scroll = QScrollArea()
        rows_scroll.setWidget(self.rows_holder)
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QScrollArea.NoFrame)
        check_box = QGroupBox(tr("System checkup"))
        check_box.setObjectName("systemCheckup")
        cv = QVBoxLayout(check_box)
        cv.addLayout(head)
        cv.addWidget(rows_scroll, 1)

        # --- watchdog -------------------------------------------------------
        self._bridge = _WatchdogBridge()
        self._bridge.event.connect(self._log)
        self.watchdog = GameWatchdog(on_event=self._bridge.event.emit)
        self.wd_toggle = QCheckBox(
            "每次启动游戏时自动优化（高优先级 + 释放输入处理核心）")
        self.wd_toggle.setToolTip(
            "免费实现与 Process Lasso 持久规则相同的核心功能。应用打开期间会持续"
            "监测；启用开机启动后可覆盖每次训练。")
        self.wd_toggle.toggled.connect(self._toggle_watchdog)
        self.wd_startup = QCheckBox("随 Windows 启动后台监测器（无窗口）")
        self.wd_startup.setChecked(startup_registered())
        self.wd_startup.toggled.connect(self._toggle_startup)
        self.wd_log = QPlainTextEdit()
        self.wd_log.setReadOnly(True)
        self.wd_log.setMaximumBlockCount(200)
        self.wd_log.setMaximumHeight(_LOG_H)
        # Hidden until the watchdog actually says something. This box has no
        # stretch, so its sizeHint comes off the top of the window before the
        # stretch factors divide anything up — and 110px of that was an empty
        # read-only pane. It made the un-stretched watchdog box TALLER than
        # the stretch-3 System checkup beside it, which opened as a 79px slot
        # over a twelve-row list. The "Evidence:" label above already says
        # what will appear here, so an empty pane was not even carrying the
        # explanation.
        self.wd_log.hide()
        self.jitter_lbl = QLabel(
            "数据依据：开启鼠标遥测完成训练后，这里会显示每局输入抖动，"
            "并对比自动优化前后的结果。")
        self.jitter_lbl.setProperty("dim", True)
        self.jitter_lbl.setWordWrap(True)
        self._jitter_runs: list[tuple[float, float]] = []   # (epoch, jitter_ms)
        wd_box = QGroupBox("后台监测器（Process Lasso 的免费替代方案）")
        wd_box.setObjectName("watchdog")
        wv = QVBoxLayout(wd_box)
        wv.addWidget(self.wd_toggle)
        wv.addWidget(self.wd_startup)
        wv.addWidget(self.jitter_lbl)
        wv.addWidget(self.wd_log)

        # --- advice -----------------------------------------------------------
        self.launch_label = QLabel(steam_launch_options(HardwareInfo()))
        self.launch_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        copy_btn = QPushButton(tr("Copy"))
        copy_btn.setFixedWidth(70)
        copy_btn.clicked.connect(self._copy_launch)
        lrow = QHBoxLayout()
        lrow.addWidget(self.launch_label, 1)
        lrow.addWidget(copy_btn)
        skip_lines = "".join(
            f"<li><code>{flag}</code> — {_skipped_reason_zh(flag, why)}</li>"
            for flag, why in skipped_launch_options())
        skips = QLabel(f"<b>不建议使用的参数：</b><ul>{skip_lines}</ul>")
        skips.setTextFormat(Qt.RichText)
        skips.setWordWrap(True)
        skips.setProperty("dim", True)
        self.recs_label = QLabel("")
        self.recs_label.setTextFormat(Qt.RichText)
        self.recs_label.setWordWrap(True)
        adv_inner = QWidget()
        av = QVBoxLayout(adv_inner)
        av.addWidget(QLabel("Steam 启动选项（右键 KovaaK's → 属性）："))
        av.addLayout(lrow)
        av.addWidget(skips)
        av.addWidget(self.recs_label)
        av.addStretch(1)
        adv_scroll = QScrollArea()
        adv_scroll.setWidget(adv_inner)
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setFrameShape(QScrollArea.NoFrame)
        adv_box = QGroupBox(tr("Recommended for your hardware"))
        adv_box.setObjectName("hardwareRecommendations")
        bv = QVBoxLayout(adv_box)
        bv.addWidget(adv_scroll)

        lay = QVBoxLayout(self)
        lay.addWidget(hw_box)
        # The checkup is what this window is for: a twelve-row list, each row
        # a status and a Fix button. Advice is reference prose that reads fine
        # a few lines at a time. 3:2 gave the list a 79px slot; the two boxes
        # with no stretch at all took their full sizeHint off the top first.
        lay.addWidget(check_box, 5)
        lay.addWidget(wd_box)
        lay.addWidget(adv_box, 2)

        self.restyle()
        self.rescan()

    # ------------------------------------------------------------- theming
    def restyle(self, *_pal) -> None:
        pal = theme.current()
        self.launch_label.setStyleSheet(
            f"font-family: Consolas, monospace; color: {pal.accent};")
        for row in self._rows():
            row.restyle()

    # ------------------------------------------------- input-health evidence
    def note_input_health(self, ih: dict) -> None:
        """Per-run input-health from the watcher: turn watchdog tweaks into
        before/after jitter evidence instead of folklore."""
        import time

        jit = ih.get("jitter_ms")
        if not jit:
            return
        self._jitter_runs.append((time.time(), float(jit)))
        del self._jitter_runs[:-50]
        tunes = self.watchdog.tune_times
        if tunes:
            t = tunes[-1]
            pre = [j for ts, j in self._jitter_runs if ts < t]
            post = [j for ts, j in self._jitter_runs if ts >= t]
            if pre and post:
                self.jitter_lbl.setText(
                    f"数据依据：上次自动优化后 {len(post)} 局的输入抖动平均为 "
                    f"{sum(post) / len(post):.2f} 毫秒；优化前 {len(pre)} 局平均为 "
                    f"{sum(pre) / len(pre):.2f} 毫秒。")
                return
            if post:
                self.jitter_lbl.setText(
                    f"数据依据：自动优化后 {len(post)} 局的输入抖动平均为 "
                    f"{sum(post) / len(post):.2f} 毫秒；本次训练尚无优化前记录可供比较。")
                return
        js = [j for _, j in self._jitter_runs]
        self.jitter_lbl.setText(
            f"数据依据：本次训练 {len(js)} 局的输入抖动平均为 "
            f"{sum(js) / len(js):.2f} 毫秒；后台监测器尚未在本次训练中执行优化。")

    # ------------------------------------------------------------- scanning
    def rescan(self) -> None:
        if self._scan is not None and self._scan.isRunning():
            return
        if self._fix is not None and self._fix.isRunning():
            return   # rebuilding the rows now would orphan the pending fixes
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("正在扫描…")
        self._scan = _ScanWorker(self.s.kovaaks_root, parent=self)
        self._scan.done.connect(self._on_scan)
        self._scan.start()

    def _on_scan(self, hw: HardwareInfo, results: list[CheckResult]) -> None:
        self.hw = hw
        self.checkup = SystemCheckup(self.s.kovaaks_root, hw)
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText(tr("Re-scan"))

        if hw.cpu_name or hw.gpu_name:
            self.hw_label.setText(f"{hw.cpu_name or '未知 CPU'}  ·  "
                                  f"{hw.gpu_name or '未知 GPU'}")
            bits = []
            if hw.ram_gb:
                bits.append(f"{hw.ram_gb:.0f} GB 内存")
            if hw.logical_cores:
                bits.append(f"{hw.logical_cores} 个逻辑线程")
            if hw.monitor_hz:
                bits.append(f"{hw.monitor_hz} Hz 显示器")
            bits.append("Windows 11" if hw.is_windows_11 else "Windows 10")
            self.hw_sub.setText("  ·  ".join(bits))
        else:
            self.hw_label.setText("无法检测硬件")
            notes = ["当前系统不是 Windows，已跳过硬件检测"
                     if n == "not Windows — detection skipped" else n
                     for n in hw.notes]
            notes = ["GPU 查询失败，无法提供针对显卡厂商的建议"
                     if n == "GPU query failed — vendor-specific advice unavailable" else n
                     for n in notes]
            self.hw_sub.setText("；".join(notes))

        # rebuild check rows
        while self.rows_layout.count() > 1:
            item = self.rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for r in results:
            self.rows_layout.insertWidget(self.rows_layout.count() - 1,
                                          _CheckRow(r, self._fix_row))
        self._refresh_fix_all()

        # advice
        self.launch_label.setText(steam_launch_options(hw))
        recs = recommended_settings(hw)
        cat_names = {"launch": "启动参数", "video": "游戏画面",
                     "driver": "显卡驱动", "windows": "Windows"}
        html = ""
        for pr, badge in ((1, "建议执行"), (2, "值得尝试"), (3, "视情况选择")):
            group = [r for r in recs if r.priority == pr]
            if not group:
                continue
            html += f"<p><b>{badge}：</b></p><ul>"
            for r in group:
                title, detail = _recommendation_zh(r)
                html += (f"<li><b>{title}</b> <i>（{cat_names.get(r.category, '')}）</i>"
                         f"<br>{detail}</li>")
            html += "</ul>"
        self.recs_label.setText(html)

    # --------------------------------------------------------------- fixes
    def _rows(self) -> list[_CheckRow]:
        out = []
        for i in range(self.rows_layout.count() - 1):
            w = self.rows_layout.itemAt(i).widget()
            if isinstance(w, _CheckRow):
                out.append(w)
        return out

    def _pending_safe(self) -> list[_CheckRow]:
        """Safe rows whose fix has not been applied (or queued) yet."""
        return [w for w in self._rows()
                if w.result.can_fix and w.result.safe
                and hasattr(w, "fix_btn") and w.fix_btn.isEnabled()]

    def _refresh_fix_all(self) -> None:
        n = len(self._pending_safe())
        self.fix_safe_btn.setEnabled(n > 0)
        self.fix_safe_btn.setText(
            f"修复全部安全项目（{n}）" if n else tr("Fix all safe items"))

    def _fix_row(self, row: _CheckRow) -> None:
        self._start_fixes([row])

    def fix_all_safe(self) -> None:
        self._start_fixes(self._pending_safe())

    def _start_fixes(self, rows: list[_CheckRow]) -> None:
        """Hand a batch to the fix worker. Never runs on the UI thread: the
        power-plan fix can take the better part of a minute."""
        if self.checkup is None or not rows:
            return
        if self._fix is not None and self._fix.isRunning():
            return
        for row in rows:
            row.mark_pending()
        # Only one batch runs at a time, so every other Fix button goes quiet
        # for the duration — clicking one would otherwise be a silent no-op.
        self._suspended = [r for r in self._rows() if r not in rows
                           and hasattr(r, "fix_btn") and r.fix_btn.isEnabled()]
        for r in self._suspended:
            r.fix_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)      # a re-scan would replace the rows
        self.fix_safe_btn.setEnabled(False)
        self._fix = _FixWorker(self.checkup, [r.result.check_id for r in rows],
                               parent=self)
        self._fix.one_done.connect(self._on_fixed)
        self._fix.finished.connect(self._on_fixes_done)
        self._fix.start()

    def _on_fixed(self, check_id: str, msg: str) -> None:
        # The row can be gone (a scan rebuilt them) — then there is nothing to
        # report the outcome on, and dropping it is the right answer.
        for row in self._rows():
            if row.result.check_id == check_id:
                row.show_outcome(msg)
                return

    def _on_fixes_done(self) -> None:
        # Re-derive the live rows rather than trusting the list captured when
        # the batch started. _on_scan takes every row out of the layout and
        # deleteLater()s it, so a scan landing mid-batch leaves these as freed
        # C++ objects and touching one raises "Internal C++ object already
        # deleted" — the same reason _on_fixed above re-queries instead of
        # holding a reference.
        live = set(self._rows())
        for row in self._suspended:
            if row in live:
                row.fix_btn.setEnabled(True)
        self._suspended = []
        self.scan_btn.setEnabled(True)
        self._refresh_fix_all()

    # ------------------------------------------------------------ watchdog
    def _toggle_watchdog(self, on: bool) -> None:
        if on:
            self.watchdog.start()
        else:
            self.watchdog.stop()

    def _toggle_startup(self, on: bool) -> None:
        self._log(register_startup() if on else unregister_startup())

    def _log(self, msg: str) -> None:
        self.wd_log.appendPlainText(_optimizer_message_zh(msg))
        self.wd_log.show()      # first line is what earns it the space

    def _copy_launch(self) -> None:
        QApplication.clipboard().setText(self.launch_label.text())
        self._log("启动参数已复制到剪贴板")

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """App is exiting: wait for the worker threads before Qt teardown (a
        QThread destroyed while running is a fatal abort in Qt 6), then close
        for real so this window can't keep the app alive."""
        self._closing = True
        for th in (self._scan, self._fix):
            if th is None or not th.isRunning():
                continue
            if not th.wait(20000):
                # Still going (a wedged powercfg is the realistic case). Take
                # it out of the child tree and park it so Qt cannot destroy it
                # with this window; see _PARKED.
                th.setParent(None)
                _PARKED.append(th)
        self.close()

    def closeEvent(self, event) -> None:
        # User-closing the window hides it; the watchdog keeps running (that
        # is its point). Real teardown happens via shutdown() at app exit.
        if getattr(self, "_closing", False):
            super().closeEvent(event)
            return
        event.ignore()
        self.hide()
