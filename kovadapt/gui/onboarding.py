"""First-run startup guide + dismissible contextual hints.

WelcomeDialog: a short paged guide shown on first launch (and on demand from
the Help menu). HintBar: a one-line contextual tip used across tabs; the ×
on any bar tucks ALL hints away (settings.show_hints=False, persisted), and
the Help menu brings them back — instructions are there for new users and
gone in one click for everyone else.
"""

from __future__ import annotations

import weakref

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from .i18n import tr

# Every live HintBar, so one dismiss / re-enable reaches all tabs.
_hint_bars: "weakref.WeakSet[HintBar]" = weakref.WeakSet()


class HintBar(QFrame):
    """One-line dismissible tip. Create it under any toolbar/header row."""

    def __init__(self, settings: Settings, text: str, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.setProperty("hint", True)
        tag = QLabel(tr("TIP"))
        tag.setStyleSheet("font-weight: 700; font-size: 11px;")
        tag.setProperty("dim", True)
        body = QLabel(text)
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        body.setProperty("dim", True)
        close = QPushButton("×")
        close.setProperty("flat", True)
        close.setFixedWidth(24)
        close.setToolTip("隐藏所有提示；可以从帮助菜单重新显示")
        close.clicked.connect(self._dismiss_all)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 6, 6)
        lay.addWidget(tag, 0, Qt.AlignTop)
        lay.addWidget(body, 1)
        lay.addWidget(close, 0, Qt.AlignTop)
        # A tip is one or two lines: never let it absorb a layout's spare
        # vertical space just because it is the only greedy widget present.
        self.setSizePolicy(self.sizePolicy().horizontalPolicy(),
                           QSizePolicy.Maximum)

        _hint_bars.add(self)
        self.setVisible(settings.show_hints)

    def _dismiss_all(self) -> None:
        set_hints_visible(self.s, False)


def set_hints_visible(settings: Settings, visible: bool) -> None:
    """Show/hide every hint bar in the app and persist the choice."""
    settings.show_hints = visible
    try:
        settings.save()
    except OSError:
        pass
    for bar in list(_hint_bars):
        bar.setVisible(visible)


# --------------------------------------------------------------------- guide
_PAGES = [
    (
        "欢迎使用 kovadapt",
        "kovadapt 会让 KovaaK's 根据<i>你的表现</i>自动调整。每局结束后，它会："
        "<ol>"
        "<li>读取本局统计数据，以及启用后的原始鼠标输入；</li>"
        "<li>更新该场景下你的强项与弱项模型；</li>"
        "<li>重写场景的 <b>[Adaptive]</b> 副本，调整目标尺寸、弱区生成位置和移动节奏。</li>"
        "</ol>"
        "原始场景不会被改动，游戏本体也不会被修改；程序只处理 KovaaK's 自己的场景文件。",
    ),
    (
        "第一次训练",
        "<ol>"
        "<li>在<b>训练总览</b>中选择场景，然后点击<b>开始自适应训练</b>。"
        "kovadapt 会开始记录、生成播放列表并通过 Steam 启动 KovaaK's。</li>"
        "<li>在游戏中打开 <b>Playlists → kovadapt adaptive</b> 并开始训练。</li>"
        "<li>每局之间，场景会自动变难、变简单或加强你经常失误的区域；"
        "直接训练原始场景也会计入数据。</li>"
        "</ol>"
        "模型从第 1 局开始适配，通常训练约 10 局后，校准结果会更加可靠。",
    ),
    (
        "游戏内浮窗与性能优化",
        "<b>游戏内浮窗</b>会显示本局与个人基线、疲劳、难度和输入状态。"
        "可在训练总览中开启，通过<b>解锁位置</b>拖动并调整透明度。"
        "游戏需要使用无边框或窗口模式才能显示浮窗。"
        "<br><br>"
        "<b>性能优化</b>会根据硬件检查系统设置，并提供可选的一键修复；"
        "后台监测器还可以在游戏启动时设置高优先级并释放输入处理核心。",
    ),
    (
        "你的本地数据",
        "训练档案、鼠标轨迹、报告和录像片段全部保存在 "
        "<code>~/.kovadapt</code>，不会上传到网络。"
        "<br><br>"
        "各页面顶部的<b>提示</b>可以帮助你熟悉软件。点击任意提示上的 × 会"
        "隐藏全部提示；之后可从帮助菜单重新显示。本指南也会一直保留在帮助菜单中。",
    ),
]


class WelcomeDialog(QDialog):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.setWindowTitle("kovadapt — 启动指南")
        self.setModal(True)
        self.resize(520, 400)

        self.pages = QStackedWidget()
        for title, body in _PAGES:
            page = QWidget()
            head = QLabel(title)
            head.setProperty("headline", True)
            text = QLabel(body)
            text.setTextFormat(Qt.RichText)
            text.setWordWrap(True)
            text.setAlignment(Qt.AlignTop)
            v = QVBoxLayout(page)
            v.addWidget(head)
            v.addSpacing(6)
            v.addWidget(text, 1)
            self.pages.addWidget(page)

        self.progress = QLabel("")
        self.progress.setProperty("dim", True)
        self.again = QCheckBox(tr("Show this guide on the next start"))
        self.again.setChecked(False)     # finishing the guide dismisses it
        self.back_btn = QPushButton(tr("Back"))
        self.back_btn.clicked.connect(lambda: self._go(-1))
        self.next_btn = QPushButton(tr("Next"))
        self.next_btn.setProperty("accent", True)
        self.next_btn.clicked.connect(self._next)

        bar = QHBoxLayout()
        bar.addWidget(self.progress)
        bar.addStretch(1)
        bar.addWidget(self.back_btn)
        bar.addWidget(self.next_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self.pages, 1)
        lay.addWidget(self.again)
        lay.addLayout(bar)
        self._sync()

    def _go(self, step: int) -> None:
        self.pages.setCurrentIndex(
            max(0, min(self.pages.count() - 1, self.pages.currentIndex() + step)))
        self._sync()

    def _next(self) -> None:
        if self.pages.currentIndex() == self.pages.count() - 1:
            self.accept()
        else:
            self._go(+1)

    def _sync(self) -> None:
        i, n = self.pages.currentIndex(), self.pages.count()
        self.progress.setText(f"{i + 1} / {n}")
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setText(tr("Get started") if i == n - 1 else tr("Next"))

    def done(self, result: int) -> None:
        # Only finishing the guide dismisses it; closing it mid-read keeps
        # onboarding_done as-is so an unread guide returns next launch.
        if result == QDialog.Accepted:
            self.s.onboarding_done = not self.again.isChecked()
            try:
                self.s.save()
            except OSError:
                pass
        super().done(result)
