"""Optimizer tab: compact launcher for the full optimizer window.

The real experience lives in optimizer_window.OptimizerWindow (a separate
top-level window — the free Process Lasso alternative). This tab shows a
one-line status and opens/raises that window; the window owns the watchdog
and keeps running when hidden.
"""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..config import Settings
from .i18n import tr


class OptimizerView(QWidget):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.window = None   # created lazily; import stays off the startup path

        head = QLabel(tr("Performance optimizer"))
        head.setProperty("headline", True)
        blurb = QLabel(
            "检测硬件与系统设置，按项目提供可选的一键修复；后台监测器可在每次"
            "启动游戏时设置高优先级并释放输入处理核心，同时给出适合当前硬件的"
            "启动参数和设置建议。\n\n"
            "所有优化均需手动启用，可撤销且只对当前用户生效；kovadapt 只会处理"
            "游戏进程，不会自动修改其他程序。"
        )
        blurb.setWordWrap(True)
        blurb.setProperty("dim", True)
        open_btn = QPushButton(tr("Open optimizer window"))
        open_btn.setProperty("accent", True)
        open_btn.clicked.connect(self.open_window)
        btn_row = QHBoxLayout()          # natural-width CTA, not a 950px slab
        btn_row.addWidget(open_btn)
        btn_row.addStretch(1)

        lay = QVBoxLayout(self)
        # ZERO, explicitly. Every section view inherited Qt's ~9px default
        # layout margin, while the section's own H1, its divider rule and
        # every panel sit flush to shell._Section's column — so bare page
        # text was the only thing indented, and lined up with nothing on the
        # screen. The column IS the measure; panels pad their own contents.
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)
        lay.addWidget(head)
        lay.addWidget(blurb)
        lay.addLayout(btn_row)
        lay.addStretch(1)

    def open_window(self) -> None:
        if self.window is None:
            from .optimizer_window import OptimizerWindow

            self.window = OptimizerWindow(self.s)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def restyle(self, *_pal) -> None:
        if self.window is not None:
            self.window.restyle()

    def note_report(self, rep) -> None:
        """Feed per-run input health into the watchdog evidence panel."""
        if self.window is not None:
            self.window.note_input_health(getattr(rep, "input_health", None) or {})

    def shutdown(self) -> None:
        if self.window is not None:
            self.window.shutdown()
