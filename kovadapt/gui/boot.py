"""Boot worker: real startup work behind the splash.

The opening is deliberately unhurried, so use it: scan the report history
and fit the cross-session skill model while the LEDs organize. Narrates
via `status` for the splash's terminal line; results land via
`trends_ready` whenever they are done (the reveal never blocks on this)."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ..config import Settings


class BootWorker(QThread):
    status = Signal(str)
    trends_ready = Signal(object)     # SkillTrends | None

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings

    def run(self) -> None:
        trends = None
        try:
            from ..analysis.skill import fit_skill, load_report_history

            self.status.emit("正在读取训练历史…")
            entries = load_report_history(self.s.profile_path)
            if entries:
                self.status.emit(
                    f"正在根据 {len(entries)} 局记录拟合能力曲线…")
                trends = fit_skill(entries)
                self.status.emit("能力模型已准备就绪")
            else:
                self.status.emit("尚无训练历史，将从零开始")
        except Exception:
            self.status.emit("")      # never let boot work break the boot
        self.trends_ready.emit(trends)
