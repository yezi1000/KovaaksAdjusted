"""Small zh-CN vocabulary layer for user-facing GUI terminology.

Internal scenario names, configuration keys, archetype identifiers, and CLI
commands intentionally stay in English.  Keeping translation at the display
boundary prevents localized labels from changing saved data or control flow.
"""

ZH_CN = {
    # Navigation and sections.
    "Dashboard": "训练总览",
    "Scenarios": "场景库",
    "Analysis": "复盘分析",
    "What changed": "调整记录",
    "Adaptability": "自适应设置",
    "Optimizer": "性能优化",
    "How it learns": "学习原理",

    # Common actions.
    "Refresh": "刷新",
    "Launch KovaaK's": "启动 KovaaK's",
    "▶  Play adaptive task": "▶  开始自适应训练",
    "Start adapting": "开始分析",
    "Generate variant": "生成自适应版本",
    "Stop": "停止",
    "Stopping…": "正在停止…",
    "Replay": "回放",
    "Whole run": "整局回放",
    "Play video clip": "播放录像片段",
    "Open report…": "打开报告…",
    "Save settings": "保存设置",
    "Reset to defaults": "恢复默认值",
    "Open optimizer window": "打开性能优化器",
    "Re-scan": "重新扫描",
    "Fix all safe items": "修复全部安全项目",
    "Copy": "复制",

    # Dashboard and scenario browser.
    "Readiness": "准备度",
    "Form": "近期状态",
    "Load": "训练负荷",
    "Scenario:": "场景：",
    "Play": "训练",
    "Overlay": "游戏内浮窗",
    "Unlock": "解锁位置",
    "Lock": "锁定位置",
    "Show when a session starts": "训练开始时自动显示",
    "Search scenarios…": "搜索场景…",
    "All types": "全部类型",
    "Name": "名称",
    "Recently played": "最近训练",
    "Most runs": "训练次数最多",
    "Scenario": "场景",
    "Archetype": "类型",
    "Runs": "局数",
    "Accuracy": "准确率",
    "Calibration": "校准度",
    "Last played": "上次训练",
    "Never": "从未",
    "Select a scenario": "请选择一个场景",
    "clicking": "点击",
    "tracking": "跟枪",
    "switching": "目标切换",

    # Analysis and replay.
    "No run analyzed yet": "尚未分析任何一局",
    "Finish a run while watching (or open a saved report).":
        "完成一局训练，或打开以前保存的报告。",
    "Notable moments": "关键片段",
    "Trajectory replay": "鼠标轨迹回放",
    "Coach — every insight shows its evidence and sources":
        "训练建议 — 每条结论均显示依据和来源",
    "accuracy": "准确率",
    "kills": "击杀",
    "pace": "节奏",
    "mean flick": "平均甩枪",
    "path": "轨迹",
    "flicks": "甩枪",
    "shots": "射击点",
    "not watching": "未在分析",
    "watching": "正在分析",

    # Settings and optimizer.
    "Mouse & sensitivity": "鼠标与灵敏度",
    "Difficulty controller": "难度控制",
    "Weak-region targeting (bandit)": "弱区定向训练",
    "Anti-autopilot movement (Ornstein-Uhlenbeck)": "防肌肉记忆移动",
    "Telemetry & clips": "鼠标遥测与录像片段",
    "Advanced engine internals": "高级引擎设置",
    "Trace-informed dodge direction": "基于轨迹的闪避方向",
    "Session fatigue": "单次训练疲劳",
    "Per-archetype overrides (clicking is the baseline)": "按训练类型单独设置",
    "Record raw mouse telemetry while watching": "分析时记录原始鼠标输入",
    "Capture video clips of notable moments (needs kovadapt[clips])":
        "录制关键片段（需要 kovadapt[clips]）",
    "Performance optimizer": "性能优化器",
    "Detected hardware": "检测到的硬件",
    "System checkup": "系统检查",
    "Recommended for your hardware": "针对当前硬件的建议",

    # Theme and help menu.
    "Auto theme": "跟随系统",
    "Light": "浅色",
    "Dark": "深色",
    "Midnight": "午夜黑",
    "Accent color": "强调色",
    "Startup guide…": "启动指南…",
    "Show hints": "显示提示",
    "Open data folder": "打开数据文件夹",
    "TIP": "提示",
    "Show this guide on the next start": "下次启动时继续显示本指南",
    "Back": "上一步",
    "Next": "下一步",
    "Get started": "开始使用",
}


def tr(text: str) -> str:
    """Translate a user-facing term, falling back to the source string."""
    return ZH_CN.get(text, text)


def archetype_name(value: str) -> str:
    """Display an archetype without changing its persisted identifier."""
    return tr(value)
