"""The zh-CN display layer must not leak into persisted identifiers."""

from kovadapt.gui.i18n import archetype_name, tr


def test_core_navigation_terms_are_localized():
    assert tr("Dashboard") == "训练总览"
    assert tr("Analysis") == "复盘分析"
    assert tr("Optimizer") == "性能优化"


def test_unknown_and_internal_values_are_stable():
    assert tr("1wall 6targets small") == "1wall 6targets small"
    assert archetype_name("tracking") == "跟枪"
    assert archetype_name("unknown-archetype") == "unknown-archetype"
