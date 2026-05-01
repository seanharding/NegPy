from unittest.mock import MagicMock

from negpy.desktop.view.widgets.sliders import CompactSlider


def test_adjust_by_emits_change_and_commit(qapp):
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)
    changed = MagicMock()
    committed = MagicMock()
    slider.valueChanged.connect(changed)
    slider.valueCommitted.connect(committed)

    slider.adjust_by(0.1)

    assert slider.value() == 1.1
    changed.assert_called_once_with(1.1)
    committed.assert_called_once_with(1.1)


def test_adjust_by_clamps_to_range(qapp):
    slider = CompactSlider("Density", 0.0, 2.0, 1.0)

    slider.adjust_by(99.0)
    assert slider.value() == 2.0

    slider.adjust_by(-99.0)
    assert slider.value() == 0.0
