import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent, QKeySequence
from PyQt6.QtWidgets import QApplication, QKeySequenceEdit

from negpy.desktop.view.widgets.key_sequence_edit import (
    KeypadAwareKeySequenceEdit,
    key_event_to_portable,
    key_event_to_sequence,
    should_capture_binding,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _key_press(key: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier) -> QKeyEvent:
    return QKeyEvent(QEvent.Type.KeyPress, key, modifiers)


def test_key_event_to_sequence_preserves_numpad_modifier():
    seq = key_event_to_sequence(_key_press(Qt.Key.Key_9, Qt.KeyboardModifier.KeypadModifier))
    assert seq is not None
    assert seq.toString(QKeySequence.SequenceFormat.PortableText) == "Num+9"


def test_key_event_to_sequence_ignores_number_row():
    assert key_event_to_sequence(_key_press(Qt.Key.Key_9)) is None


def test_key_event_to_portable_modifier_chord():
    assert key_event_to_portable(_key_press(Qt.Key.Key_T, Qt.KeyboardModifier.AltModifier)) == "Alt+T"


def test_key_event_to_portable_numpad():
    assert key_event_to_portable(_key_press(Qt.Key.Key_9, Qt.KeyboardModifier.KeypadModifier)) == "Num+9"


def test_key_event_to_portable_single_key():
    assert key_event_to_portable(_key_press(Qt.Key.Key_Q)) == "Q"


def test_should_capture_modifier_chord_even_when_unbound():
    event = _key_press(Qt.Key.Key_T, Qt.KeyboardModifier.AltModifier)
    assert should_capture_binding(event, frozenset(), portable="Alt+T") is True


def test_should_capture_single_key_only_when_bound():
    event = _key_press(Qt.Key.Key_Q)
    assert should_capture_binding(event, frozenset({"Q"}), portable="Q") is True
    assert should_capture_binding(event, frozenset(), portable="Q") is False


def test_should_not_capture_single_key_while_typing():
    event = _key_press(Qt.Key.Key_G)
    assert should_capture_binding(event, frozenset({"G"}), portable="G", allow_single_key=False) is False


def test_should_still_capture_chord_while_typing():
    event = _key_press(Qt.Key.Key_T, Qt.KeyboardModifier.AltModifier)
    assert should_capture_binding(event, frozenset(), portable="Alt+T", allow_single_key=False) is True


def test_keypad_aware_edit_differs_from_stock_qkeysequenceedit(qapp):
    stock = QKeySequenceEdit()
    aware = KeypadAwareKeySequenceEdit()
    event = _key_press(Qt.Key.Key_9, Qt.KeyboardModifier.KeypadModifier)

    stock.keyPressEvent(event)
    aware.keyPressEvent(event)

    assert stock.keySequence().toString(QKeySequence.SequenceFormat.PortableText) == "9"
    assert aware.keySequence().toString(QKeySequence.SequenceFormat.PortableText) == "Num+9"


def test_keypad_aware_edit_keeps_number_row_binding(qapp):
    edit = KeypadAwareKeySequenceEdit()
    edit.keyPressEvent(_key_press(Qt.Key.Key_9))
    assert edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText) == "9"


def test_numpad_and_number_row_bindings_do_not_conflict():
    row = QKeySequence("9")
    pad = QKeySequence("Num+9")
    assert row.toString(QKeySequence.SequenceFormat.PortableText) != pad.toString(QKeySequence.SequenceFormat.PortableText)
    assert row.matches(pad) == QKeySequence.SequenceMatch.NoMatch
