"""Key sequence capture widget.

Qt's QKeySequenceEdit drops KeyboardModifier.KeypadModifier, so numpad digits
are stored as the same binding as the number row (``9`` instead of ``Num+9``).
We preserve the keypad modifier so both keys can be bound independently.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QKeyCombination, Qt
from PyQt6.QtGui import QKeyEvent, QKeySequence
from PyQt6.QtWidgets import QKeySequenceEdit

_KEYPAD_MODIFIERS = (
    Qt.KeyboardModifier.ShiftModifier,
    Qt.KeyboardModifier.ControlModifier,
    Qt.KeyboardModifier.AltModifier,
    Qt.KeyboardModifier.MetaModifier,
    Qt.KeyboardModifier.KeypadModifier,
)

_MODIFIER_KEYS = frozenset(
    {
        Qt.Key.Key_Shift,
        Qt.Key.Key_Control,
        Qt.Key.Key_Alt,
        Qt.Key.Key_AltGr,
        Qt.Key.Key_Meta,
    }
)

_CHORD_MODIFIER_MASK = (
    Qt.KeyboardModifier.ControlModifier
    | Qt.KeyboardModifier.AltModifier
    | Qt.KeyboardModifier.MetaModifier
    | Qt.KeyboardModifier.KeypadModifier
)


def _sequence_from_event(event: QKeyEvent) -> QKeySequence | None:
    key = event.key()
    if key in (Qt.Key.Key_unknown,) or key in _MODIFIER_KEYS:
        return None

    mods = Qt.KeyboardModifier.NoModifier
    for modifier in _KEYPAD_MODIFIERS:
        if event.modifiers() & modifier:
            mods |= modifier
    return QKeySequence(QKeyCombination(mods, Qt.Key(key)))


def key_event_to_sequence(event: QKeyEvent) -> QKeySequence | None:
    """Build a portable QKeySequence from a numpad key event."""
    if not event.modifiers() & Qt.KeyboardModifier.KeypadModifier:
        return None
    return _sequence_from_event(event)


def key_event_to_portable(event: QKeyEvent) -> str | None:
    """Return the portable binding string for a key event, or None if incomplete."""
    sequence = _sequence_from_event(event)
    if sequence is None or sequence.isEmpty():
        return None
    return sequence.toString(QKeySequence.SequenceFormat.PortableText)


def normalize_binding(key: str) -> str:
    return key.strip().casefold()


def format_binding_display(portable: str) -> str:
    return normalize_binding(portable)


def should_capture_binding(
    event: QKeyEvent,
    known_bindings: frozenset[str],
    *,
    portable: str | None = None,
    allow_single_key: bool = True,
) -> bool:
    """Decide whether a search-field key press should be treated as shortcut lookup."""
    resolved = portable if portable is not None else key_event_to_portable(event)
    if not resolved:
        return False

    normalized = normalize_binding(resolved)
    normalized_known = {normalize_binding(binding) for binding in known_bindings if binding}

    if event.modifiers() & _CHORD_MODIFIER_MASK:
        return True

    if allow_single_key and normalized in normalized_known:
        return True

    return False


KnownBindingsProvider = Callable[[], frozenset[str]]


class KeypadAwareKeySequenceEdit(QKeySequenceEdit):
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Backspace:
            super().keyPressEvent(event)
            return

        sequence = key_event_to_sequence(event)
        if sequence is not None:
            self.setKeySequence(sequence)
            event.accept()
            return

        super().keyPressEvent(event)
