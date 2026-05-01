import sys
import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication

from negpy.desktop.controller import AppController
from negpy.desktop.session import DesktopSessionManager, AppState
from negpy.services.rendering.preview_manager import PreviewManager

if not QApplication.instance():
    _app = QApplication(sys.argv)


class TestAppController(unittest.TestCase):
    def setUp(self):
        self.mock_session_manager = MagicMock(spec=DesktopSessionManager)
        self.mock_session_manager.state = AppState()
        self.mock_session_manager.repo = MagicMock()

        # Patch GPU-touching classes before AppController.__init__ so no real GPU is created
        with (
            patch("negpy.desktop.controller.RenderWorker") as mock_rw_class,
            patch("negpy.desktop.controller.PreviewManager") as mock_pm_class,
        ):
            mock_rw_class.return_value = MagicMock()
            mock_pm_class.return_value = MagicMock(spec=PreviewManager)
            mock_pm_class.return_value.load_linear_preview.return_value = (None, (0, 0), {})
            self.controller = AppController(self.mock_session_manager)

    def tearDown(self):
        import gc

        # Stop all background threads before the controller is GC'd
        for thread in [
            self.controller.render_thread,
            self.controller.export_thread,
            self.controller.thumb_thread,
            self.controller.norm_thread,
            self.controller.discovery_thread,
            self.controller.preview_load_thread,
        ]:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()
        del self.controller
        gc.collect()

    def test_load_file_emits_zoom_reset(self):
        """Test that loading a file normally resets the zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng")

        mock_slot.assert_called_once_with(1.0)
        self.assertFalse(self.controller.state.hq_preview)

    def test_load_file_preserve_zoom(self):
        """Test that load_file with preserve_zoom=True skips resetting zoom."""
        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.load_file("dummy.dng", preserve_zoom=True)

        mock_slot.assert_not_called()

    def test_toggle_hq_preview_preserves_zoom(self):
        """Test that toggling HQ mode persists via session and preserves zoom."""
        self.controller.state.current_file_path = "dummy.dng"

        mock_slot = MagicMock()
        self.controller.zoom_requested.connect(mock_slot)

        self.controller.toggle_hq_preview()

        # Persistence delegated to session
        self.mock_session_manager.set_hq_preview.assert_called_once_with(True)

        # Zoom should NOT be reset
        mock_slot.assert_not_called()

    def test_preview_loaded_updates_state_and_emits_signal(self):
        """Successful preview loads should publish dimensions before rendering starts."""
        mock_slot = MagicMock()
        self.controller.preview_loaded.connect(mock_slot)
        self.controller.request_render = MagicMock()

        raw = object()
        dims = (1234, 5678)

        self.controller._on_preview_loaded("dummy.dng", raw, dims)

        self.assertIs(self.controller.state.preview_raw, raw)
        self.assertEqual(self.controller.state.original_res, dims)
        self.assertEqual(self.controller.state.current_file_path, "dummy.dng")
        mock_slot.assert_called_once_with()
        self.controller.request_render.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
