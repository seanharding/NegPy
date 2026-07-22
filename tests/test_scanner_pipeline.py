from dataclasses import replace

from negpy.domain.models import WorkspaceConfig
from negpy.features.process.scanner import scanner_token


def test_config_roundtrips_scanner_fields():
    matrix = (1.0, -0.09, -0.01, 0.05, 1.12, -0.33, 0.11, -0.34, 1.1)
    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, scanner_matrix=matrix, scanner_profile="My Scanner"))
    restored = WorkspaceConfig.from_flat_dict(cfg.to_dict())
    assert restored.process.scanner_matrix == matrix
    assert restored.process.scanner_profile == "My Scanner"
    # Tuple coercion survives a JSON round-trip (lists back to tuples).
    from_list = WorkspaceConfig.from_flat_dict({**cfg.to_dict(), "scanner_matrix": list(matrix)})
    assert from_list.process.scanner_matrix == matrix


def test_scanner_token_changes_with_matrix():
    off = replace(WorkspaceConfig().process, scanner_matrix=None)
    on = replace(WorkspaceConfig().process, scanner_matrix=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    on2 = replace(WorkspaceConfig().process, scanner_matrix=(1, -0.1, 0, 0, 1, 0, 0, 0, 1))
    assert scanner_token(off) != scanner_token(on)  # off vs on invalidates the render cache
    assert scanner_token(on) != scanner_token(on2)  # a different matrix invalidates too
    assert scanner_token(off) == "|sc:0"


def test_default_config_has_scanner_off():
    p = WorkspaceConfig().process
    assert p.scanner_matrix is None
    assert p.scanner_profile == "None"
