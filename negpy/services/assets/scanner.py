import os
import re
import tomllib
from typing import List, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.paths import get_resource_path

# The "off" entry — no scanner correction. Unlike crosstalk's "Default", there is no
# built-in matrix: a scanner matrix only exists once the user calibrates their setup.
NONE_NAME = "None"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug or "scanner"


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class ScannerProfiles:
    """
    TOML I/O for user scanner (sensor+light) correction matrices, mirroring
    CrosstalkProfiles. Files live in APP_CONFIG.scanner_dir; the "None" entry means no
    correction. Disk I/O only on dropdown build / selection — matrices are baked into
    ProcessConfig for rendering.
    """

    NONE_NAME = NONE_NAME

    @staticmethod
    def _scan_dir(directory: str) -> dict:
        """Maps display-name -> flat 9-float matrix for valid .toml files in a directory."""
        result: dict = {}
        if not os.path.isdir(directory):
            return result
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            parsed = ScannerProfiles._parse_file(os.path.join(directory, fname))
            if parsed is None:
                continue
            name, matrix = parsed
            name = name or fname[:-5]
            if name != NONE_NAME:
                result[name] = matrix
        return result

    @staticmethod
    def scan_bundled() -> dict:
        """Read-only matrices shipped with the app (usually none — scanner matrices are
        per-setup), keyed by display name."""
        return ScannerProfiles._scan_dir(get_resource_path("scanner"))

    @staticmethod
    def scan_user() -> dict:
        """User-calibrated matrices in the docs folder, keyed by display name."""
        return ScannerProfiles._scan_dir(APP_CONFIG.scanner_dir)

    @staticmethod
    def _scan() -> dict:
        """Bundled ∪ user matrices, keyed by display name; bundled wins."""
        return {**ScannerProfiles.scan_user(), **ScannerProfiles.scan_bundled()}

    @staticmethod
    def _parse_file(path: str) -> Optional[tuple]:
        """Parses a .toml file to (name, flat 9-float list), or None if invalid."""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            rows = data.get("matrix")
            if not isinstance(rows, list) or len(rows) != 3:
                return None
            flat: List[float] = []
            for row in rows:
                if not isinstance(row, list) or len(row) != 3:
                    return None
                for v in row:
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        return None
                    flat.append(float(v))
            raw_name = data.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
            return name, flat
        except Exception:
            return None

    @staticmethod
    def list_profiles() -> List[str]:
        """["None", *sorted custom display-names]."""
        return [NONE_NAME, *sorted(ScannerProfiles._scan().keys())]

    @staticmethod
    def get_matrix(name: str) -> Optional[List[float]]:
        """Flat 9-float list for a profile, or None for "None" / missing / invalid
        (None ⇒ the render path applies no scanner correction)."""
        if name == NONE_NAME:
            return None
        return ScannerProfiles._scan().get(name)

    @staticmethod
    def is_bundled(name: str) -> bool:
        """True for read-only profiles: the "None" entry or any bundled matrix."""
        return name == NONE_NAME or name in ScannerProfiles.scan_bundled()

    @staticmethod
    def path_for_name(name: str) -> str:
        """Filesystem path a user profile with this display name would use."""
        return os.path.join(APP_CONFIG.scanner_dir, f"{_slugify(name)}.toml")

    @staticmethod
    def save(name: str, matrix: List[float]) -> str:
        """Write a user profile TOML (row-major 3×3) and return its path."""
        os.makedirs(APP_CONFIG.scanner_dir, exist_ok=True)
        rows = "\n".join("  [{:.6g}, {:.6g}, {:.6g}],".format(*matrix[i * 3 : i * 3 + 3]) for i in range(3))
        content = f'name = "{_escape_toml_string(name)}"\nmatrix = [\n{rows}\n]\n'
        path = ScannerProfiles.path_for_name(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    @staticmethod
    def delete(name: str) -> None:
        """Remove the user profile file whose display name matches (no-op if absent)."""
        directory = APP_CONFIG.scanner_dir
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            parsed = ScannerProfiles._parse_file(os.path.join(directory, fname))
            display = (parsed[0] if parsed and parsed[0] else fname[:-5]) if parsed else None
            if display == name:
                os.remove(os.path.join(directory, fname))
