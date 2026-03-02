from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]
src_pkg = Path(__file__).resolve().parent.parent / "src" / "narrativepulse"
if src_pkg.exists():
    __path__.append(str(src_pkg))  # type: ignore[attr-defined]
