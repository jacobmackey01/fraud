from __future__ import annotations

try:
    from .operating_point_figure import main
except ImportError:
    from operating_point_figure import main


if __name__ == "__main__":
    raise SystemExit(main())
