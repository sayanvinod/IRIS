"""Helpers for locating Ultralytics training run directories."""
from pathlib import Path


def resolve_run_dir(run_dir: Path) -> Path:
    """Find run directory containing weights/best.pt.

    Canonical output location (after 2_train.py fix):
      runs/classify/<run_name>/weights/best.pt

    Legacy / fallback layouts handled:
      runs/<run_name>
      runs/classify/runs/<run_name>
      runs/classify/runs/classify/<run_name>
    """
    run_dir = Path(run_dir)

    # Fast path: caller gave the exact directory
    if (run_dir / "weights" / "best.pt").exists():
        return run_dir.resolve()

    run_name = run_dir.name

    # Ordered by likelihood — canonical path first
    candidates = [
        Path("runs") / "classify" / run_name,      # canonical (2_train.py v2)
        Path("runs") / run_name,                    # bare runs/ layout
        Path("runs") / "classify" / "runs" / run_name,
        Path("runs") / "classify" / "runs" / "classify" / run_name,
    ]
    for candidate in candidates:
        if (candidate / "weights" / "best.pt").exists():
            print(f"[runs_utils] Resolved '{run_name}' → {candidate.resolve()}")
            return candidate.resolve()

    # Last resort: recursive search under runs/
    runs_root = Path("runs")
    if runs_root.exists():
        for weights in runs_root.rglob("weights/best.pt"):
            if weights.parent.parent.name == run_name:
                found = weights.parent.parent
                print(f"[runs_utils] Resolved '{run_name}' via search → {found.resolve()}")
                return found.resolve()

    tried = [run_dir, *candidates]
    raise FileNotFoundError(
        f"No weights/best.pt found for run '{run_name}'.\nSearched:\n  "
        + "\n  ".join(str(p.resolve()) for p in tried)
        + f"\n  Recursive search under: {Path('runs').resolve()}"
    )
