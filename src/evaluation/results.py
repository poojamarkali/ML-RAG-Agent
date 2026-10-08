"""Save evaluation runs to data/eval_results/<name>.json so you can look at the
latest numbers later (CLI or GET /admin/eval/results) without re-running them."""

import json
from datetime import datetime

from config import settings


def save_results(name: str, rows: list[dict]):
    settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
    data = {"run_at": datetime.now().isoformat(timespec="seconds"), "rows": rows}
    (settings.eval_results_dir / f"{name}.json").write_text(json.dumps(data, indent=2, default=str))


def load_results() -> dict:
    folder = settings.eval_results_dir
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))} if folder.exists() else {}
