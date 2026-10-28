"""Merge data/sweep/* into the static files the frontend loads: web/public/data/*.json.

    python -m core.export
"""

import json
from pathlib import Path

from core.evaluate import MAX_NEW_TOKENS, PROMPTS
from core.sweep import OUT as SWEEP, make_grid, write_json

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "web" / "public" / "data"


def main():
    layers = json.loads((SWEEP / "layers.json").read_text(encoding="utf-8"))
    order = ["fp32"] + [c["id"] for c in make_grid("full")]
    configs, whist = [], {}
    for cid in order:
        path = SWEEP / "configs" / f"{cid}.json"
        if not path.exists():
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        whist[cid] = cfg.pop("whist")
        configs.append(cfg)
    assert configs and configs[0]["id"] == "fp32", "run the sweep first (fp32 baseline missing)"

    write_json(DEST / "layers.json", layers)
    write_json(DEST / "configs.json", {"prompts": PROMPTS, "max_new_tokens": MAX_NEW_TOKENS, "configs": configs})
    write_json(DEST / "whist.json", whist)
    sizes = {p.name: f"{p.stat().st_size / 1e6:.2f} MB" for p in DEST.glob("*.json")}
    print(f"exported {len(configs)} configs -> {DEST}  {sizes}")


if __name__ == "__main__":
    main()
