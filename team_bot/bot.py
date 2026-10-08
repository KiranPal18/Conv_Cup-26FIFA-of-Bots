from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .smart_policy import DEFAULT_PARAMS, choose_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="team_bot/models/self_play_params.json")
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    params = {**DEFAULT_PARAMS, **{
        name: float(model["params"][name])
        for name in DEFAULT_PARAMS if name in model["params"]
    }}

    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("type") == "match_end":
                return
            if message.get("type") != "observation":
                continue
            action = choose_action(message["observation"], params)
            print(json.dumps(action, separators=(",", ":")), flush=True)
        except Exception as error:
            # Diagnostics belong on stderr. Return a safe action if one turn fails.
            print(f"bot error: {error}", file=sys.stderr, flush=True)
            print('{"move":"STAY"}', flush=True)


if __name__ == "__main__":
    main()
