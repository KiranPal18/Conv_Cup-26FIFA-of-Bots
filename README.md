# Vector United

Vector United is a standalone, deterministic bot for the official AI Soccer Arena game.
It requires Python 3.11 or newer and no third-party packages or network access.
The included JSON model contains settings selected through paired self-play.

The policy uses the visible obstacle map to choose movement routes. In possession,
it searches legal move and kick combinations by simulating the ball's wall and
obstacle bounces, prefers direct goals, and moves toward an open attacking lane
when a safe shot is unavailable. Without possession, it estimates where a moving
ball can be intercepted and chases the opponent from the goal side.

Run the bot from this folder with:

```bash
python3 -m team_bot.bot --model team_bot/models/self_play_params.json
```

## State and action interface

The process reads one JSON message per line from standard input. For a message
with `"type": "observation"`, it uses the nested `observation` object, including
`player_id`, `opponent_id`, `attack_direction`, `state` (players, ball, field,
and obstacles), and `action_space`. It writes exactly one JSON action line to
standard output and flushes it immediately. An action contains a `move` from
the permitted directions or `STAY`; when the bot possesses the ball, it may
also include `kick` with a legal `direction` and `power`. On `match_end`, the
process exits. Diagnostics go to standard error.

`submission.json` is the entry point for the competition runner. The bot and
model use only relative paths inside this folder. `requirements.txt` lists no
third-party packages.
