"""Deterministic, obstacle-aware soccer policy with exact ball-path search."""

from __future__ import annotations

import heapq
import math
from typing import Any


ROOT_HALF = math.sqrt(0.5)
VECTORS = {
    "STAY": (0.0, 0.0),
    "UP": (0.0, 1.0), "UP_RIGHT": (ROOT_HALF, ROOT_HALF),
    "RIGHT": (1.0, 0.0), "DOWN_RIGHT": (ROOT_HALF, -ROOT_HALF),
    "DOWN": (0.0, -1.0), "DOWN_LEFT": (-ROOT_HALF, -ROOT_HALF),
    "LEFT": (-1.0, 0.0), "UP_LEFT": (-ROOT_HALF, ROOT_HALF),
}

DEFAULT_PARAMS = {
    "runway": 35.0,
    "lane": 0.36,
    "space_weight": 0.25,
    "lane_weight": 0.14,
    "danger_weight": 3.0,
    "shot_threshold": 35.0,
    "shot_risk": 75.0,
    "release_distance": 10.5,
    "defense_blend": 0.72,
    "future_opponent_weight": 0.0,
}


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def xy(data: dict[str, float]) -> tuple[float, float]:
    return float(data["x"]), float(data["y"])


def hits(position: tuple[float, float], radius: float, box: dict[str, float]) -> bool:
    cx = min(max(position[0], box["x"]), box["x"] + box["width"])
    cy = min(max(position[1], box["y"]), box["y"] + box["height"])
    return dist(position, (cx, cy)) < radius


def valid(position: tuple[float, float], field: dict[str, float], obstacles: list[dict[str, float]]) -> bool:
    r = field["player_radius"]
    return (r <= position[0] <= field["width"] - r
            and r <= position[1] <= field["height"] - r
            and not any(hits(position, r + 0.03, box) for box in obstacles))


def moves(me: tuple[float, float], field: dict[str, float], obstacles: list[dict[str, float]]) -> list[tuple[str, tuple[float, float]]]:
    result = [("STAY", me)]
    for name, (dx, dy) in VECTORS.items():
        if name != "STAY":
            target = (me[0] + dx * field["player_speed"], me[1] + dy * field["player_speed"])
            if valid(target, field, obstacles):
                result.append((name, target))
    return result


def line_clear(a: tuple[float, float], b: tuple[float, float], obstacles: list[dict[str, float]], margin: float) -> bool:
    dx, dy = b[0] - a[0], b[1] - a[1]
    for box in obstacles:
        lo_x, hi_x = box["x"] - margin, box["x"] + box["width"] + margin
        lo_y, hi_y = box["y"] - margin, box["y"] + box["height"] + margin
        near, far = 0.0, 1.0
        for origin, delta, low, high in ((a[0], dx, lo_x, hi_x), (a[1], dy, lo_y, hi_y)):
            if abs(delta) < 1e-10:
                if low <= origin <= high:
                    continue
                near, far = 1.0, 0.0
                break
            first, last = (low - origin) / delta, (high - origin) / delta
            near, far = max(near, min(first, last)), min(far, max(first, last))
            if near > far:
                break
        if near <= far:
            return False
    return True


class Navigator:
    def __init__(self, field: dict[str, float], obstacles: list[dict[str, float]]):
        self.obstacles = obstacles
        self.margin = field["player_radius"] + 0.1
        offset = self.margin + 0.4
        self.nodes = [p for box in obstacles
                      for p in ((x, y)
                                for x in (box["x"] - offset, box["x"] + box["width"] + offset)
                                for y in (box["y"] - offset, box["y"] + box["height"] + offset))
                      if valid(p, field, obstacles)]
        self.edges: list[list[tuple[int, float]]] = [[] for _ in self.nodes]
        for i, a in enumerate(self.nodes):
            for j in range(i + 1, len(self.nodes)):
                b = self.nodes[j]
                if line_clear(a, b, obstacles, self.margin):
                    length = dist(a, b)
                    self.edges[i].append((j, length))
                    self.edges[j].append((i, length))

    def costs(self, target: tuple[float, float]) -> list[float]:
        result = [float("inf")] * len(self.nodes)
        queue = []
        for i, node in enumerate(self.nodes):
            if line_clear(node, target, self.obstacles, self.margin):
                result[i] = dist(node, target)
                heapq.heappush(queue, (result[i], i))
        while queue:
            cost, i = heapq.heappop(queue)
            if cost != result[i]:
                continue
            for j, length in self.edges[i]:
                if cost + length < result[j]:
                    result[j] = cost + length
                    heapq.heappush(queue, (result[j], j))
        return result

    def length(self, start: tuple[float, float], target: tuple[float, float], costs: list[float]) -> float:
        direct = dist(start, target)
        best = direct if line_clear(start, target, self.obstacles, self.margin) else float("inf")
        for node, cost in zip(self.nodes, costs):
            if cost < best and line_clear(start, node, self.obstacles, self.margin):
                best = min(best, dist(start, node) + cost)
        return best if math.isfinite(best) else direct + 80.0


def advance_ball(
    x: float, y: float, vx: float, vy: float, remaining: float,
    field: dict[str, float], obstacles: list[dict[str, float]], sign: int = 0,
) -> tuple[float, float, float, float, float, bool, list[tuple[float, float]]]:
    """Simulate one game iteration, matching the engine's ball substeps."""
    r = 1.5
    if remaining <= 1e-8 or vx * vx + vy * vy <= 1e-12:
        return x, y, 0.0, 0.0, 0.0, False, []
    travel = min(8.0, remaining)
    substeps = max(1, math.ceil(travel / 0.675))
    step = travel / substeps
    path = []
    left = (field["width"] - field["goal_width"]) / 2
    right = left + field["goal_width"]
    for _ in range(substeps):
        norm = math.hypot(vx, vy)
        old_x, old_y = x, y
        nx, ny = x + vx / norm * step, y + vy / norm * step
        if left <= nx <= right and ((sign > 0 and ny + r >= field["height"])
                                     or (sign < 0 and ny - r <= 0)):
            path.append((nx, ny))
            return nx, ny, 0.0, 0.0, 0.0, True, path
        if nx - r < 0 or nx + r > field["width"]:
            vx = -vx
            nx = min(max(nx, r), field["width"] - r)
        if ny - r < 0 or ny + r > field["height"]:
            vy = -vy
            ny = min(max(ny, r), field["height"] - r)
        for box in obstacles:
            if not hits((nx, ny), r, box):
                continue
            crossed_x = old_x <= box["x"] - r or old_x >= box["x"] + box["width"] + r
            crossed_y = old_y <= box["y"] - r or old_y >= box["y"] + box["height"] + r
            if crossed_x:
                vx = -vx
            if crossed_y:
                vy = -vy
            if not crossed_x and not crossed_y:
                vx, vy = -vx, -vy
            norm = math.hypot(vx, vy)
            nx, ny = old_x + vx / norm * 0.05, old_y + vy / norm * 0.05
            break
        x, y = nx, ny
        remaining = max(0.0, remaining - step)
        path.append((x, y))
    if remaining <= 1e-8:
        vx, vy = 0.0, 0.0
    return x, y, vx, vy, remaining, False, path


def loose_trajectory(ball: dict[str, Any], field: dict[str, float], obstacles: list[dict[str, float]]) -> list[tuple[float, float]]:
    x, y = float(ball["x"]), float(ball["y"])
    vx, vy = float(ball["velocity"]["x"]), float(ball["velocity"]["y"])
    remaining = float(ball["remaining_kick_distance"])
    result = []
    for _ in range(14):
        if remaining > 0 and vx * vx + vy * vy > 0:
            x, y, vx, vy, remaining, _, _ = advance_ball(x, y, vx, vy, remaining, field, obstacles)
        result.append((x, y))
    return result


def shot_score(
    origin: tuple[float, float], direction: str, power: int, field: dict[str, float],
    obstacles: list[dict[str, float]], opponent: tuple[float, float], sign: int,
    shot_risk: float,
) -> tuple[bool, float]:
    dx, dy = VECTORS[direction]
    x, y = origin[0] + dx * 4.55, origin[1] + dy * 4.55
    vx, vy = dx * 8.0, dy * 8.0
    remaining = (32.0, 64.0, 96.0)[power - 1]
    risk = 0.0
    for turn in range(1, 13):
        x, y, vx, vy, remaining, goal, path = advance_ball(x, y, vx, vy, remaining, field, obstacles, sign)
        for location in path:
            reachable = 4.5 + 4.0 * turn
            gap = dist(location, opponent)
            if gap < reachable:
                risk = max(risk, min(1.0, (reachable - gap + 3.0) / 9.0))
        if goal:
            return True, 90.0 - shot_risk * risk
        if remaining <= 1e-8:
            break
    progress = sign * (y - origin[1]) / field["height"]
    return False, 5.0 * progress - 8.0 * risk - 0.025 * abs(x - field["width"] / 2)


def possession_action(
    observation: dict[str, Any], me: tuple[float, float], opponent: tuple[float, float],
    options: list[tuple[str, tuple[float, float]]], nav: Navigator, sign: int,
    params: dict[str, float],
) -> dict[str, Any]:
    state = observation["state"]
    field, obstacles = state["field"], state["obstacles"]
    steps = int(state["ball"].get("possession_steps", 0))
    goal_y = field["height"] if sign > 0 else 0.0
    dribble = (-float("inf"), "STAY")
    # The players move simultaneously before a tackle or kick is resolved.
    # Estimate the defender's position at that execution time. The estimate
    # is uncertain because its action is hidden, so training controls how
    # strongly the policy uses it.
    future_opponent = opponent
    future_weight = params["future_opponent_weight"]
    if future_weight > 0.0:
        chasing_move = min(moves(opponent, field, obstacles), key=lambda item: dist(item[1], me))[1]
        future_opponent = (
            opponent[0] + future_weight * (chasing_move[0] - opponent[0]),
            opponent[1] + future_weight * (chasing_move[1] - opponent[1]),
        )
    runway_y = me[1] + sign * min(params["runway"], max(0.0, abs(goal_y - me[1]) - 17.0))
    for fraction in (params["lane"], 1.0 - params["lane"]):
        target = (field["width"] * fraction, runway_y)
        costs = nav.costs(target)
        for move, location in options:
            gap = dist(location, future_opponent)
            value = -nav.length(location, target, costs) + params["space_weight"] * min(gap, 25.0)
            value += params["lane_weight"] * min(dist(target, future_opponent), 40.0)
            if gap < 15.0:
                value -= (15.0 - gap) * (params["danger_weight"] if steps >= 3 else 0.8)
            if move == "STAY":
                value -= 1.5
            if value > dribble[0]:
                dribble = (value, move)

    best = (-float("inf"), None)
    for move, origin in options:
        for direction in observation["action_space"]["kick"]["direction"]:
            if sign * VECTORS[direction][1] <= 0:
                continue
            for power in observation["action_space"]["kick"]["power"]:
                goal, value = shot_score(origin, direction, power, field, obstacles, opponent, sign,
                                         params["shot_risk"])
                if value > best[0]:
                    best = (value, {"move": move, "kick": {"direction": direction, "power": power}})

    must_release = steps >= 8 or (steps >= 3 and dist(me, opponent) < params["release_distance"])
    if best[1] is not None and (best[0] >= params["shot_threshold"] or must_release or
                                (abs(goal_y - me[1]) < 30 and best[0] > 8)):
        return best[1]
    return {"move": dribble[1]}


def choose_action(observation: dict[str, Any], params: dict[str, float] | None = None) -> dict[str, Any]:
    if params is None:
        params = DEFAULT_PARAMS
    state = observation["state"]
    field, obstacles, ball = state["field"], state["obstacles"], state["ball"]
    me = xy(state["players"][observation["player_id"]])
    opponent = xy(state["players"][observation["opponent_id"]])
    sign = 1 if observation["attack_direction"] == "UP" else -1
    options = moves(me, field, obstacles)
    nav = Navigator(field, obstacles)
    if ball["possession"] == observation["player_id"]:
        return possession_action(observation, me, opponent, options, nav, sign, params)
    if ball["possession"] == observation["opponent_id"]:
        target = opponent
        if dist(me, opponent) > 18:
            own_goal = (field["width"] / 2, 0.0 if sign > 0 else field["height"])
            blend = params["defense_blend"]
            target = (blend * opponent[0] + (1 - blend) * own_goal[0],
                      blend * opponent[1] + (1 - blend) * own_goal[1])
        targets = [(target, 1)]
    else:
        targets = [(location, i + 1) for i, location in enumerate(loose_trajectory(ball, field, obstacles))]
    best = (-float("inf"), "STAY")
    for target, turn in targets:
        costs = nav.costs(target)
        for move, location in options:
            path = nav.length(location, target, costs)
            if ball["possession"] == observation["opponent_id"]:
                score = -path + (0.2 if move != "STAY" else 0.0)
            else:
                lead = max(0.0, path - 5.0) / field["player_speed"]
                score = -max(turn, lead) * 4.0 - 0.18 * path
                if path <= 5.0 and turn <= 2:
                    score += 8.0
                if move == "STAY":
                    score -= 0.4
            if score > best[0]:
                best = (score, move)
    return {"move": best[1]}
