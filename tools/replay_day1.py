"""本地第一天回放器。

用 docs/request.txt 的真实地图做 mock 游戏服务器, 逐回合调用真实 agent 决策循环
(handle_payload + StrategyEngine), 把响应命令按任务书规则结算, 跑完整个白天,
用于确定性复现"工人采集/建造失效 + 开拓者被任务模块冻结"的问题。

用法: python tools/replay_day1.py [--rounds 70] [--llm-resp-file FILE]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from future_war_agent.controller import handle_payload
from future_war_agent.strategy import engine as engine_module
from future_war_agent.strategy import forecast as forecast_module
from future_war_agent.strategy.engine import StrategyEngine
from future_war_agent.telemetry import TelemetryRecorder


class VirtualClock:
    """确定性虚拟时钟: 每次调用推进固定步长.

    step_ms=0 时所有耗时为 0 (本地"极速"档, 预算/看门狗永不触发);
    step_ms>0 时模拟慢服务器 (真实对局中 phase2 约 360ms/回合,
    NightForecast 100ms 看门狗持续超时, 决策路径与时序强相关).
    """

    def __init__(self, step_ms: float) -> None:
        self._step_ns = int(step_ms * 1_000_000)
        self._now_ns = 0

    def monotonic(self) -> float:
        self._now_ns += self._step_ns
        return self._now_ns / 1_000_000_000

    def perf_counter_ns(self) -> int:
        self._now_ns += self._step_ns
        return self._now_ns

    def install(self) -> None:
        engine_module.perf_counter_ns = self.perf_counter_ns
        forecast_module.monotonic = self.monotonic

TEMPLATE = json.loads(
    (ROOT / "docs" / "request.txt").read_text(encoding="utf-8")
)

WEAPON_NAMES = frozenset({"gatling", "railgun", "rocket"})
WEAPON_COST = 25
START_GOLD = 75
MAX_WEAPONS = 3
MINE_USES = 10
VENDOR_POS = (20, 16)
WEAPON_SHOP_POS = (25, 20)
MINE_RESPAWN_SPARES = {
    "stone": [(18, 8), (3, 12), (35, 28)],
    "iron": [(30, 25), (2, 18), (38, 12)],
    "copper": [(12, 6), (33, 4), (24, 30)],
}


def chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class MockGame:
    """按任务书规则结算单回合命令的最小游戏引擎 (仅白天语义)。"""

    def __init__(self, llm_responses: dict[int, str]) -> None:
        zones = TEMPLATE["mapInfo"]["zones"]
        self.mines: dict[tuple[int, int], str] = {}
        self.task_points: list[tuple[int, int]] = []
        for zone in zones:
            pos = (zone["pos"]["x"], zone["pos"]["y"])
            kind = zone["neutralType"]
            if kind in {"stone", "iron", "copper"}:
                self.mines[pos] = kind
            elif "TaskPoint" in kind:
                self.task_points.append(pos)
        self.enemy_structures: dict[tuple[int, int], str] = {}
        for role in TEMPLATE["teamEnemy"]["roles"]:
            pos = (role["pos"]["x"], role["pos"]["y"])
            self.enemy_structures[pos] = role["roleType"]

        station = TEMPLATE["teamOur"]["roles"][0]
        self.station_pos = (station["pos"]["x"], station["pos"]["y"])
        self.station_cells = {
            self.station_pos,
            (self.station_pos[0] + 1, self.station_pos[1]),
            (self.station_pos[0], self.station_pos[1] + 1),
            (self.station_pos[0] + 1, self.station_pos[1] + 1),
        }
        self.gold = START_GOLD
        self.roles: dict[int, dict[str, object]] = {
            10010: {"type": "worker", "pos": (9, 22), "backpack": []},
            10012: {"type": "worker", "pos": (8, 22), "backpack": []},
            10011: {"type": "pioneer", "pos": (9, 23), "backpack": []},
        }
        self.structures: dict[tuple[int, int], dict[str, object]] = {}
        self.next_structure_id = 40000
        self.mine_uses: dict[tuple[int, int], int] = {}
        self.last_results: dict[str, bool] = {}
        self.active_task: str | None = None
        self.llm_responses = llm_responses
        self.events: list[str] = []

    # ------------------------------------------------------------------ payload
    def build_payload(self, round_no: int) -> dict[str, object]:
        payload = copy.deepcopy(TEMPLATE)
        payload["roundNo"] = round_no
        payload["llmResp"] = self.llm_responses.get(round_no, "")
        payload["phaseTask"] = self.active_task or ""
        payload["errors"] = []
        payload["lastRoundRoleActionResults"] = dict(self.last_results)
        payload["teamOur"]["goldNum"] = self.gold

        roles: list[dict[str, object]] = []
        roles.append(
            {
                "id": 10013,
                "pos": {"x": self.station_pos[0], "y": self.station_pos[1]},
                "roleType": "station",
                "health": 1500,
                "attackPower": 0,
                "attackRange": 0,
                "level": 1,
                "backPackCapability": 0,
                "backpack": [],
            }
        )
        weapon_specs = {
            "gatling": (10, 4),
            "railgun": (10, 7),
            "rocket": (20, 2147483647),
        }
        for pos, structure in sorted(self.structures.items()):
            role_type = structure["type"]
            base = {
                "id": structure["id"],
                "pos": {"x": pos[0], "y": pos[1]},
                "roleType": role_type,
                "health": 1000,
                "attackPower": 0,
                "attackRange": 0,
                "level": 1,
                "backPackCapability": 0,
                "backpack": [],
            }
            if role_type in weapon_specs:
                power, rng = weapon_specs[role_type]
                base["attackPower"] = power
                base["attackRange"] = rng
            roles.append(base)
        for rid, role in self.roles.items():
            roles.append(
                {
                    "id": rid,
                    "pos": {"x": role["pos"][0], "y": role["pos"][1]},
                    "roleType": role["type"],
                    "health": 220 if role["type"] == "worker" else 200,
                    "attackPower": 0,
                    "attackRange": 0,
                    "level": 1,
                    "backPackCapability": 100 if role["type"] == "worker" else 40,
                    "backpack": list(role["backpack"]),
                }
            )
        payload["teamOur"]["roles"] = roles

        zones: list[dict[str, object]] = []
        for pos, kind in self.mines.items():
            zones.append({"neutralType": kind, "pos": {"x": pos[0], "y": pos[1]}})
        for zone in TEMPLATE["mapInfo"]["zones"]:
            if zone["neutralType"] not in {"stone", "iron", "copper"}:
                zones.append(copy.deepcopy(zone))
        payload["mapInfo"]["zones"] = zones
        payload["robot"]["roles"] = []
        return payload

    # ------------------------------------------------------------------结算
    def _static_blocked(self, pos: tuple[int, int]) -> bool:
        return (
            pos in self.station_cells
            or pos in self.structures
            or pos in self.mines
            or pos in self.enemy_structures
            or pos == VENDOR_POS
            or pos == WEAPON_SHOP_POS
            or pos in self.task_points
        )

    def apply_commands(self, round_no: int, commands: dict[str, dict]) -> None:
        self.events = []
        results: dict[str, bool] = {}

        movers: dict[int, tuple[int, int]] = {}
        parsed: dict[int, dict] = {}
        for rid_str, cmd in commands.items():
            rid = int(rid_str)
            parsed[rid] = cmd
            if cmd.get("action") == "move" and rid in self.roles:
                targets = cmd.get("targetPos") or []
                if targets:
                    movers[rid] = (targets[0]["x"], targets[0]["y"])

        occupied = {tuple(role["pos"]) for role in self.roles.values()}
        targets_count: dict[tuple[int, int], int] = {}
        for target in movers.values():
            targets_count[target] = targets_count.get(target, 0) + 1
        for rid, target in movers.items():
            current = tuple(self.roles[rid]["pos"])
            clash = targets_count[target] > 1
            swap = (
                not clash
                and target in movers.values()
                and any(
                    other_target == current and other != rid
                    for other, other_target in movers.items()
                )
            )
            blocked_by_stationary = (
                target in occupied
                and not any(
                    other == target and other_rid != rid and other_rid in movers
                    for other_rid, other in (
                        (r, tuple(self.roles[r]["pos"])) for r in self.roles
                    )
                )
            )
            if (
                chebyshev(current, target) != 1
                or self._static_blocked(target)
                or clash
                or swap
                or blocked_by_stationary
            ):
                results[str(rid)] = False
                self.events.append(
                    f"move-blocked id={rid} {current}->{target}"
                )
            else:
                self.roles[rid]["pos"] = list(target)
                results[str(rid)] = True

        for rid, cmd in parsed.items():
            if str(rid) in results:
                continue
            if rid not in self.roles:
                results[str(rid)] = False
                continue
            role = self.roles[rid]
            pos = tuple(role["pos"])
            kind = cmd.get("action")
            ok = False
            if kind == "collect":
                targets = cmd.get("targetPos") or []
                target = (targets[0]["x"], targets[0]["y"]) if targets else None
                if (
                    target in self.mines
                    and chebyshev(pos, target) == 1
                    and len(role["backpack"]) < (100 if role["type"] == "worker" else 40)
                ):
                    material = self.mines[target]
                    role["backpack"].append(material)
                    self.mine_uses[target] = self.mine_uses.get(target, 0) + 1
                    if self.mine_uses[target] >= MINE_USES:
                        self._respawn_mine(target, material)
                    ok = True
                    self.events.append(
                        f"collect id={rid} {material} @ {pos}"
                        f" backpack={len(role['backpack'])}"
                    )
            elif kind == "build":
                targets = cmd.get("targetPos") or []
                name = cmd.get("name")
                if targets:
                    target = (targets[0]["x"], targets[0]["y"])
                    adjacent = chebyshev(pos, target) == 1
                    free = (
                        not self._static_blocked(target)
                        and target not in occupied
                    )
                    if name == "wall":
                        stone_count = role["backpack"].count("stone")
                        if (
                            adjacent
                            and free
                            and stone_count >= 1
                        ):
                            role["backpack"].remove("stone")
                            self.structures[target] = {
                                "id": self.next_structure_id,
                                "type": "wall",
                            }
                            self.next_structure_id += 1
                            ok = True
                            self.events.append(f"build-wall id={rid} @ {target}")
                    elif name in WEAPON_NAMES:
                        weapon_count = sum(
                            1
                            for s in self.structures.values()
                            if s["type"] in WEAPON_NAMES
                        )
                        if (
                            adjacent
                            and free
                            and self.gold >= WEAPON_COST
                            and weapon_count < MAX_WEAPONS
                        ):
                            self.gold -= WEAPON_COST
                            self.structures[target] = {
                                "id": 10020 + weapon_count,
                                "type": name,
                            }
                            ok = True
                            self.events.append(f"build-weapon id={rid} {name} @ {target}")
            elif kind == "sell":
                if chebyshev(pos, VENDOR_POS) == 1:
                    name = cmd.get("name")
                    prices = {
                        item["name"]: item["price"]
                        for item in TEMPLATE["vendorShopList"]
                    }
                    quantity = cmd.get("num")
                    available = role["backpack"].count(name)
                    actual = min(quantity or available, available)
                    if name in prices and actual > 0:
                        for _ in range(actual):
                            role["backpack"].remove(name)
                        self.gold += prices[name] * actual
                        ok = True
                        self.events.append(
                            f"sell id={rid} {name}x{actual} gold={self.gold}"
                        )
            elif kind == "buy":
                # 任务书: buy 仅在武器商店可用; 小贩只收购矿石, 不出售.
                prices = {
                    item["name"]: item["price"]
                    for item in TEMPLATE["weaponShopList"]
                }
                name = cmd.get("name")
                quantity = cmd.get("num") or 1
                if (
                    name in prices
                    and chebyshev(pos, WEAPON_SHOP_POS) == 1
                ):
                    cost = prices[name] * quantity
                    if self.gold >= cost:
                        self.gold -= cost
                        role["backpack"].extend([name] * quantity)
                        ok = True
                        self.events.append(f"buy id={rid} {name}x{quantity}")
            elif kind == "acceptTask":
                if role["type"] == "pioneer":
                    near_task = any(
                        chebyshev(pos, tp) == 1 for tp in self.task_points
                    )
                    if near_task and not self.active_task:
                        self.active_task = "任务类型:自进化类1 前往任务点完成挑战"
                        ok = True
                        self.events.append(f"acceptTask id={rid} @ {pos}")
            else:
                ok = True
            results[str(rid)] = ok

        self.last_results = results

    def _respawn_mine(self, pos: tuple[int, int], material: str) -> None:
        del self.mines[pos]
        del self.mine_uses[pos]
        for spare in MINE_RESPAWN_SPARES[material]:
            if (
                spare not in self.mines
                and not self._static_blocked(spare)
                and spare not in {tuple(r["pos"]) for r in self.roles.values()}
            ):
                self.mines[spare] = material
                self.events.append(f"mine-respawn {material} {pos}->{spare}")
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=70)
    parser.add_argument("--llm-resp-file", type=str, default="")
    parser.add_argument("--clock-step-ms", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    clock = VirtualClock(args.clock_step_ms)
    clock.install()

    llm_responses: dict[int, str] = {}
    if args.llm_resp_file:
        raw = json.loads(Path(args.llm_resp_file).read_text(encoding="utf-8"))
        llm_responses = {int(k): v for k, v in raw.items()}

    game = MockGame(llm_responses)
    telemetry = TelemetryRecorder(max_samples=512)
    engine = StrategyEngine(telemetry=telemetry, clock=clock.monotonic)

    pioneer_missing = 0
    pioneer_trail: list[tuple[int, tuple[int, int], str]] = []
    worker_trails: dict[int, list[tuple[int, int]]] = {10010: [], 10012: []}
    stone_collected = 0

    for round_no in range(1, args.rounds + 1):
        payload = game.build_payload(round_no)
        response = handle_payload(
            payload,
            planner=engine.plan,
            telemetry=telemetry,
            clock=clock.monotonic,
        )
        commands = response.get("roleCommandMap") or {}
        game.apply_commands(round_no, commands)

        command_map = {int(rid): cmd.get("action", "?") for rid, cmd in commands.items()}
        if 10011 not in command_map:
            pioneer_missing += 1
        pioneer_pos = tuple(game.roles[10011]["pos"])
        pioneer_trail.append(
            (round_no, pioneer_pos, command_map.get(10011, "NONE"))
        )
        for wid in worker_trails:
            worker_trails[wid].append(tuple(game.roles[wid]["pos"]))
        stone_collected += sum(
            1
            for role in game.roles.values()
            for item in role["backpack"]
            if item == "stone"
        )

        if args.verbose or round_no % 5 == 0 or game.events:
            walls = sum(1 for s in game.structures.values() if s["type"] == "wall")
            weapons = sum(1 for s in game.structures.values() if s["type"] in WEAPON_NAMES)
            print(
                f"R{round_no:>3} gold={game.gold:>3} walls={walls} weapons={weapons} "
                f"10010={worker_trails[10010][-1]} bp={len(game.roles[10010]['backpack'])} "
                f"10012={worker_trails[10012][-1]} bp={len(game.roles[10012]['backpack'])} "
                f"pioneer={pioneer_pos} cmd={command_map.get(10011, 'NONE')}"
            )
            for event in game.events:
                print(f"      {event}")

    walls = sum(1 for s in game.structures.values() if s["type"] == "wall")
    weapons = sum(1 for s in game.structures.values() if s["type"] in WEAPON_NAMES)
    print()
    print("=== 第一天回放总结 ===")
    print(f"回合数: {args.rounds}")
    print(f"最终金币: {game.gold}")
    print(f"建造围墙数: {walls}")
    print(f"建造武器数: {weapons}")
    print(f"工人背包(10010): {game.roles[10010]['backpack']}")
    print(f"工人背包(10012): {game.roles[10012]['backpack']}")
    print(f"开拓者无命令回合数: {pioneer_missing}/{args.rounds}")
    print(f"开拓者最终位置: {tuple(game.roles[10011]['pos'])} (起点 (9,23))")
    print(f"累计采集石头: {stone_collected}")
    accepted = game.active_task is not None
    print(f"白天是否接取任务: {accepted}")

    worker_summary = {}
    for wid, trail in worker_trails.items():
        reversals = 0
        for i in range(2, len(trail)):
            prev_delta = (trail[i - 1][0] - trail[i - 2][0], trail[i - 1][1] - trail[i - 2][1])
            curr_delta = (trail[i][0] - trail[i - 1][0], trail[i][1] - trail[i - 1][1])
            if prev_delta != (0, 0) and curr_delta != (0, 0):
                dot = prev_delta[0] * curr_delta[0] + prev_delta[1] * curr_delta[1]
                if dot < 0:
                    reversals += 1
        distinct = len(set(trail))
        worker_summary[wid] = (distinct, reversals)
        print(
            f"工人{wid}: 停留格子数={distinct} 方向折返次数={reversals}"
        )
    if walls == 0:
        print(">>> 复现: 第一天未建造任何围墙")
    if weapons == 0:
        print(">>> 复现: 第一天未建造任何武器")
    if pioneer_missing > args.rounds * 0.1:
        print(f">>> 复现: 开拓者 {pioneer_missing} 回合被冻结(无命令)")


if __name__ == "__main__":
    main()
