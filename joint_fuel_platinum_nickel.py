"""Probe exact Nickel completions while expanding Fuel and Platinum alternatives."""

from collections import Counter, deque
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from completion_aware_downstream import (
    BASELINE, GOAL, EARLY_WIDTH, CompletionSearch, sampled_days,
)
from early_gold_experiment import export_result
from four_stop_suffix_search import Budget, candidate, check_result, replay
from global_multistart_beam import (
    CatalogShortlist, MAX_SOLVES, MISSION_END, SearchLimit, TANK, TIME_LIMIT,
    VALIDATION_SECONDS,
)
from middle_route_repair import SOURCE, check_complete, load_reference, report, sweep
from six_stop_suffix_search import route_arrays


OUTPUT = Path(__file__).with_name("joint_fuel_platinum_nickel_best_result.json")
JOINT_SOLVES = 100_000
NICKEL_SOLVES = 60_000
VALIDATION_SOLVES = 215
PROBES_PER_PARENT = 24


def qualifies(score, threshold):
    return score > threshold if threshold == GOAL else score >= threshold


def required_stay(total, deposit, threshold):
    """Return the first legal integer stay that can reach the requested score."""
    if not qualifies(total + deposit, threshold):
        return None
    return next((days for days in range(1, 61)
                 if qualifies(total + min(deposit, days / 30), threshold)), None)


def flight_bands(earliest, latest):
    dates = range(earliest, latest + 1)
    return tuple(part for index in range(3)
                 if (part := dates[len(dates) * index // 3:len(dates) * (index + 1) // 3]))


def refill_details(fuel, deposit, stay, burn):
    available = TANK * min(deposit, stay / 30)
    added = min(available, TANK - (fuel - burn))
    return {"refill": added, "clipped": available - added, "fuel_burn": burn}


def completion_rank(state):
    return min(state.collected_materials), state.remaining_dv


class JointSearch(CompletionSearch):
    def schedule(self, state, stage):
        dates, minimum, reserve = super().schedule(state, stage)
        if stage == 0:
            target = self.reference[self.tail + 1].current_id
            goal_stay = required_stay(state.collected_materials[0],
                                      self.engine.udp.asteroid_masses[target], GOAL)
            goal_latest = MISSION_END - reserve - goal_stay if goal_stay else None
            if goal_latest is not None and goal_latest >= state.current_day + 1:
                dates = tuple(dict.fromkeys((*dates, goal_latest)))
        return dates, minimum, reserve

    def select_departures(self, states, protected):
        def rank(state):
            return state.remaining_dv / (TANK / 30) - state.current_day

        goal_states = [state for state in states if state.collected_materials[0] > GOAL]
        preferred = self.diverse(goal_states, EARLY_WIDTH // 2, rank, protected)
        remaining = self.diverse(states, EARLY_WIDTH, rank, protected)
        return list(dict.fromkeys((*preferred, *remaining)))[:EARLY_WIDTH]

    def nickel_transfer(self, parent, target, arrival):
        # Clamp the shared Lambert budget for this call, so cache hits cost zero.
        end = self.budget.solve_limit
        before = self.budget.solves
        self.budget.solve_limit = min(end, before + NICKEL_SOLVES - self.nickel_solves)
        try:
            return self.transfer(parent.current_id, target, parent.current_day, arrival)
        finally:
            self.nickel_solves += self.budget.solves - before
            self.budget.solve_limit = end

    def platinum_alternatives(self, fuel, stats):
        masses = self.engine.udp.asteroid_masses
        deposits = {
            kind: max((masses[target] for target in self.shortlist.groups[kind]
                       if target not in fuel.visited_asteroids), default=0)
            for kind in (1, 2)
        }
        reference = self.reference[-2].route[-1]
        preferred = fuel.current_day + reference.arrival_day - self.reference[-3].current_day
        earliest = fuel.current_day + 1
        visits = deque()
        for threshold in (GOAL, BASELINE):
            if not qualifies(fuel.collected_materials[0], threshold):
                continue
            minimum, nickel = (required_stay(fuel.collected_materials[kind], deposits[kind],
                                             threshold) for kind in (1, 2))
            if minimum is None or nickel is None:
                stats["no score-eligible deposits"] += 1
                continue
            latest = MISSION_END - nickel - 1 - minimum
            if earliest > latest:
                stats["no score-eligible window"] += 1
            dates = sampled_days(earliest, latest, preferred, latest, 14)
            visits.append(deque((threshold, day, MISSION_END - nickel - 1) for day in dates))

        def arrivals(threshold, day, departure_limit):
            self.budget.check_time()
            maximum = min(60, departure_limit - day)
            targets = self.refinement.targets(fuel, day, 1, reference.asteroid_id)
            for target in targets:
                self.budget.check_time()
                minimum = required_stay(fuel.collected_materials[1], masses[target], threshold)
                if target in fuel.visited_asteroids:
                    stats["visited"] += 1
                    continue
                if minimum is None:
                    stats["deposit too small"] += 1
                    continue
                if minimum > maximum:
                    stats["insufficient mining time"] += 1
                    continue
                stats["transfer attempts"] += 1
                burn = self.transfer(fuel.current_id, target, fuel.current_day, day)
                if not math.isfinite(burn) or burn < 0:
                    stats["invalid Lambert"] += 1
                    yield ()
                    continue
                if burn > fuel.remaining_dv:
                    stats["insufficient fuel"] += 1
                    yield ()
                    continue
                option = candidate(self.engine.udp, target, day, burn)
                children = []
                for stay in self.refinement.stays(fuel, 2, option, minimum, maximum):
                    child = self.engine.apply_candidate(fuel, option, stay)
                    if child is None:
                        stats["transition"] += 1
                    else:
                        children.append(child)
                yield children

        pending = deque()
        while visits:
            lane = visits.popleft()
            if lane:
                pending.append(arrivals(*lane.popleft()))
                visits.append(lane)
        deferred = deque()
        while pending or deferred:
            if pending:
                expansion = pending.popleft()
                batch = next(expansion, None)
                if batch is not None:
                    pending.append(expansion)
                    if batch:
                        yield batch[0]
                        deferred.extend(batch[1:])
                    else:
                        yield None
            if deferred:
                yield deferred.popleft()

    def probe_nickel(self, parent, stats, outcomes):
        earliest = parent.current_day + 1
        total = parent.collected_materials[2]
        masses = self.engine.udp.asteroid_masses
        unvisited = (target for target in self.shortlist.groups[2]
                     if target not in parent.visited_asteroids)
        max_deposit = max((masses[target] for target in unvisited), default=0)
        windows = []
        for threshold in (BASELINE, GOAL):
            if not all(qualifies(value, threshold) for value in parent.collected_materials[:2]):
                continue
            stay = required_stay(total, max_deposit, threshold)
            if stay is not None and earliest <= MISSION_END - stay:
                windows.append((threshold, MISSION_END - stay))
        if not windows:
            stats["no score-eligible window"] += 1
            return

        reference = self.reference[-1].route[-1]
        seen = set()
        by_band = {}
        probes = 0

        def probe(band, target, arrival, threshold):
            nonlocal probes
            self.budget.check_time()
            if (target, arrival) in seen:
                return
            if target in parent.visited_asteroids:
                stats["visited"] += 1
                return
            stay = required_stay(total, masses[target], threshold)
            if stay is None:
                stats["deposit too small"] += 1
                return
            maximum = min(60, MISSION_END - arrival)
            if stay > maximum:
                stats["insufficient mining time"] += 1
                return
            if probes == PROBES_PER_PARENT:
                return
            seen.add((target, arrival))
            probes += 1
            stats["transfer attempts"] += 1
            stats[("short", "medium", "long")[band] + " flight probes"] += 1
            dv = self.nickel_transfer(parent, target, arrival)
            if not math.isfinite(dv) or dv < 0:
                stats["invalid Lambert"] += 1
                return
            margin = parent.remaining_dv - dv
            if band not in by_band or margin > by_band[band][0]:
                by_band[band] = margin, target, arrival, threshold
            if outcomes["margin"] is None or margin > outcomes["margin"]:
                outcomes.update(margin=margin, margin_flight=arrival - parent.current_day)
            if margin < 0:
                stats["insufficient fuel"] += 1
                return
            child = self.engine.apply_candidate(
                parent, candidate(self.engine.udp, target, arrival, dv), maximum,
            )
            if child is None or not qualifies(completion_rank(child)[0], threshold):
                stats["transition or yield"] += 1
                return
            outcomes["floor"] += 1
            outcomes["goal"] += completion_rank(child)[0] > GOAL
            if outcomes["best"] is None or completion_rank(child) > completion_rank(outcomes["best"]):
                outcomes["best"] = child
                outcomes["best_flight"] = arrival - parent.current_day

        # Alternate score windows within each flight band, then rotate across bands.
        queues = [deque() for _ in range(3)]
        for threshold, latest in reversed(windows):
            for band, dates in enumerate(flight_bands(earliest, latest)):
                center = dates[len(dates) // 2]
                for arrival in dict.fromkeys((center, dates[-1], dates[0])):
                    queues[band].append((threshold, arrival))
        pending = deque()
        while any(queues):
            for band, queue in enumerate(queues):
                if queue:
                    threshold, arrival = queue.popleft()
                    self.budget.check_time()
                    targets = list(self.refinement.targets(parent, arrival, 2, reference.asteroid_id))
                    if arrival != reference.arrival_day:
                        targets.append(targets.pop(0))
                    pending.append((band, threshold, arrival, iter(targets)))

        def coarse(limit):
            while pending and probes < limit:
                band, threshold, arrival, targets = pending.popleft()
                before = probes
                for target in targets:
                    probe(band, target, arrival, threshold)
                    if probes > before:
                        pending.append((band, threshold, arrival, targets))
                        break

        coarse(PROBES_PER_PARENT // 2)
        deadlines = dict(windows)
        seeds = tuple(by_band.items())
        for offset in (-1, 1, -2, 2):
            for band, (_, target, arrival, threshold) in seeds:
                day = arrival + offset
                if earliest <= day <= deadlines[threshold]:
                    probe(band, target, day, threshold)
        coarse(PROBES_PER_PARENT)
        if probes == PROBES_PER_PARENT:
            stats["per-parent probe limit"] += 1

    def finish_joint(self, frontier, joint_end, time_end):
        stats = Counter()
        self.nickel_solves = 0
        joint_start = self.budget.solves
        self.budget.solve_limit = joint_start + max(0, joint_end - joint_start) // 5
        now = time.perf_counter()
        self.budget.deadline = min(time_end, now + max(0, time_end - now) / 5)
        gold_by_route = {state.route: state for state in frontier}
        generated, reason = sweep(self.children(state, 1, stats) for state in frontier)
        fuels = self.select_departures(generated, self.reference[-3])
        print(f"Fuel: generated {len(generated)}, retained {len(fuels)} | {reason} | {dict(stats)}")
        self.budget.solve_limit = joint_end
        self.budget.deadline = time_end
        platinum_stats, nickel_stats, counts = Counter(), Counter(), Counter()
        best = self.reference[-1]
        seen = set()
        fuel_logs = {}

        def examine(fuel, parent):
            nonlocal best
            if parent in seen:
                return
            seen.add(parent)
            if fuel not in fuel_logs:
                gold = gold_by_route[fuel.route[:-1]]
                stop = fuel.route[-1]
                burn = self.transfer(gold.current_id, fuel.current_id, gold.current_day,
                                     stop.arrival_day)
                fuel_logs[fuel] = refill_details(
                    gold.remaining_dv, self.engine.udp.asteroid_masses[fuel.current_id],
                    stop.mining_days, burn,
                )
            outcomes = dict(margin=None, margin_flight=None, floor=0, goal=0,
                            best=None, best_flight=None)
            try:
                self.probe_nickel(parent, nickel_stats, outcomes)
            finally:
                child = outcomes.pop("best")
                counts["floor candidates"] += outcomes["floor"]
                counts["goal candidates"] += outcomes["goal"]
                counts["floor parents"] += bool(outcomes["floor"])
                counts["goal parents"] += bool(outcomes["goal"])
                if child is not None and completion_rank(child) > completion_rank(best):
                    best = child
                print(json.dumps(dict(
                    ancestry=self.describe_ancestry(parent),
                    control=parent == self.reference[-2],
                    fuel_id=fuel.current_id, fuel_departure=fuel.current_day,
                    fuel_after_refill=fuel.remaining_dv, **fuel_logs[fuel],
                    platinum_id=parent.current_id, platinum_departure=parent.current_day,
                    platinum_burn=fuel.remaining_dv - parent.remaining_dv,
                    fuel_after_platinum=parent.remaining_dv,
                    screened_score=completion_rank(child)[0] if child is not None else None,
                    **outcomes,
                ), allow_nan=False))

        def alternatives(fuel):
            for parent in self.platinum_alternatives(fuel, platinum_stats):
                if parent is not None:
                    examine(fuel, parent)
                yield ()

        try:
            examine(self.reference[-3], self.reference[-2])
            _, reason = sweep(alternatives(fuel) for fuel in fuels)
        except SearchLimit as error:
            reason = str(error)
        if self.nickel_solves >= NICKEL_SOLVES:
            reason += " | Nickel fresh solve cap reached"
        print(f"Joint stop: {reason} | fresh solves {self.budget.solves - joint_start} "
              f"| Nickel fresh solves {self.nickel_solves}")
        print(f"Platinum: {dict(platinum_stats)} | nickel: {dict(nickel_stats)} "
              f"| parents probed {len(seen)} | {dict(counts)}")
        return best

    def run(self):
        end, time_end = self.budget.solve_limit, self.budget.deadline
        start, now = self.budget.solves, time.perf_counter()
        # Leave some preparation budget even in a small diagnostic run.
        prep_end = max(start + (end - start) // 3, end - JOINT_SOLVES)
        prep_time = now + (time_end - now) / 3
        try:
            self.budget.solve_limit = start + (prep_end - start) // 3
            self.budget.deadline = now + (prep_time - now) / 3
            repaired = self.repair()
            seeds = self.continue_route(repaired, start + 2 * (prep_end - start) // 3,
                                        now + 2 * (prep_time - now) / 3)
            print(f"7230 retained {len(seeds)} | changed repair schedules "
                  f"{sum(self.describe_ancestry(state) != self.describe_ancestry(self.reference[-1]) for state in seeds)}")
            self.budget.solve_limit, self.budget.deadline = prep_end, prep_time
            stats = Counter()
            generated, reason = sweep(self.children(seed, 0, stats) for seed in seeds)
            gold = self.select_departures(generated, self.reference[self.tail + 1])
            print(f"Gold: generated {len(generated)}, retained {len(gold)} "
                  f"| goal-capable {sum(state.collected_materials[0] > GOAL for state in gold)} "
                  f"| {reason} | {dict(stats)}")
            self.budget.deadline = time_end
            return self.finish_joint(gold, min(end, self.budget.solves + JOINT_SOLVES), time_end)
        finally:
            self.budget.solve_limit, self.budget.deadline = end, time_end


def validate_improvement(best, reference, indices, engine, budget):
    check_complete(reference, indices, best, engine.udp)
    official = engine.validate_route(best, require_full_route=True)
    states, minimum = replay(route_arrays(best), engine)
    final = states[-1]
    check_complete(reference, indices, final, engine.udp)
    replayed = engine.validate_route(final, require_full_route=True)
    check_result({
        "score": completion_rank(best)[0], "official_score": official,
        "remaining_dv": best.remaining_dv, "final_day": best.current_day,
        "collected_materials": {str(kind): mass
                                for kind, mass in enumerate(best.collected_materials)},
    }, final, replayed)
    print(f"Candidate official score {official!r} | independently replayed {replayed!r}")
    if min(official, replayed) <= BASELINE:
        return None
    budget.check_time()
    return final, replayed, minimum


def run(started, deadline):
    import beam_search as engine

    budget = Budget(engine.pk.lambert_problem, deadline)
    engine.pk.lambert_problem = budget
    try:
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        reference, indices, minimum, official = load_reference(saved, engine)
        budget.solve_limit = MAX_SOLVES - VALIDATION_SOLVES
        budget.deadline = deadline - VALIDATION_SECONDS
        shortlist = CatalogShortlist(engine.np, engine.udp, engine.esa["T_START"].mjd2000,
                                     budget.check_time)
        best = JointSearch(engine, shortlist, budget, reference, indices).run()
        budget.solve_limit, budget.deadline = MAX_SOLVES, deadline
        print(f"Best screened score {completion_rank(best)[0]!r}")
        verified = None
        if completion_rank(best)[0] > BASELINE:
            try:
                verified = validate_improvement(best, reference, indices, engine, budget)
            except (RuntimeError, ValueError) as error:
                print(f"Candidate validation failed: {error}")
        if verified is None:
            print("Reporting the verified incumbent.")
            report(reference[-1], official, minimum, budget, started)
            return 0
        final, official, minimum = verified
        try:
            export_result(final, official, output=OUTPUT,
                          algorithm="Joint Fuel-Platinum-Nickel optimization")
        except FileExistsError:
            print(f"{OUTPUT.name} already exists and was preserved.")
        else:
            print(f"Saved {OUTPUT.name}")
        report(final, official, minimum, budget, started)
        return 0
    except (RuntimeError, ValueError) as error:
        print(f"{error} | Lambert solves {budget.solves} "
              f"| runtime {time.perf_counter() - started:.2f} s")
        return 1
    finally:
        engine.pk.lambert_problem = budget.lambert


def main():
    started = time.perf_counter()
    deadline = started + TIME_LIMIT
    try:
        return subprocess.run(
            [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", str(started)],
            cwd=SOURCE.parent, timeout=max(0, deadline - time.perf_counter()),
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"{TIME_LIMIT}-second wall-clock limit reached. Worker stopped.")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        start = float(sys.argv[2])
        sys.exit(run(start, start + TIME_LIMIT))
    sys.exit(main())
