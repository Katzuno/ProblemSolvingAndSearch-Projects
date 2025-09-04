#!/usr/bin/env python3
"""
P3 — Local Search (AIMA p. ~120): N‑Queens via Hill‑Climbing cu Random Restarts

Algoritm de local search: Steepest‑Ascent Hill‑Climbing cu opțiuni de Sideways Moves și Random Restarts

Output (agregat pe TRIALS):
- Rata de succes, pasi medii, restarts medii, timp mediu, best/worst conflicts
- Un exemplu de soluție (daca exista)
"""
from __future__ import annotations
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================
# CONFIG
# =============================
N: int = 8                       # dimensiunea tablei N‑Queens
TRIALS: int = 50                 # numar rulari independente

MAX_RESTARTS: int = 200          # maxim restarts permis per trial
ALLOW_SIDEWAYS: bool = True      # permite mutari cu delta==0
SIDEWAYS_LIMIT: int = 100        # max mutari sideways consecutive per restart
STEP_LIMIT: int = 10000          # max steps per trial (safety)

# Setam random_seed sa avem acelasi rezultat la diferite rulari. None pentru random complet
RANDOM_SEED: Optional[int] = 42

# Export rezultate
EXPORT_PER_TRIAL_CSV: bool = True
PER_TRIAL_CSV_PATH: str = "p3_trials.csv"
EXPORT_SUMMARY_CSV: bool = True
SUMMARY_CSV_PATH: str = "p3_summary.csv"

# Experimente multi‑N (pentru graficul ratei de succes vs N)

RUN_MULTI_N: bool = True
N_GRID: list[int] = [8, 16, 32]
TRIALS_PER_N: int = 30

# Plot
PLOT_SUCCESS_VS_N: bool = True
SUCCESS_PNG_PATH: str = "p3_success_vs_n.png"


@dataclass
class RunMetrics:
    solved: bool
    steps: int
    restarts: int
    time_sec: float
    start_conflicts: int
    end_conflicts: int
    solution: Optional[List[int]]

@dataclass
class Summary:
    algo: str
    n: int
    trials: int
    successes: int
    avg_steps: float
    avg_restarts: float
    avg_time_sec: float
    best_end_conflicts: int
    worst_end_conflicts: int
    example_solution: Optional[List[int]]

# Starea: listă col[r] = coloana damei pe linia r

_def_pair = lambda c: (c * (c - 1)) // 2

class ConflictCounter:
    """Numara rapid conflictele și evalueaza delta pentru mutari single‑queen.
    Diagonale:
      d1 = r - c + (N-1)  în [0, 2N-2]
      d2 = r + c          în [0, 2N-2]
    Pairs = sum C(cnt,2) pe coloane și diagonale.
    """
    def __init__(self, n: int, state: List[int]):
        self.n = n
        self.state = state[:]  # copie
        self.col = [0] * n
        self.d1 = [0] * (2 * n - 1)
        self.d2 = [0] * (2 * n - 1)
        for r, c in enumerate(state):
            self.col[c] += 1
            self.d1[r - c + (n - 1)] += 1
            self.d2[r + c] += 1
        self.total_pairs = self._pairs_sum()

    def _pairs_sum(self) -> int:
        return sum(_def_pair(x) for x in self.col) \
             + sum(_def_pair(x) for x in self.d1) \
             + sum(_def_pair(x) for x in self.d2)

    def total_conflicts(self) -> int:
        return self.total_pairs

    def _affected_indices(self, r: int, c_old: int, c_new: int):
        d1_old = r - c_old + (self.n - 1)
        d2_old = r + c_old
        d1_new = r - c_new + (self.n - 1)
        d2_new = r + c_new
        affected = {
            ("col", c_old), ("col", c_new),
            ("d1", d1_old), ("d1", d1_new),
            ("d2", d2_old), ("d2", d2_new),
        }
        return affected, d1_old, d2_old, d1_new, d2_new

    def delta_if_move(self, r: int, c_new: int) -> int:
        c_old = self.state[r]
        if c_new == c_old:
            return 0
        affected, d1_old, d2_old, d1_new, d2_new = self._affected_indices(r, c_old, c_new)
        pairs_before = 0
        for kind, idx in affected:
            if kind == "col":
                pairs_before += _def_pair(self.col[idx])
            elif kind == "d1":
                pairs_before += _def_pair(self.d1[idx])
            else:
                pairs_before += _def_pair(self.d2[idx])

        # simulare
        self.col[c_old] -= 1; self.d1[d1_old] -= 1; self.d2[d2_old] -= 1
        self.col[c_new] += 1; self.d1[d1_new] += 1; self.d2[d2_new] += 1
        pairs_after = 0
        for kind, idx in affected:
            if kind == "col":
                pairs_after += _def_pair(self.col[idx])
            elif kind == "d1":
                pairs_after += _def_pair(self.d1[idx])
            else:
                pairs_after += _def_pair(self.d2[idx])
        self.col[c_new] -= 1; self.d1[d1_new] -= 1; self.d2[d2_new] -= 1
        self.col[c_old] += 1; self.d1[d1_old] += 1; self.d2[d2_old] += 1
        return (pairs_after - pairs_before)

    def apply_move(self, r: int, c_new: int) -> None:
        c_old = self.state[r]
        if c_new == c_old:
            return
        affected, d1_old, d2_old, d1_new, d2_new = self._affected_indices(r, c_old, c_new)
        # scade vechi
        before = 0
        for kind, idx in affected:
            if kind == "col": before += _def_pair(self.col[idx])
            elif kind == "d1": before += _def_pair(self.d1[idx])
            else: before += _def_pair(self.d2[idx])
        # aplică
        self.col[c_old] -= 1; self.d1[d1_old] -= 1; self.d2[d2_old] -= 1
        self.col[c_new] += 1; self.d1[d1_new] += 1; self.d2[d2_new] += 1
        self.state[r] = c_new
        # adauga nou
        after = 0
        for kind, idx in affected:
            if kind == "col": after += _def_pair(self.col[idx])
            elif kind == "d1": after += _def_pair(self.d1[idx])
            else: after += _def_pair(self.d2[idx])
        self.total_pairs += (after - before)

# =============================
# Init & afișare board
# =============================

def random_state(n: int) -> List[int]:
    return [random.randrange(n) for _ in range(n)]


def pretty_board(state: List[int]) -> str:
    n = len(state)
    lines = []
    for r in range(n):
        row = ["·"] * n
        row[state[r]] = "Q"
        lines.append(" ".join(row))
    return "\n".join(lines)

# =============================
# Hill‑Climbing cu Restarts
# =============================

def hill_climb_with_restarts(n: int) -> RunMetrics:
    start_time = time.perf_counter()
    restarts = 0
    steps = 0
    sideways_left = SIDEWAYS_LIMIT if ALLOW_SIDEWAYS else 0

    # stare inițiala a trial‑ului
    cc = ConflictCounter(n, random_state(n))
    initial_conf0 = cc.total_conflicts()

    while True:
        current_conf = cc.total_conflicts()
        if current_conf == 0:
            end_time = time.perf_counter() - start_time
            return RunMetrics(True, steps, restarts, end_time, start_conflicts=initial_conf0, end_conflicts=0, solution=cc.state[:])

        if steps >= STEP_LIMIT:
            end_time = time.perf_counter() - start_time
            return RunMetrics(False, steps, restarts, end_time, start_conflicts=initial_conf0, end_conflicts=current_conf, solution=None)

        # cauta cel mai bun vecin (steepest descent)
        best_delta = 0
        best_moves: List[Tuple[int, int]] = []
        for r in range(n):
            c_old = cc.state[r]
            for c_new in range(n):
                if c_new == c_old: continue
                delta = cc.delta_if_move(r, c_new)
                if delta < best_delta:
                    best_delta = delta; best_moves = [(r, c_new)]
                elif delta == best_delta and delta != 0:
                    best_moves.append((r, c_new))

        if not best_moves:
            # sideways -> (delta==0)
            if ALLOW_SIDEWAYS and sideways_left > 0:
                sideways: List[Tuple[int, int]] = []
                for r in range(n):
                    c_old = cc.state[r]
                    for c_new in range(n):
                        if c_new == c_old: continue
                        if cc.delta_if_move(r, c_new) == 0:
                            sideways.append((r, c_new))
                if sideways:
                    r, c_new = random.choice(sideways)
                    cc.apply_move(r, c_new)
                    steps += 1
                    sideways_left -= 1
                    continue

            # restart
            restarts += 1
            if restarts > MAX_RESTARTS:
                end_time = time.perf_counter() - start_time
                return RunMetrics(False, steps, restarts, end_time, start_conflicts=initial_conf0, end_conflicts=cc.total_conflicts(), solution=None)
            sideways_left = SIDEWAYS_LIMIT if ALLOW_SIDEWAYS else 0
            cc = ConflictCounter(n, random_state(n))
            continue

        # aplica una dintre cele mai bune mutari
        r, c_new = random.choice(best_moves)
        cc.apply_move(r, c_new)
        steps += 1

# =============================
# Agregare & raportare
# =============================

def summarize(metrics_list: List[RunMetrics]) -> Summary:
    successes = sum(1 for m in metrics_list if m.solved)
    avg_steps = sum(m.steps for m in metrics_list) / len(metrics_list)
    avg_restarts = sum(m.restarts for m in metrics_list) / len(metrics_list)
    avg_time = sum(m.time_sec for m in metrics_list) / len(metrics_list)
    best_end = min(m.end_conflicts for m in metrics_list)
    worst_end = max(m.end_conflicts for m in metrics_list)
    example = next((m.solution for m in metrics_list if m.solved and m.solution is not None), None)
    return Summary(algo="HC_RR", n=N, trials=len(metrics_list), successes=successes,
                   avg_steps=avg_steps, avg_restarts=avg_restarts, avg_time_sec=avg_time,
                   best_end_conflicts=best_end, worst_end_conflicts=worst_end,
                   example_solution=example)


def print_summary(s: Summary) -> None:
    rate = 100.0 * s.successes / s.trials if s.trials else 0.0
    print(f"P3 — Local Search on N-Queens (N={s.n})")
    print(f"Algorithm: {s.algo}")
    print(f"Trials: {s.trials} | Successes: {s.successes} ({rate:.1f}%)")
    print(f"Avg steps: {s.avg_steps:.1f} | Avg restarts: {s.avg_restarts:.1f} | Avg time: {s.avg_time_sec*1000:.2f} ms")
    print(f"End conflicts (best/worst): {s.best_end_conflicts}/{s.worst_end_conflicts}")
    if s.example_solution is not None:
        print("\nExample solution board:")
        print(pretty_board(s.example_solution))


def _write_csv(path: str, header: list[str], rows: list[list]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# -----------------------------
# Multi‑N experiments & plotting
# -----------------------------

def run_trials_for_n(n: int, trials: int) -> dict:
    runs: List[RunMetrics] = []
    for _ in range(trials):
        runs.append(hill_climb_with_restarts(n))
    successes = sum(1 for m in runs if m.solved)
    return {
        "N": n,
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials if trials else 0.0,
        "avg_steps": sum(m.steps for m in runs) / trials,
        "avg_time_sec": sum(m.time_sec for m in runs) / trials,
    }


def plot_success_vs_n(data: list[dict], png_path: str) -> None:
    if not data:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[WARN] matplotlib indisponibil — nu pot genera graficul.")
        return
    Ns = [d["N"] for d in data]
    rates = [100.0 * d["success_rate"] for d in data]
    plt.figure(figsize=(6,4))
    plt.plot(Ns, rates, marker="o")
    plt.title("Rata de succes vs N (HC_RR)")
    plt.xlabel("N")
    plt.ylabel("Succes (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # --- Rulare de baza pe N curent ---
    runs: List[RunMetrics] = [hill_climb_with_restarts(N) for _ in range(TRIALS)]
    s = summarize(runs)
    print_summary(s)

    # Export per-trial
    if EXPORT_PER_TRIAL_CSV:
        header = ["trial","solved","steps","restarts","time_sec","start_conflicts","end_conflicts"]
        rows = [[i+1, m.solved, m.steps, m.restarts, round(m.time_sec,6), m.start_conflicts, m.end_conflicts] for i,m in enumerate(runs)]
        _write_csv(PER_TRIAL_CSV_PATH, header, rows)
        print(f"Saved per-trial CSV -> {PER_TRIAL_CSV_PATH}")

    if EXPORT_SUMMARY_CSV:
        header = ["algo","N","trials","successes","success_rate","avg_steps","avg_restarts","avg_time_sec","best_end_conflicts","worst_end_conflicts"]
        rate = s.successes / s.trials if s.trials else 0.0
        rows = [[s.algo, s.n, s.trials, s.successes, round(rate,4), round(s.avg_steps,3), round(s.avg_restarts,3), round(s.avg_time_sec,6), s.best_end_conflicts, s.worst_end_conflicts]]
        _write_csv(SUMMARY_CSV_PATH, header, rows)
        print(f"Saved summary CSV -> {SUMMARY_CSV_PATH}")

    # --- Experimente multi-N ---
    if RUN_MULTI_N and N_GRID:
        multi_data: list[dict] = []
        for n in N_GRID:
            md = run_trials_for_n(n, TRIALS_PER_N)
            multi_data.append(md)
            print(f"N={n}: success={md['successes']}/{md['trials']} (rate={md['success_rate']*100:.1f}%), avg_steps={md['avg_steps']:.1f}")

        if EXPORT_SUMMARY_CSV:
            header = ["N","trials","successes","success_rate","avg_steps","avg_time_sec"]
            rows = [[d["N"], d["trials"], d["successes"], round(d["success_rate"],4), round(d["avg_steps"],3), round(d["avg_time_sec"],6)] for d in multi_data]
            _write_csv("p3_multiN_summary.csv", header, rows)
            print("Saved multi-N CSV -> p3_multiN_summary.csv")

        if PLOT_SUCCESS_VS_N:
            plot_success_vs_n(multi_data, SUCCESS_PNG_PATH)
            print(f"Saved plot -> {SUCCESS_PNG_PATH}")

if __name__ == "__main__":
    main()
