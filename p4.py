"""
P4 — Adversarial Search (AIMA ~p.161): X&0 cu Alpha‑Beta (Minimax cu pruning)

- Alpha‑Beta pruning (aceleași decizii ca Minimax, dar mai eficient)

- limita de adancime + euristica pentru stari neterminale
- Metrici per decizie: noduri extinse, adancime atinsa, timp, valoarea aleasa

- 2 moduri "self_play" (X vs O, ambele AI Alpha‑Beta) sau "human_vs_ai" (input din terminal)

- Export CSV cu pașii jocului și sumar pe joc
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# =============================
# CONFIG
# =============================
BOARD_SIZE: int = 3
PLAY_MODE: str = "self_play"             # "self_play" sau "human_vs_ai"
DEPTH_LIMIT: Optional[int] = None        # None = cautare completa; sau limita
USE_HEURISTIC: bool = True               # daca DEPTH_LIMIT este setat, folosește evaluare euristica
MOVE_ORDERING: bool = True               # ordonare mutari dupa scor euristic (ajuta alpha‑beta)

# Experimente multiple (doar pentru self_play)
GAMES: int = 3                           # cate jocuri / experimente să ruleze

# Export CSV
EXPORT_CSV: bool = True
CSV_MOVES_PATH: str = "p4_moves.csv"
CSV_SUMMARY_PATH: str = "p4_summary.csv"

# =============================
# Init & helpere
# =============================
Cell = int  # 0..8 pentru o tablă 3x3 stocata liniar
Board = Tuple[str, ...]

MAX_PLAYER = 'X'
MIN_PLAYER = 'O'
EMPTY = ' '

WIN_LINES: List[Tuple[Cell, Cell, Cell]] = [
    (0,1,2),(3,4,5),(6,7,8),  # rânduri
    (0,3,6),(1,4,7),(2,5,8),  # coloane
    (0,4,8),(2,4,6)           # diagonale
]

@dataclass
class DecisionStats:
    nodes_expanded: int = 0
    max_depth: int = 0
    time_sec: float = 0.0
    chosen_value: int = 0

@dataclass
class GameStats:
    game_id: int
    winner: str
    moves: int
    total_time: float
    total_nodes: int

def initial_board() -> Board:
    return tuple([EMPTY] * (BOARD_SIZE * BOARD_SIZE))


def print_board(b: Board) -> None:
    n = BOARD_SIZE
    for r in range(n):
        row = [b[r*n + c] for c in range(n)]
        print(' | '.join(x if x != EMPTY else '·' for x in row))
        if r < n-1:
            print('-' * (n*4 - 3))


def actions(b: Board) -> List[Cell]:
    return [i for i, v in enumerate(b) if v == EMPTY]


def result(b: Board, move: Cell, player: str) -> Board:
    lst = list(b)
    lst[move] = player
    return tuple(lst)


def winner(b: Board) -> Optional[str]:
    for a, c, d in WIN_LINES:
        if b[a] != EMPTY and b[a] == b[c] == b[d]:
            return b[a]
    return None


def terminal_test(b: Board) -> bool:
    return winner(b) is not None or all(v != EMPTY for v in b)


def utility(b: Board) -> int:
    w = winner(b)
    if w == MAX_PLAYER:
        return 1
    elif w == MIN_PLAYER:
        return -1
    else:
        return 0

# =============================
# Euristica pentru stari neterminale (linii necontestate)
# =============================

def heuristic(b: Board) -> int:
    # Scor pe linii: +1/+10 pentru 1/2 X pe linie fara O
    # simetric negativ pentru O
    score = 0
    for a, c, d in WIN_LINES:
        line = (b[a], b[c], b[d])
        x = line.count('X')
        o = line.count('O')
        if x > 0 and o == 0:
            score += (1 if x == 1 else 10)
        elif o > 0 and x == 0:
            score -= (1 if o == 1 else 10)
    return score

# =============================
# Ordonare mutari
# =============================

def ordered_actions(b: Board, player: str) -> List[Cell]:
    acts = actions(b)
    if not MOVE_ORDERING or DEPTH_LIMIT is None:
        return acts
    # Heuristic lookahead
    scored = []
    for m in acts:
        nb = result(b, m, player)
        val = heuristic(nb)
        scored.append((val if player == MAX_PLAYER else -val, m))
    scored.sort(reverse=True)
    return [m for _, m in scored]

# =============================
# Alpha‑Beta
# =============================

def alphabeta_decision(b: Board, player: str) -> Tuple[Cell, DecisionStats]:
    stats = DecisionStats()
    start = time.perf_counter()

    def max_value(s: Board, alpha: int, beta: int, depth: int) -> int:
        stats.nodes_expanded += 1
        stats.max_depth = max(stats.max_depth, depth)
        if terminal_test(s):
            return utility(s)
        if DEPTH_LIMIT is not None and depth >= DEPTH_LIMIT:
            return heuristic(s) if USE_HEURISTIC else 0
        v = -10**9
        for m in ordered_actions(s, MAX_PLAYER):
            v = max(v, min_value(result(s, m, MAX_PLAYER), alpha, beta, depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(s: Board, alpha: int, beta: int, depth: int) -> int:
        stats.nodes_expanded += 1
        stats.max_depth = max(stats.max_depth, depth)
        if terminal_test(s):
            return utility(s)
        if DEPTH_LIMIT is not None and depth >= DEPTH_LIMIT:
            return heuristic(s) if USE_HEURISTIC else 0
        v = 10**9
        for m in ordered_actions(s, MIN_PLAYER):
            v = min(v, max_value(result(s, m, MIN_PLAYER), alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    best_move: Optional[Cell] = None
    if player == MAX_PLAYER:
        best_val = -10**9
        alpha, beta = -10**9, 10**9
        for m in ordered_actions(b, MAX_PLAYER):
            v = min_value(result(b, m, MAX_PLAYER), alpha, beta, 1)
            if v > best_val:
                best_val, best_move = v, m
            alpha = max(alpha, best_val)
    else:
        best_val = 10**9
        alpha, beta = -10**9, 10**9
        for m in ordered_actions(b, MIN_PLAYER):
            v = max_value(result(b, m, MIN_PLAYER), alpha, beta, 1)
            if v < best_val:
                best_val, best_move = v, m
            beta = min(beta, best_val)

    stats.time_sec = time.perf_counter() - start
    stats.chosen_value = best_val
    return best_move if best_move is not None else actions(b)[0], stats



def decide(b: Board, player: str) -> Tuple[Cell, DecisionStats]:
    return alphabeta_decision(b, player)


def self_play_game(game_id: int = 1) -> Tuple[Board, str, List[Dict[str, Any]]]:
    b = initial_board()
    player = MAX_PLAYER
    move_log: List[Dict[str, Any]] = []
    total_nodes = 0
    total_time = 0.0

    while not terminal_test(b):
        move, stats = decide(b, player)
        b = result(b, move, player)
        total_nodes += stats.nodes_expanded
        total_time += stats.time_sec
        move_log.append({
            'game_id': game_id,
            'move_idx': len(move_log)+1,
            'player': player,
            'cell': move,
            'value': stats.chosen_value,
            'nodes': stats.nodes_expanded,
            'depth': stats.max_depth,
            'time_sec': round(stats.time_sec, 6)
        })
        player = MIN_PLAYER if player == MAX_PLAYER else MAX_PLAYER

    w = winner(b)
    return b, (w if w is not None else 'DRAW'), move_log


def human_vs_ai() -> None:
    b = initial_board()
    human = MIN_PLAYER  # umanul joacă implicit cu O (al doilea)
    ai = MAX_PLAYER
    print("Tu ești O. AI este X. Joacă introducând un index 0..8 (stânga‑dreapta, sus‑jos).")
    print_board(b)
    while not terminal_test(b):
        # Mutarea AI
        move, _ = decide(b, ai)
        b = result(b, move, ai)
        print("\nAI (X) a jucat la", move)
        print_board(b)
        if terminal_test(b):
            break
        # Mutarea jucatorului (input terminal))
        while True:
            try:
                m = int(input("Mutarea ta (0..8): "))
                if m in actions(b):
                    b = result(b, m, human)
                    break
            except Exception:
                pass
            print("Mutare invalidă, încearcă din nou…")
        print_board(b)
    w = winner(b)
    print("\nRezultat:", (w if w else "REMIZĂ"))

def _write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    import csv
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def main() -> None:
    if PLAY_MODE == 'human_vs_ai':
        human_vs_ai()
        return

    # self‑play
    all_moves: List[Dict[str, Any]] = []
    summaries: List[GameStats] = []
    for gid in range(1, GAMES+1):
        t0 = time.perf_counter()
        board, w, moves = self_play_game(gid)
        total_time = time.perf_counter() - t0
        total_nodes = sum(m['nodes'] for m in moves)
        summaries.append(GameStats(game_id=gid, winner=w, moves=len(moves), total_time=total_time, total_nodes=total_nodes))
        all_moves.extend(moves)

        print(f"\n=== GAME {gid} ===")
        print_board(board)
        print("Winner:", w)
        print(f"Moves: {len(moves)} | Total time: {total_time*1000:.2f} ms | Nodes expanded: {total_nodes}")

    if EXPORT_CSV:
        header = ['game_id','move_idx','player','cell','value','nodes','depth','time_sec']
        rows = [[m[k] for k in header] for m in all_moves]
        _write_csv(CSV_MOVES_PATH, header, rows)
        print(f"Saved moves CSV -> {CSV_MOVES_PATH}")


        header2 = ['game_id','winner','moves','total_time_sec','total_nodes']
        rows2 = [[s.game_id, s.winner, s.moves, round(s.total_time,6), s.total_nodes] for s in summaries]
        _write_csv(CSV_SUMMARY_PATH, header2, rows2)
        print(f"Saved summary CSV -> {CSV_SUMMARY_PATH}")

if __name__ == '__main__':
    main()
