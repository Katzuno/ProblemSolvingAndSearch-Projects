"""
P1 & P2 — Search on the Romania map

P1: Uninformed Search — compare BFS vs UCS
P2: Informed Search — A*
    compare performance/solution vs P1 algorithms on the same problem.

"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import heapq

# -----------------------------
# CONFIGURATION CONSTANTS
# -----------------------------
START_CITY = "Arad"
GOAL_CITY = "Bucharest"
HEURISTIC_TARGET = "Bucharest"
RUN_P1 = ("bfs", "ucs")          # p1 -two uninformed algorithms
INFORMED_ALGO = "astar"            # p2 - one informed algorithm

# -----------------------------
# Problem & Node classes
# -----------------------------
class Problem:
    def __init__(self, initial: Any, goal: Any):
        self.initial = initial
        self.goal = goal

    def actions(self, state: Any) -> Iterable[Any]:
        raise NotImplementedError

    def result(self, state: Any, action: Any) -> Any:
        raise NotImplementedError

    def step_cost(self, state: Any, action: Any, next_state: Any) -> float:
        return 1.0

    def goal_test(self, state: Any) -> bool:
        return state == self.goal

@dataclass(order=True)
class Node:
    priority: float
    state: Any = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)
    action: Optional[Any] = field(default=None, compare=False)
    path_cost: float = field(default=0.0, compare=False)
    depth: int = field(default=0, compare=False)

    @staticmethod
    def root(state: Any) -> 'Node':
        return Node(priority=0.0, state=state, parent=None, action=None, path_cost=0.0, depth=0)

    def child(self, problem: 'Problem', action: Any, next_state: Any, step_cost: float, priority: float) -> 'Node':
        return Node(priority=priority, state=next_state, parent=self, action=action, path_cost=self.path_cost + step_cost, depth=self.depth + 1)

    def solution(self) -> List[Any]:
        node, actions = self, []
        while node and node.action is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))

    def path(self) -> List[Any]:
        node, states = self, []
        while node:
            states.append(node.state)
            node = node.parent
        return list(reversed(states))

# -----------------------------
# Romania map graph
# -----------------------------
ROMANIA_EDGES: Dict[str, Dict[str, int]] = {
    'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
    'Zerind': {'Arad': 75, 'Oradea': 71},
    'Oradea': {'Zerind': 71, 'Sibiu': 151},
    'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
    'Timisoara': {'Arad': 118, 'Lugoj': 111},
    'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
    'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
    'Drobeta': {'Mehadia': 75, 'Craiova': 120},
    'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
    'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
    'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
    'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
    'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
    'Giurgiu': {'Bucharest': 90},
    'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
    'Hirsova': {'Urziceni': 98, 'Eforie': 86},
    'Eforie': {'Hirsova': 86},
    'Vaslui': {'Urziceni': 142, 'Iasi': 92},
    'Iasi': {'Vaslui': 92, 'Neamt': 87},
    'Neamt': {'Iasi': 87},
}

SLD_TO_BUCHAREST: Dict[str, int] = {
    'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242, 'Eforie': 161,
    'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151, 'Iasi': 226, 'Lugoj': 244,
    'Mehadia': 241, 'Neamt': 234, 'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193,
    'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80, 'Vaslui': 199, 'Zerind': 374
}

class RomaniaMap(Problem):
    def __init__(self, initial: str, goal: str):
        super().__init__(initial=initial, goal=goal)
        self.graph = ROMANIA_EDGES

    def actions(self, state: str) -> Iterable[str]:
        return self.graph[state].keys()

    def result(self, state: str, action: str) -> str:
        return action

    def step_cost(self, state: str, action: str, next_state: str) -> float:
        return float(self.graph[state][next_state])

# -----------------------------
# Heuristic(s)
# -----------------------------
def sld_to(goal: str) -> Callable[[str], float]:
    if goal == 'Bucharest':
        table = SLD_TO_BUCHAREST
        return lambda s: float(table.get(s, 0.0))
    else:
        return lambda s: 0.0

# -----------------------------
# Metrics helper
# -----------------------------
@dataclass
class Metrics:
    expanded: int = 0
    frontier_peek: int = 0
    runtime_sec: float = 0.0

# -----------------------------
# Uninformed search algorithms
# -----------------------------

def graph_bfs(problem: Problem) -> Tuple[Optional[Node], Metrics]:
    start = time.perf_counter()
    metrics = Metrics()
    node = Node.root(problem.initial)
    if problem.goal_test(node.state):
        metrics.runtime_sec = time.perf_counter() - start
        return node, metrics

    frontier: deque[Node] = deque([node])
    frontier_states: Set[Any] = {node.state}
    explored: Set[Any] = set()

    while frontier:
        metrics.frontier_peek = max(metrics.frontier_peek, len(frontier))
        node = frontier.popleft()
        frontier_states.discard(node.state)
        explored.add(node.state)
        metrics.expanded += 1

        for a in problem.actions(node.state):
            child_state = problem.result(node.state, a)
            if child_state not in explored and child_state not in frontier_states:
                child = node.child(problem, a, child_state, problem.step_cost(node.state, a, child_state), priority=0.0)
                if problem.goal_test(child.state):
                    metrics.runtime_sec = time.perf_counter() - start
                    return child, metrics
                frontier.append(child)
                frontier_states.add(child.state)
    metrics.runtime_sec = time.perf_counter() - start
    return None, metrics


def graph_ucs(problem: Problem) -> Tuple[Optional[Node], Metrics]:
    start = time.perf_counter()
    metrics = Metrics()

    node = Node.root(problem.initial)
    frontier: List[Tuple[float, int, Node]] = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, node))
    frontier_costs: Dict[Any, float] = {node.state: 0.0}
    explored: Set[Any] = set()

    while frontier:
        metrics.frontier_peek = max(metrics.frontier_peek, len(frontier))
        _, _, node = heapq.heappop(frontier)
        if node.state in explored:
            continue
        if problem.goal_test(node.state):
            metrics.runtime_sec = time.perf_counter() - start
            return node, metrics
        explored.add(node.state)
        metrics.expanded += 1

        for a in problem.actions(node.state):
            s2 = problem.result(node.state, a)
            new_cost = node.path_cost + problem.step_cost(node.state, a, s2)
            if s2 not in explored and (s2 not in frontier_costs or new_cost < frontier_costs[s2]):
                counter += 1
                child = node.child(problem, a, s2, new_cost - node.path_cost, priority=new_cost)
                frontier_costs[s2] = new_cost
                heapq.heappush(frontier, (new_cost, counter, child))
    metrics.runtime_sec = time.perf_counter() - start
    return None, metrics

# -----------------------------
# Informed search algorithms
# -----------------------------

def best_first_search(problem: Problem, f: Callable[[Node], float]) -> Tuple[Optional[Node], Metrics]:
    start = time.perf_counter()
    metrics = Metrics()
    node = Node.root(problem.initial)
    frontier: List[Tuple[float, int, Node]] = []
    counter = 0
    heapq.heappush(frontier, (f(node), counter, node))
    best_f: Dict[Any, float] = {node.state: f(node)}
    explored: Set[Any] = set()

    while frontier:
        metrics.frontier_peek = max(metrics.frontier_peek, len(frontier))
        _, _, node = heapq.heappop(frontier)
        if node.state in explored:
            continue
        if problem.goal_test(node.state):
            metrics.runtime_sec = time.perf_counter() - start
            return node, metrics
        explored.add(node.state)
        metrics.expanded += 1
        for a in problem.actions(node.state):
            s2 = problem.result(node.state, a)
            step = problem.step_cost(node.state, a, s2)
            child = node.child(problem, a, s2, step, priority=0.0)
            val = f(child)
            if s2 not in explored and (s2 not in best_f or val < best_f[s2]):
                counter += 1
                best_f[s2] = val
                heapq.heappush(frontier, (val, counter, child))
    metrics.runtime_sec = time.perf_counter() - start
    return None, metrics


def a_star(problem: Problem, h: Callable[[str], float]) -> Tuple[Optional[Node], Metrics]:
    return best_first_search(problem, f=lambda n: n.path_cost + h(n.state))


AlgorithmFn = Callable[[Problem], Tuple[Optional[Node], Metrics]]


def run_algo(name: str, problem: Problem, hgoal: Optional[str] = None) -> Tuple[str, Optional[Node], Metrics]:
    name = name.lower()
    if name == 'bfs':
        node, metrics = graph_bfs(problem)
    elif name == 'ucs':
        node, metrics = graph_ucs(problem)
    elif name == 'astar':
        h = sld_to(hgoal or problem.goal)
        node, metrics = a_star(problem, h)
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    return name, node, metrics


def summarize(name: str, node: Optional[Node], metrics: Metrics) -> str:
    if node is None:
        return f"{name.upper():<7} -> FAIL | expanded={metrics.expanded} frontier_peak={metrics.frontier_peek} time={metrics.runtime_sec:.4f}s"
    path_str = ' -> '.join(node.path())
    return (
        f"{name.upper():<7} -> cost={node.path_cost:.0f} | steps={len(node.solution())} | "
        f"expanded={metrics.expanded} | frontier_peak={metrics.frontier_peek} | time={metrics.runtime_sec:.4f}s\n"
        f"           path: {path_str}"
    )

# -----------------------------
# Main runner
# -----------------------------

def run_all():
    problem = RomaniaMap(START_CITY, GOAL_CITY)
    print(f"Problem: from {START_CITY} to {GOAL_CITY}")
    print(f"Heuristic target: {HEURISTIC_TARGET}")

    # P1 comparison (two uninformed) = breadth-first search / uniform-cost search
    print("\n== P1 (UNINFORMED) Compare:", ', '.join(a.upper() for a in RUN_P1))
    for algo in RUN_P1:
        name, node, metrics = run_algo(algo, problem, hgoal=HEURISTIC_TARGET)
        print(summarize(name, node, metrics))

    # P2 — one informed algorithm = a*
    print("\n== P2 (INFORMED) Algorithm:", INFORMED_ALGO.upper())
    name, node, metrics = run_algo(INFORMED_ALGO, problem, hgoal=HEURISTIC_TARGET)
    print(summarize(name, node, metrics))

if __name__ == '__main__':
    run_all()
