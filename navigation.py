"""Range-only occupancy mapping and replanning; no obstacle coordinates required."""

from __future__ import annotations

import heapq
import math

import numpy as np


class RangeNavigator:
    """An inflated 2D occupancy grid with 8-connected A* and local path following.

    Inputs are robot XY, a goal, and measured laser hit points. Unknown space is
    explored as free; new range measurements trigger replanning at 5 Hz. The
    fixed operating bounds prevent unlimited detours around an impassable wall.
    """

    def __init__(self, goal=(9.5, 0.0), bounds=(-1.5, 11.0, -3.0, 3.0),
                 resolution=0.10, inflation=0.60, speed=0.20):
        self.goal = np.asarray(goal, dtype=float)
        self.bounds = tuple(bounds)
        self.resolution = float(resolution)
        self.inflation = float(inflation)
        self.speed = float(speed)
        self.shape = (int(round((bounds[1] - bounds[0]) / resolution)) + 1,
                      int(round((bounds[3] - bounds[2]) / resolution)) + 1)
        self.hits = np.zeros(self.shape, dtype=bool)
        self.blocked = self.hits.copy()
        self.caution = self.hits.copy()
        self.path = []
        self.mode = "SCANNING"
        self.replans = 0
        self.route_changes = 0
        self.seen_hit_count = 0
        self.goal_reached = False
        self._last_side = 0
        cells = math.ceil(inflation / resolution)
        self._inflation_offsets = [(dx, dy) for dx in range(-cells, cells + 1)
                                   for dy in range(-cells, cells + 1)
                                   if math.hypot(dx, dy) * resolution <= inflation + 1e-9]
        caution_cells = math.ceil((inflation + 0.25) / resolution)
        self._caution_offsets = [(dx, dy) for dx in range(-caution_cells, caution_cells + 1)
                                 for dy in range(-caution_cells, caution_cells + 1)
                                 if math.hypot(dx, dy) * resolution <= inflation + 0.25]

    def cell(self, point):
        return (int(round((point[0] - self.bounds[0]) / self.resolution)),
                int(round((point[1] - self.bounds[2]) / self.resolution)))

    def point(self, cell):
        return np.array([self.bounds[0] + cell[0] * self.resolution,
                         self.bounds[2] + cell[1] * self.resolution])

    def _inside(self, cell):
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def observe(self, hit_points):
        """Accumulate only observed surfaces, never a simulator obstacle AABB."""
        for point in hit_points:
            cell = self.cell(point)
            if self._inside(cell):
                self.hits[cell] = True
        self.seen_hit_count = int(self.hits.sum())
        self.blocked.fill(False)
        self.caution.fill(False)
        nx, ny = self.shape
        for grid, offsets in ((self.blocked, self._inflation_offsets), (self.caution, self._caution_offsets)):
            for dx, dy in offsets:
                sx0, sx1 = max(0, -dx), min(nx, nx - dx)
                sy0, sy1 = max(0, -dy), min(ny, ny - dy)
                grid[sx0 + dx:sx1 + dx, sy0 + dy:sy1 + dy] |= self.hits[sx0:sx1, sy0:sy1]

    def _free(self, cell):
        return self._inside(cell) and not self.blocked[cell]

    def segment_free(self, first, second):
        count = max(2, math.ceil(np.linalg.norm(np.asarray(second) - first) / (self.resolution * 0.4)))
        return all(self._free(self.cell(point)) for point in np.linspace(first, second, count))

    def plan(self, position):
        if self.goal_reached:
            self.mode = "GOAL_REACHED"
            return
        self.replans += 1
        start, goal = self.cell(position), self.cell(self.goal)
        if not self._free(start) or not self._free(goal):
            self.path = []
            self.mode = "STOP_BLOCKED"
            return
        queue = [(0.0, start)]
        costs = {start: 0.0}
        previous = {}
        visited = set()
        moves = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
        while queue:
            _, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                cells = [goal]
                while cells[-1] != start:
                    cells.append(previous[cells[-1]])
                self.path = [self.point(cell) for cell in reversed(cells)]
                self.mode = "AVOIDING" if any(abs(point[1]) > 0.30 for point in self.path[:20]) else "FORWARD"
                return
            for dx, dy in moves:
                nxt = current[0] + dx, current[1] + dy
                if not self._free(nxt):
                    continue
                # Diagonal edges must not squeeze between blocked grid corners.
                if dx and dy and (not self._free((current[0] + dx, current[1]))
                                  or not self._free((current[0], current[1] + dy))):
                    continue
                cost = costs[current] + math.hypot(dx, dy) + (0.6 if self.caution[nxt] else 0.0)
                if cost >= costs.get(nxt, math.inf):
                    continue
                previous[nxt] = current
                costs[nxt] = cost
                x, y = abs(nxt[0] - goal[0]), abs(nxt[1] - goal[1])
                heuristic = max(x, y) + (math.sqrt(2.0) - 1.0) * min(x, y)
                heapq.heappush(queue, (cost + heuristic, nxt))
        self.path = []
        self.mode = "STOP_BLOCKED"

    def command(self, position):
        position = np.asarray(position, dtype=float)
        distance = float(np.linalg.norm(self.goal - position))
        if self.goal_reached or distance < 0.18:
            self.goal_reached = True
            self.mode = "GOAL_REACHED"
            return np.zeros(2)
        if not self.path:
            self.mode = "STOP_BLOCKED"
            return np.zeros(2)
        # Follow the farthest collision-free point within a short lookahead.
        target = self.path[min(1, len(self.path) - 1)]
        for point in self.path[1:]:
            if np.linalg.norm(point - position) > 0.50:
                break
            if not self.segment_free(position, point):
                break
            target = point
        delta = target - position
        length = float(np.linalg.norm(delta))
        if length < 1e-6 or not self.segment_free(position, target):
            self.mode = "STOP_BLOCKED"
            return np.zeros(2)
        side = int(np.sign(delta[1])) if abs(delta[1]) > 0.15 else 0
        if side and side != self._last_side:
            self.route_changes += 1
            self._last_side = side
        return delta / length * min(self.speed, 0.8 * distance)
