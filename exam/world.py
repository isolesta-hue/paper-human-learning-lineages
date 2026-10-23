"""The exam world's rules (exam_spec/GENERATOR.md 2): a side view 5 x 5 grid with gravity; people stand on the floor and
move sideways; things are solid (one per cell), rest on the floor or on another thing, fall when their support goes; a push
moves the pushed thing and every thing touching it beyond by one cell, or nothing when the far end is blocked; boxes hide
what is put in them. The scene record is the visible state, one line per person or thing."""
import copy

W, H = 5, 5


class World:
    def __init__(self):
        self.pos = {}           # id -> (x, y) when visible
        self.inside = {}        # box id -> [ids] (hidden)
        self.box = set()        # ids that are boxes
        self.people = set()

    def copy(self):
        return copy.deepcopy(self)

    # ---- queries
    def at(self, x, y):
        for k, p in self.pos.items():
            if p == (x, y):
                return k
        return None

    def free(self, x, y):
        return 0 <= x < W and 0 <= y < H and self.at(x, y) is None

    def things(self):
        return [k for k in self.pos if k not in self.people]

    def top_of(self, k):
        """the highest thing in k's column above k, or k itself."""
        x, y = self.pos[k]
        while self.at(x, y + 1) is not None and self.at(x, y + 1) not in self.people:
            y += 1
        return self.at(x, y)

    def has_on_top(self, k):
        x, y = self.pos[k]
        return self.at(x, y + 1) is not None

    def scene(self, moment):
        lines = [{"id": k, "cells": [list(self.pos[k])]} for k in sorted(self.pos)]
        return {"moment": moment, "lines": lines}

    # ---- rules
    def settle(self):
        """gravity: every thing without support falls to the first supported cell (lowest first, until nothing moves)."""
        moved = True
        while moved:
            moved = False
            for k in sorted(self.things(), key=lambda k_: self.pos[k_][1]):
                x, y = self.pos[k]
                if y > 0 and self.at(x, y - 1) is None:
                    self.pos[k] = (x, y - 1)
                    moved = True

    def place(self, k, x, y, person=False, box=False):
        assert self.free(x, y), (k, x, y)
        self.pos[k] = (x, y)
        if person:
            self.people.add(k)
        if box:
            self.box.add(k)
            self.inside.setdefault(k, [])

    def go(self, c, t):
        """person c moves to the floor cell right beside thing t's column (the nearer free side)."""
        tx = self.pos[t][0]
        cx = self.pos[c][0]
        sides = sorted([tx - 1, tx + 1], key=lambda x: abs(x - cx))
        for x in sides:
            if 0 <= x < W and (self.at(x, 0) is None or self.at(x, 0) == c):
                self.pos[c] = (x, 0)
                return True
        return False

    def column(self, k):
        """k and every thing stacked on it, bottom first."""
        x, y = self.pos[k]
        out = [k]
        while self.at(x, y + 1) is not None and self.at(x, y + 1) not in self.people:
            y += 1
            out.append(self.at(x, y))
        return out

    def push(self, c, t):
        """person c (on the floor right beside t, t on the floor) pushes t away from itself: t and every thing touching it
        in that direction move one cell -- each with what is stacked on it -- if the floor cell after the last one is free;
        otherwise nothing moves (the wall or a person blocks: solidity)."""
        cx, cy = self.pos[c]
        tx, ty = self.pos[t]
        if ty != 0 or cy != 0 or abs(cx - tx) != 1:
            return False
        d = 1 if tx > cx else -1
        row = [t]
        x = tx + d
        while 0 <= x < W and self.at(x, 0) is not None:
            if self.at(x, 0) in self.people:
                return False
            row.append(self.at(x, 0))
            x += d
        if not (0 <= x < W):
            return False                            # blocked: the wall (solid)
        cols = [self.column(k) for k in row]
        for col in reversed(cols):
            for k in col:
                px, py = self.pos[k]
                self.pos[k] = (px + d, py)
        self.settle()
        return True

    def put_on(self, c, t, base):
        """t is put on top of base's column; whatever stood on t stays and falls to its next support (gravity)."""
        if t == base or t in self.people or t not in self.pos or base not in self.pos or t in self.column(base):
            return False
        top = self.top_of(base)
        x, y = self.pos[top]
        if y + 1 >= H or (self.at(x, y + 1) is not None):
            return False
        old = self.pos.pop(t)
        if self.at(x, y + 1) is not None:
            self.pos[t] = old
            return False
        self.pos[t] = (x, y + 1)
        self.settle()
        return True

    def put_in(self, c, t, b):
        """t goes into box b and is hidden; whatever stood on t falls to its next support (gravity)."""
        if b not in self.box or b not in self.pos or t == b or t in self.people or t not in self.pos or t in self.box:
            return False
        self.pos.pop(t)
        self.inside[b].append(t)
        self.settle()
        return True

    def take_out(self, c, t, b):
        """t comes out of box b and is set right beside the box (the side nearer the person; on top of what already
        stands there if the floor cell is taken by a thing)."""
        if b not in self.box or t not in self.inside.get(b, []):
            return False
        bx = self.pos[b][0]
        cx = self.pos[c][0] if c in self.pos else bx
        for x in sorted([bx - 1, bx + 1], key=lambda x_: abs(x_ - cx)):
            if not (0 <= x < W):
                continue
            y = 0
            while self.at(x, y) is not None and self.at(x, y) not in self.people and y < H - 1:
                y += 1
            if self.free(x, y):
                self.inside[b].remove(t)
                self.pos[t] = (x, y)
                return True
        return False

    def relation(self, a, b):
        """the spatial word for a with respect to b (closed-v1): on / under / in / beside, or None."""
        if b in self.box and a in self.inside.get(b, []):
            return "in"
        if a not in self.pos or b not in self.pos:
            return None
        (ax, ay), (bx, by) = self.pos[a], self.pos[b]
        if ax == bx and ay == by + 1:
            return "on"
        if ax == bx and ay == by - 1:
            return "under"
        if ay == by and abs(ax - bx) == 1:
            return "beside"
        return None
