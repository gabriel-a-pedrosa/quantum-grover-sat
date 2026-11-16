"""Grover for n x n Sudoku with any number of symbols per cell.

Each cell uses ceil(log2(values)) qubits and unused bit patterns are marked
as illegal. Cells are numbered row by row, cell = row * n + col.
"""

import math
from itertools import combinations

import pennylane as qml

from . import grover as g


def cell_bits(values):
    return max(1, math.ceil(math.log2(values)))


def distinct_pairs(n, box=None):
    """All cell pairs that have to be different (rows, columns, boxes)."""
    groups = [[r * n + c for c in range(n)] for r in range(n)]     # rows
    groups += [[r * n + c for r in range(n)] for c in range(n)]    # columns
    if box is not None:
        br, bc = box
        for r0 in range(0, n, br):
            for c0 in range(0, n, bc):
                groups.append([(r0 + i) * n + (c0 + j)
                               for i in range(br) for j in range(bc)])
    pairs = set()
    for grp in groups:
        for a, b in combinations(grp, 2):
            pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


# --- classical side ---------------------------------------------------------

def decode_board(index, n, values):
    """Turn a measurement index into the list of cell values."""
    bits = cell_bits(values)
    total = n * n * bits
    cells = []
    for cell in range(n * n):
        v = 0
        for b in range(bits):
            wire = cell * bits + b
            v = (v << 1) | ((index >> (total - 1 - wire)) & 1)
        cells.append(v)
    return cells


def board_grid(cells, n):
    """Reshape a flat list of cell values into n rows."""
    return [cells[r * n:(r + 1) * n] for r in range(n)]


def is_valid(cells, n, values, box=None, givens=None):
    if any(v >= values for v in cells):
        return False
    if givens:
        for cell, val in givens.items():
            if cells[cell] != val:
                return False
    for a, b in distinct_pairs(n, box):
        if cells[a] == cells[b]:
            return False
    return True


def sudoku_solutions(n, values=2, box=None, givens=None):
    total = n * n * cell_bits(values)
    return [i for i in range(2 ** total)
            if is_valid(decode_board(i, n, values), n, values, box, givens)]


# --- the quantum solver -----------------------------------------------------

def solve_sudoku(n, values=2, box=None, givens=None, iters=None,
                 shots=None, force=False, draw=False, device=None):
    bits = cell_bits(values)
    ncells = n * n
    n_board = ncells * bits

    pairs = distinct_pairs(n, box)
    forbidden = list(range(values, 2 ** bits))
    # one flag qubit per constraint: first the "must differ" pairs, then the
    # "must not be an illegal value" checks
    constraints = [("diff", p) for p in pairs]
    constraints += [("valid", (cell, code))
                    for cell in range(ncells) for code in forbidden]

    n_flags = len(constraints)
    n_scratch = bits if bits > 1 else 0

    board = list(range(n_board))
    flags = list(range(n_board, n_board + n_flags))
    scratch = list(range(n_board + n_flags, n_board + n_flags + n_scratch))
    anc = n_board + n_flags + n_scratch
    n_wires = anc + 1

    if n_wires > 26 and not force:
        raise ValueError("this board needs %d qubits, too many to simulate "
                         "(pass force=True to try anyway)" % n_wires)

    def cell_wires(cell):
        return list(range(cell * bits, cell * bits + bits))

    def compute_flags():
        for (kind, payload), flag in zip(constraints, flags):
            if kind == "diff":
                a, b = payload
                g.not_equal(cell_wires(a), cell_wires(b), flag, scratch)
            else:
                cell, code = payload
                code_bits = [(code >> (bits - 1 - i)) & 1 for i in range(bits)]
                qml.PauliX(flag)                          # assume valid...
                g.flip_if(cell_wires(cell), flag, code_bits)  # ...0 iff illegal

    def oracle():
        compute_flags()
        g.mark_and(flags, anc)     # every constraint satisfied
        compute_flags()            # uncompute the flags

    # known cells: fix their qubits, keep them out of the search/diffuser
    givens = givens or {}
    fixed = {}
    for cell, val in givens.items():
        for i, w in enumerate(cell_wires(cell)):
            fixed[w] = (val >> (bits - 1 - i)) & 1
    free = [w for w in board if w not in fixed]

    def prep():
        for w, bit in fixed.items():
            if bit:
                qml.PauliX(w)

    if iters is None:
        M = max(len(sudoku_solutions(n, values, box, givens)), 1)
        iters = g.n_iterations(2 ** len(free), M)

    return g.run_grover(oracle, free, anc, iters, n_wires,
                        measure=board, prep=prep, shots=shots, draw=draw,
                        device=device)
