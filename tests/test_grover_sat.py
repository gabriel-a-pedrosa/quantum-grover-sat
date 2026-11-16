# A few sanity checks: the quantum runs should amplify exactly the states
# that the (classical) brute-force search says are solutions.

from grover_sat import (
    solve_cnf, solve_dnf, cnf_solutions, dnf_solutions,
    solve_sudoku, sudoku_solutions, decode_board, board_grid,
)
from grover_sat.grover import top_states, n_iterations


def amplified(probs, n_bits):
    return {label for label, _ in top_states(probs, n_bits)}


def test_n_iterations():
    assert n_iterations(16, 1) == 3      # one solution out of 16
    assert n_iterations(8, 0) == 1       # degenerate cases clamp to 1
    assert n_iterations(8, 8) == 1


def test_cnf_matches_bruteforce():
    clauses = [[1, 2, -3], [-1, -2, 3], [1, 2, 3]]
    # 5 solutions out of 8, so M > N/2 and the closed-form estimate is poor;
    # 2 iterations work here (same situation as the report's formula).
    probs = solve_cnf(clauses, n_vars=3, iters=2)
    expected = {format(i, "03b") for i in cnf_solutions(clauses, 3)}
    assert amplified(probs, 3) == expected


def test_cnf_single_solution():
    clauses = [[1], [2], [3]]            # only 111 works
    assert cnf_solutions(clauses, 3) == [0b111]
    probs = solve_cnf(clauses, 3)
    assert top_states(probs, 3)[0][0] == "111"
    assert probs[0b111] > 0.9


def test_dnf_original_formula():
    # F = (A & ~B) | (A & C) | (~A & B & ~C) | (C & ~B)
    F = [[1, -2], [1, 3], [-1, 2, -3], [3, -2]]
    assert len(dnf_solutions(F, 3)) == 5
    probs = solve_dnf(F, 3, iters=2)     # M > N/2: 2 iterations (see report)
    expected = {format(i, "03b") for i in dnf_solutions(F, 3)}
    assert amplified(probs, 3) == expected


def test_binary_2x2_sudoku():
    probs = solve_sudoku(2, values=2)
    boards = {tuple(map(tuple, board_grid(decode_board(int(lbl, 2), 2, 2), 2)))
              for lbl, _ in top_states(probs, 4)}
    assert boards == {((0, 1), (1, 0)), ((1, 0), (0, 1))}


def test_sudoku_with_known_cell():
    probs = solve_sudoku(2, values=2, givens={0: 0})
    assert sudoku_solutions(2, 2, givens={0: 0}) == [6]   # only 0110 fits
    best = top_states(probs, 4)[0]
    assert best[0] == "0110" and best[1] > 0.9


def test_three_symbol_cells_skip_illegal_values():
    # 3 symbols need 2 qubits/cell; code '11' (=3) must never show up
    probs = solve_sudoku(2, values=3)
    amplified_boards = [decode_board(int(lbl, 2), 2, 3) for lbl, _ in top_states(probs, 8)]
    expected = [decode_board(i, 2, 3) for i in sudoku_solutions(2, 3)]
    assert sorted(amplified_boards) == sorted(expected)
