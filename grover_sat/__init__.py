"""Grover's algorithm for boolean SAT and Sudoku."""

from .grover import (
    n_iterations,
    diffuser,
    run_grover,
    plot_probs,
    top_states,
)
from .sat import (
    solve_cnf,
    solve_dnf,
    cnf_solutions,
    dnf_solutions,
)
from .sudoku import (
    solve_sudoku,
    sudoku_solutions,
    decode_board,
    board_grid,
    distinct_pairs,
)

__version__ = "1.0"
