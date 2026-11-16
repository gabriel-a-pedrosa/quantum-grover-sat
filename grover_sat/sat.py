"""Grover for boolean SAT, with formulas in CNF or DNF.

Literals are DIMACS-style, v means variable v-1 is True and -v means it is
False. Wires are the variables, then one work qubit per clause, then the ancilla.
"""

import pennylane as qml

from . import grover as g


# --- classical side (to count solutions and to double check the quantum run) -

def bits_of(index, n_vars):
    # variable j lives on wire j; with big-endian probs that is bit (n-1-j)
    return [(index >> (n_vars - 1 - j)) & 1 for j in range(n_vars)]


def _literal_true(lit, bits):
    v = bits[abs(lit) - 1]
    return v == 1 if lit > 0 else v == 0


def cnf_true(clauses, bits):
    return all(any(_literal_true(l, bits) for l in clause) for clause in clauses)


def dnf_true(terms, bits):
    return any(all(_literal_true(l, bits) for l in term) for term in terms)


def cnf_solutions(clauses, n_vars):
    return [i for i in range(2 ** n_vars) if cnf_true(clauses, bits_of(i, n_vars))]


def dnf_solutions(terms, n_vars):
    return [i for i in range(2 ** n_vars) if dnf_true(terms, bits_of(i, n_vars))]


# --- writing each clause / term onto a work qubit ---------------------------

def _term_onto(term, work, var_wires):
    # work = 1 iff the term (AND of literals) is True.
    # a literal is True when its variable is 1 (positive) or 0 (negative)
    controls = [var_wires[abs(l) - 1] for l in term]
    values = [1 if l > 0 else 0 for l in term]
    g.flip_if(controls, work, values)


def _clause_onto(clause, work, var_wires):
    # work = 1 iff the clause (OR of literals) is satisfied.
    # the clause only fails when every literal is False, so start at 1 and
    # flip back to 0 in exactly that case
    controls = [var_wires[abs(l) - 1] for l in clause]
    values = [0 if l > 0 else 1 for l in clause]
    qml.PauliX(work)               # assume satisfied...
    g.flip_if(controls, work, values)  # ...undo it iff every literal is False


# --- the quantum solvers ----------------------------------------------------

def solve_cnf(clauses, n_vars, iters=None, shots=None, draw=False, device=None):
    work = list(range(n_vars, n_vars + len(clauses)))
    var = list(range(n_vars))
    anc = n_vars + len(clauses)

    def oracle():
        for clause, w in zip(clauses, work):
            _clause_onto(clause, w, var)
        g.mark_and(work, anc)              # all clauses satisfied
        for clause, w in zip(clauses, work):
            _clause_onto(clause, w, var)   # uncompute the work qubits

    if iters is None:
        iters = g.n_iterations(2 ** n_vars, max(len(cnf_solutions(clauses, n_vars)), 1))
    return g.run_grover(oracle, var, anc, iters, anc + 1,
                        shots=shots, draw=draw, device=device)


def solve_dnf(terms, n_vars, iters=None, shots=None, draw=False, device=None):
    work = list(range(n_vars, n_vars + len(terms)))
    var = list(range(n_vars))
    anc = n_vars + len(terms)

    def oracle():
        for term, w in zip(terms, work):
            _term_onto(term, w, var)
        g.mark_or(work, anc)               # at least one term satisfied
        for term, w in zip(terms, work):
            _term_onto(term, w, var)       # uncompute the work qubits

    if iters is None:
        iters = g.n_iterations(2 ** n_vars, max(len(dnf_solutions(terms, n_vars)), 1))
    return g.run_grover(oracle, var, anc, iters, anc + 1,
                        shots=shots, draw=draw, device=device)
