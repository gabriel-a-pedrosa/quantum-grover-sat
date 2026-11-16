"""Generic Grover pieces shared by the SAT and Sudoku code.

The oracle flips an ancilla prepared in |->, so the flip becomes a -1 phase
on the solutions (phase kickback).
"""

import numpy as np
import pennylane as qml


def n_iterations(N, M):
    """Optimal number of Grover iterations for M solutions out of N states."""
    # k = pi/(4 theta) - 1/2 with theta = arcsin(sqrt(M/N)), poor when M > N/2
    if M <= 0 or M >= N:
        return 1
    theta = np.arcsin(np.sqrt(M / N))
    k = round(np.pi / (4 * theta) - 0.5)
    return max(int(k), 1)


def hadamards(wires):
    for w in wires:
        qml.Hadamard(w)


def diffuser(wires):
    """Grover diffuser: reflection about the uniform superposition."""
    wires = list(wires)
    hadamards(wires)
    for w in wires:
        qml.PauliX(w)
    # multi-controlled Z on all the wires (phase flip on |1...1>)
    if len(wires) == 1:
        qml.PauliZ(wires[0])
    else:
        qml.ctrl(qml.PauliZ(wires[-1]), control=wires[:-1])
    for w in wires:
        qml.PauliX(w)
    hadamards(wires)


# --- small helpers to write the oracles with --------------------------------

def flip_if(controls, target, values=None):
    """Flip `target` when the control qubits equal `values` (default all 1s)."""
    controls = list(controls)
    if not controls:
        qml.PauliX(target)
    elif values is None:
        qml.MultiControlledX(wires=[*controls, target])
    else:
        qml.MultiControlledX(wires=[*controls, target], control_values=list(values))


def mark_and(controls, target):
    # flip target iff every control is 1
    flip_if(controls, target)


def mark_or(controls, target):
    # flip target iff at least one control is 1, using OR = not(all zero)
    controls = list(controls)
    qml.PauliX(target)
    flip_if(controls, target, [0] * len(controls))


def not_equal(cell_a, cell_b, target, scratch):
    """Flip target when the two cells hold different values."""
    # wider cells XOR each bit pair into scratch and OR them, calling it twice undoes it
    a, b = list(cell_a), list(cell_b)
    if len(a) == 1:
        qml.CNOT(wires=[a[0], target])
        qml.CNOT(wires=[b[0], target])
        return
    s = list(scratch)[:len(a)]
    for ai, bi, si in zip(a, b, s):
        qml.CNOT(wires=[ai, si])
        qml.CNOT(wires=[bi, si])
    mark_or(s, target)
    for ai, bi, si in zip(a, b, s):
        qml.CNOT(wires=[ai, si])
        qml.CNOT(wires=[bi, si])


def run_grover(oracle, free_wires, ancilla, iters, n_wires,
               measure=None, prep=None, shots=None, draw=False, device=None):
    """Run Grover and return the probabilities (or the circuit drawing if draw=True)."""
    free_wires = list(free_wires)
    measure = free_wires if measure is None else list(measure)

    dev = device if device is not None else qml.device(
        "default.qubit", wires=n_wires, shots=shots)

    @qml.qnode(dev)
    def circuit():
        if prep is not None:
            prep()
        hadamards(free_wires)
        qml.PauliX(ancilla)      # ancilla in |->
        qml.Hadamard(ancilla)
        for _ in range(iters):
            oracle()
            diffuser(free_wires)
        return qml.probs(wires=measure)

    if draw:
        return qml.draw_mpl(circuit)()
    return np.array(circuit())


def plot_probs(probs, n_bits, title="Probability distribution", ax=None):
    """Quick bar chart of a probability vector over n_bits qubits."""
    import matplotlib.pyplot as plt
    labels = [format(i, "0%db" % n_bits) for i in range(len(probs))]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, n_bits * 1.1), 3))
    ax.bar(labels, probs)
    ax.set_xlabel("basis state")
    ax.set_ylabel("probability")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=60)
    return ax


def top_states(probs, n_bits, threshold=None):
    """Basis states that ended up clearly above a flat distribution."""
    if threshold is None:
        threshold = 1.5 / len(probs)
    out = [(format(i, "0%db" % n_bits), float(p))
           for i, p in enumerate(probs) if p >= threshold]
    out.sort(key=lambda t: t[1], reverse=True)
    return out
