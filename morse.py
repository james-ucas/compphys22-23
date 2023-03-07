from typing import Callable

import matplotlib.figure, matplotlib.axes
import numpy as np
from numpy import ndarray


def morse_potential(x: ndarray, rho: float, epsilon: float = 1.0, r0: float = 1.0) -> ndarray:
    """
    Compute the Morse potential at pair distance x with well depth epsilon,
    equilibrium distance r0, and shape parameter rho
    """
    x = x / r0
    v = np.exp(rho * (1 - x))
    return epsilon * v * (v - 2)


def pairwise_potential(potential: Callable[[ndarray], ndarray], xss: ndarray) -> float:
    """
    >>> import numpy as np
    >>> morse_13_6 = np.loadtxt("morse-n_13-rho_6.txt")
    >>> rho = 6.0
    >>> pairwise_potential(potential = lambda x: morse_potential(x, rho), xss = morse_13_6) # doctest: +ELLIPSIS
    -42.439862...
    """
    atoms, *_ = xss.shape
    left, right = np.triu_indices(atoms, k=1)

    return potential(
        np.linalg.norm(xss[left] - xss[right], axis=1)
    ).sum()


def question2(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Subplot) -> None:
    """
    Plot the morse potential on r/r0 in (0, 2)
    for rho in {3, 6, 10, 14} .
    """
    x = np.linspace(0.1, 2.0, 100)
    rhos = [3, 6, 10, 14]
    linestyles = ['-', '--', '-.', ':']

    for rho, linestyle in zip(rhos, linestyles):
        ax.plot(
            x,
            morse_potential(x, rho=rho),
            linestyle=linestyle,
            label=fr"$\rho = {rho:.0f}$"
        )

    ax.set_xlim(0, 2)
    ax.set_ylim(-1.1, 0.5)
    ax.legend(loc="upper left")
    ax.set_xlabel("$r/r_{0}$")
    ax.set_ylabel("$V/\epsilon$")


def question3() -> float:
    """
    Compute the total energy of the Morse cluster in "morse-n_13-rho_6.txt"
    """
    morse_13_6 = np.loadtxt("morse-n_13-rho_6.txt")
    return pairwise_potential(
        potential=lambda x: morse_potential(x, rho=6),
        xss=morse_13_6
    )


def main():
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6, 6 / 2 ** 0.5))

    question2(fig, ax)

    plt.savefig("morse.png", dpi=160)
    plt.show()

    energy = question3()
    print(f"the total potential energy of the Morse cluster is {energy} epsilon.")


if __name__ == '__main__':
    main()
