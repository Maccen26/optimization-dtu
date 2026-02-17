from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    """Helper for plotting objective contours with constraint overlays.

    - objective: callable f(x) -> float where x is a 2D point (np.ndarray of shape (2,)).
    - constraints: iterable of callables g_i(x) -> float using the convention g_i(x) >= 0 is feasible.
    """

    def __init__(
        self,
        objective: Optional[Callable[[np.ndarray], float]] = None,
        constraints: Optional[Iterable[Callable[[np.ndarray], float]]] = None,
        x: Optional[Sequence[float]] = None,
    ) -> None:
        self.objective: Optional[Callable[[np.ndarray], float]] = objective
        self.constraints: List[Callable[[np.ndarray], float]] = list(constraints or [])
        self.x: Optional[np.ndarray] = np.asarray(x, dtype=float) if x is not None else None

    def set_objective(self, objective: Callable[[np.ndarray], float]) -> None:
        self.objective = objective

    def set_constraints(self, constraints: Iterable[Callable[[np.ndarray], float]]) -> None:
        self.constraints = list(constraints)

    def set_x(self, x: Sequence[float]) -> None:
        self.x = np.asarray(x, dtype=float)

    def plot_contour(
        self,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        grid_size: int = 150,
        levels: int = 25,
        cmap: str = "viridis",
        feasible_color: str = "#8fd19e",
        feasible_alpha: float = 0.25,
        infeasible_hatch: str = "//",
        constraint_color: str = "#d62728",
        ax: Optional[plt.Axes] = None,
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot objective contours and shade feasible vs. infeasible regions.

        - xlim/ylim: bounds for the plot; default comes from x guess if provided, else (-5, 5).
        - grid_size: resolution of the mesh grid.
        - levels: number of contour levels for the objective.
        - constraints must satisfy g(x) >= 0 to be feasible; boundaries g(x) = 0 are drawn.
        """

        if self.objective is None:
            raise ValueError("Objective function must be set before plotting.")

        default_bounds = (-5.0, 5.0)
        if self.x is not None and self.x.size >= 2:
            span = 3.0
            x_center, y_center = float(self.x[0]), float(self.x[1])
            xlim = xlim or (x_center - span, x_center + span)
            ylim = ylim or (y_center - span, y_center + span)
        xlim = xlim or default_bounds
        ylim = ylim or default_bounds

        xs = np.linspace(xlim[0], xlim[1], grid_size)
        ys = np.linspace(ylim[0], ylim[1], grid_size)
        X, Y = np.meshgrid(xs, ys)

        def eval_obj(x_val: float, y_val: float) -> float:
            return float(self.objective(np.array([x_val, y_val])))

        Z = np.fromiter((eval_obj(xv, yv) for xv, yv in zip(X.ravel(), Y.ravel())), float)
        Z = Z.reshape(X.shape)

        fig, ax = (plt.subplots() if ax is None else (ax.figure, ax))
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        fig.colorbar(contour, ax=ax, label="Objective")

        feasible_mask = np.ones_like(X, dtype=bool)
        for constraint in self.constraints:
            values = np.fromiter(
                (float(constraint(np.array([xv, yv]))) for xv, yv in zip(X.ravel(), Y.ravel())),
                float,
            ).reshape(X.shape)
            feasible_mask &= values >= 0.0
            ax.contour(X, Y, values, levels=[0.0], colors=constraint_color, linewidths=1.0)

        if self.constraints:
            ax.contourf(
                X,
                Y,
                feasible_mask.astype(int),
                levels=[0.5, 1.5],
                colors=[feasible_color],
                alpha=feasible_alpha,
            )
            ax.contourf(
                X,
                Y,
                (~feasible_mask).astype(int),
                levels=[0.5, 1.5],
                colors="none",
                hatches=[infeasible_hatch],
            )

        if self.x is not None and self.x.size >= 2:
            ax.plot(self.x[0], self.x[1], marker="o", color="#1f77b4", markersize=6, label="x guess")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Objective contour with constraints")
        if self.constraints:
            ax.legend(loc="upper right")

        if show:
            plt.show()

        return fig, ax
