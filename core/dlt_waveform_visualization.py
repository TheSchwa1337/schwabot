# -*- coding: utf-8 -*-
""""""
DLT Waveform Visualization Module.

Provides comprehensive visualization capabilities for the Delta Lock Transform (DLT)
waveform engine, including comparative analysis with advanced mathematical metrics.

This module integrates with:
- DLTWaveformEngine for core waveform processing
- Advanced mathematical core for comparative overlays
- Matplotlib for visualization rendering
- NumPy for numerical operations

Features:
- Real-time DLT state visualization
- Triplet lock event highlighting
- Phase projection and drift correction plots
- Greyscale confidence mapping
- State transition color coding
- Optional comparative metric overlays
""""""

from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt

import numpy as np

from schwabot.core.dlt_waveform_engine import DLTState, DLTWaveformEngine


def visualize_dlt_waveform()
    time_series: List[float],
        observer_confirmations: Optional[List[bool]] = None,
            comparative_overlay_fn: Optional[Callable[[List[float]], Any]] = None,
            overlay_label: str = "Comparative Metric",
            title: str = "DLT Waveform Engine Visualization",
            figsize: Tuple[int, int] = (12, 8),
            show: bool = True,
            ) -> Tuple[plt.Figure, np.ndarray]:
    """"""
    Visualize the DLT waveform engine's operation on a time series.'

    Args:
        time_series: List of float values (e.g., price or signal)
        observer_confirmations: Optional list of bools for observer lock
        comparative_overlay_fn: Optional function for overlay (e.g., phase/entropy)
        overlay_label: Label for the comparative overlay
        title: Plot title
        figsize: Figure size as (width, height) tuple
        show: Whether to call plt.show()

    Returns:
        Tuple of (figure, axes) for further customization
    """"""
    engine = DLTWaveformEngine()
    deltas = []
    phase_projections = []
    drift_corrections = []
    greyscales = []
    confidences = []
    states = []
    triplet_locks = []
    state_colors = []
    hash_states = []

    state_color_map = {}
        DLTState.LOCKED: "#4caf50",
            DLTState.FADED: "#bdbdbd",
                DLTState.WAITING: "#1976d2",
                DLTState.UNLOCKED: "#f44336",
}
    for i in range(1, len(time_series)):
        obs = ()
            observer_confirmations[i]
            if observer_confirmations and i < len(observer_confirmations)
            else None
        )
        result = engine.update(time_series[i], time_series[i - 1], observer=obs)
        deltas.append(result.meta["delta"])
        phase_projections.append(result.phase_projection)
        drift_corrections.append(result.drift_correction)
        greyscales.append(result.greyscale)
        confidences.append(result.confidence)
        states.append(result.state)
        triplet_locks.append(result.lock_triplet is not None)
        state_colors.append(state_color_map.get(result.state, "#eeeeee"))
        hash_states.append(result.hash_state)

    x = np.arange(1, len(time_series))
    fig, axes = plt.subplots()
        5, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.3]}
    )

    # 1. Deltas
    axes[0].plot(x, deltas, label="Delta", color="#1976d2")
    axes[0].scatter()
        x,
            [d if t else np.nan for d, t in zip(deltas, triplet_locks)],
                color="#ff9800",
                label="Triplet Lock",
                marker="o",
                zorder=5,
                )
    axes[0].set_ylabel("Delta")
    axes[0].legend(loc="upper right")

    # 2. Phase projection
    axes[1].plot(x, phase_projections, label="Phase Projection", color="#388e3c")
    axes[1].set_ylabel("Phase")
    axes[1].legend(loc="upper right")

    # 3. Drift correction
    axes[2].plot(x, drift_corrections, label="Drift Correction", color="#f44336")
    axes[2].set_ylabel("Drift")
    axes[2].legend(loc="upper right")

    # 4. Greyscale confidence
    axes[3].plot(x, greyscales, label="Greyscale Confidence", color="#616161")
    axes[3].plot(x, confidences, label="Confidence", color="#2196f3", linestyle=":")
    axes[3].set_ylabel("Confidence")
    axes[3].legend(loc="upper right")

    # 5. State color bar
    axes[4].imshow([state_colors], aspect="auto", extent=[x[0], x[-1], 0, 1])
    axes[4].set_yticks([])
    axes[4].set_ylabel("State")
    axes[4].set_xlabel("Tick")

    # Optional: Comparative overlay
    if comparative_overlay_fn is not None:
        overlay = comparative_overlay_fn(time_series)
        ax_overlay = axes[0].twinx()
        ax_overlay.plot()
            x, overlay[1:], label=overlay_label,
                color="#9c27b0", alpha=0.5, linestyle="--"
        )
        ax_overlay.set_ylabel(overlay_label, color="#9c27b0")
        ax_overlay.legend(loc="lower right")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if show:
        plt.show()
    return fig, axes