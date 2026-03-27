#!/usr/bin/env python3
"""Run the KATL 2024 backtest."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from enhanced_detection.katl_2024.config import INCIDENT
from enhanced_detection.katl_2024.clearances import CLEARANCES
from enhanced_detection.backtest_runner import run_backtest, generate_comparison_plot

INCIDENT.clearances = list(CLEARANCES)


def run():
    result = run_backtest(INCIDENT)
    plot_path = os.path.join(os.path.dirname(__file__), "results.png")
    generate_comparison_plot(result, plot_path)
    return result


if __name__ == "__main__":
    run()
