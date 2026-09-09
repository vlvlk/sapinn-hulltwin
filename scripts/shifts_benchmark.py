"""Entry point: uv run python scripts/shifts_benchmark.py

Roadmap stage 5: trains the physics-structured power model on the Shifts
power-consumption train split and reports median power error on the
in-domain (dev_in) and shifted (dev_out) development splits.
"""

from hulltwin.data.shifts import run_benchmark

if __name__ == "__main__":
    run_benchmark("data/external/power_consumption_upload/real_data")
