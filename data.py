"""Synthetic fleet and trend data used by the interactive demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

MACHINE_TYPES = ["CNC spindle", "Hydraulic pump", "Conveyor motor", "Air compressor", "Industrial fan"]
SITES = ["North plant", "West plant", "Central plant"]


def generate_demo_fleet(rows: int = 36, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wear = rng.beta(1.7, 2.1, rows)
    temperature = 61 + 18 * wear + rng.normal(0, 5, rows)
    vibration = 1.25 + 5.0 * wear + rng.normal(0, 0.72, rows)
    pressure = 116 - 25 * wear + rng.normal(0, 5, rows)
    current = 11.5 + 11.0 * wear + rng.normal(0, 2.3, rows)

    # Add distinct sensor faults as well as gradual wear.
    for index in rng.choice(rows, size=max(2, rows // 12), replace=False):
        channel = int(rng.integers(0, 4))
        if channel == 0:
            temperature[index] += rng.uniform(8, 15)
        elif channel == 1:
            vibration[index] += rng.uniform(1.5, 3.0)
        elif channel == 2:
            pressure[index] -= rng.uniform(9, 17)
        else:
            current[index] += rng.uniform(4, 8)

    now = pd.Timestamp.now().floor("min")
    return pd.DataFrame(
        {
            "machine_id": [f"M-{index + 1:03d}" for index in range(rows)],
            "asset_type": [MACHINE_TYPES[index % len(MACHINE_TYPES)] for index in range(rows)],
            "site": [SITES[index % len(SITES)] for index in range(rows)],
            "temperature": temperature.round(1),
            "vibration": vibration.round(2),
            "pressure": pressure.round(1),
            "current": current.round(1),
            "runtime_hours": rng.integers(1100, 24000, rows),
            "days_since_service": rng.integers(4, 240, rows),
            "_wear": wear,
            "reading_at": now - pd.to_timedelta(rng.integers(1, 50, rows), unit="m"),
        }
    )


def generate_sensor_history(machine: pd.Series, points: int = 48, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.linspace(0, 1, points)
    wear = float(machine.get("_wear", 0.45))
    starts = {
        "temperature": 61 + 18 * wear,
        "vibration": 1.25 + 5.0 * wear,
        "pressure": 116 - 25 * wear,
        "current": 11.5 + 11.0 * wear,
    }
    deviations = {"temperature": 3.1, "vibration": 0.5, "pressure": 3.7, "current": 1.5}
    times = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=points, freq="30min")
    output: dict[str, object] = {"time": times}
    for feature in ("temperature", "vibration", "pressure", "current"):
        start = starts[feature]
        end = float(machine[feature])
        jitter = rng.normal(0, deviations[feature], points)
        output[feature] = (start + steps * (end - start) + jitter).round(2)
    return pd.DataFrame(output)
