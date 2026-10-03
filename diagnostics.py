"""Readable risk bands and operator-oriented maintenance guidance."""

from __future__ import annotations

import pandas as pd

SENSOR_LIMITS = {
    "temperature": {"label": "Temperature", "unit": "°C", "high": 85, "direction": "high"},
    "vibration": {"label": "Vibration", "unit": "mm/s", "high": 4.5, "direction": "high"},
    "pressure": {"label": "Pressure", "unit": "kPa", "low": 90, "direction": "low"},
    "current": {"label": "Current", "unit": "A", "high": 22, "direction": "high"},
}


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.55:
        return "High"
    if probability >= 0.30:
        return "Elevated"
    return "Low"


def sensor_drivers(machine: pd.Series) -> list[str]:
    drivers: list[str] = []
    for feature, rule in SENSOR_LIMITS.items():
        value = float(machine[feature])
        if rule["direction"] == "high" and value >= float(rule["high"]):
            drivers.append(f"{rule['label']} {value:.1f} {rule['unit']} is above the demo reference limit")
        if rule["direction"] == "low" and value <= float(rule["low"]):
            drivers.append(f"{rule['label']} {value:.1f} {rule['unit']} is below the demo reference limit")
    return drivers


def maintenance_guidance(probability: float, machine: pd.Series) -> tuple[str, str, list[str]]:
    band = risk_band(probability)
    drivers = sensor_drivers(machine)
    if band == "Critical":
        action = "Escalate now: verify the readings, inspect the asset, and follow site procedures for load reduction or a controlled stop."
        timing = "Immediate review"
    elif band == "High":
        action = "Create a priority work order and inspect the flagged subsystem at the next safe opportunity."
        timing = "This shift"
    elif band == "Elevated":
        action = "Review the trend and plan a targeted inspection during the next maintenance window."
        timing = "Next maintenance window"
    else:
        action = "Continue normal monitoring and keep the planned preventive maintenance schedule."
        timing = "Routine monitoring"
    if not drivers:
        drivers = ["Combined sensor pattern raised the model score; review recent trend and sensor calibration."]
    return band, timing, [action, *drivers]
