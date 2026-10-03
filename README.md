# Aegis Predictive Maintenance

An interactive Python dashboard that demonstrates how an industrial IoT fleet can use temperature, vibration, pressure, and current readings to prioritize maintenance before equipment failure.

## What it does

- Scores a simulated 3 6-asset factory fleet with a Random Forest classifier.
- Accepts a current telemetry CSV and returns a failure-risk score for each valid row.
- Trains a custom model from historical labeled telemetry, with a stratified holdout evaluation.
- Sorts assets into Low, Elevated, High, and Critical risk bands and surfaces sensor readings to review.
- Shows recent sensor trends and suggested next steps, and exports the scored data to CSV.
- Includes a downloadable sample labeled dataset to try the custom training flow.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL, normally `http://localhost:8501`.

## Use your own data

### Train a model

Choose **Train from labeled CSV** in the sidebar and upload historical records. The CSV needs one numeric column for each sensor and a binary outcome column:

| Required field | Expected measurement |
| --- | --- |
| `temperature` | °C |
| `vibration` | mm/s |
| `pressure` | kPa |
| `current` | A |
| failure outcome | For example, `failure_within_24h` with 0 = no failure and 1 = failure |

Common headers such as `temp_c`, `vibration_mm_s`, `pressure_kpa`, and `current_a` are recognized. A `pressure_bar` column is converted to kPa. The selected target must have two classes. Use at least 40 complete records, with at least five examples of each class. A reproducible stratified 20% holdout is used for the displayed metrics; the final model is then fit to all valid labeled rows.

### Score current telemetry

Upload a separate CSV under **Current telemetry CSV**. It must include the four sensor readings; an optional `machine_id` column is shown in the dashboard. Other columns (asset, site, runtime, and service metadata) pass through to the exported CSV. Invalid sensor rows are skipped and reported.

For a quick custom-model walkthrough, start with the **Sample labeled CSV** download, train on it, then upload a separate current telemetry CSV to score.

## Risk and maintenance guidance

The built-in classifier is trained only on generated data. Its target is a synthetic failure label derived from combinations of sensor excursions, and its scores are for demonstration—not evidence of real-world failure likelihood. The displayed percentages are raw Random Forest class scores and are not calibrated probabilities. Custom-model scores are only as useful as the label quality, sensor calibration, representative coverage, and the time horizon in the training data; assess calibration and performance on time-separated site data before operational use.

The displayed reference limits (85 °C temperature, 4.5 mm/s vibration, 90 kPa pressure, and 22 A current) are example values, not universal industrial alarm limits. Confirm sensor units, failure horizon, model performance on a time-separated site dataset, maintenance actions, and applicable safety procedures with the plant reliability team before operational use. This app does not connect to PLCs or control equipment.

## Project layout

```text
app.py              Streamlit dashboard and CSV workflow
pdm/model.py        Synthetic-data generator and Random Forest training/scoring
pdm/data.py         Demo fleet and trend generation
pdm/diagnostics.py  Risk bands and operator guidance
requirements.txt    Python dependencies
```
