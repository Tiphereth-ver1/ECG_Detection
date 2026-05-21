import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_TEMPLATE = BASE_DIR / "dataSet" / "ProjectTestDataAnalysis.mat"
PREDICTIONS_PATH = BASE_DIR / "predictions.npy"
HRV_METRICS_PATH = BASE_DIR / "outputs" / "qrs_eval" / "hrv_metrics.csv"
OUTPUT_PATH = BASE_DIR / "dataSet" / "ProjectTestDataAnalysis_filled.mat"

HRV_COLUMNS = {
    "pNN50": "pNN50",
    "RMSSD": "RMSSD",
    "avgRR": "avgRR",
    "sdRR": "sdRR",
    "LF": "LF",
    "HF": "HF",
    "LF_HFratio": "LF_HF",
}


def read_hrv_metrics(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No HRV rows found in {path}")

    return rows


def hrv_array(rows, csv_column):
    return np.asarray([float(row[csv_column]) for row in rows], dtype=float).reshape(1, -1)


def main():
    data = loadmat(ANALYSIS_TEMPLATE)
    predictions = np.load(PREDICTIONS_PATH, allow_pickle=True).item()
    metrics = read_hrv_metrics(HRV_METRICS_PATH)

    n_records = data["QRS"].shape[1]
    if len(predictions) != n_records:
        raise ValueError(f"Expected {n_records} prediction records, found {len(predictions)}")
    if len(metrics) != n_records:
        raise ValueError(f"Expected {n_records} HRV rows, found {len(metrics)}")

    for i in range(n_records):
        pred = np.asarray(predictions[i + 1], dtype=np.uint32) + np.uint32(1)
        data["QRS"][0, i] = pred.reshape(1, -1)

    for mat_column, csv_column in HRV_COLUMNS.items():
        data[mat_column] = hrv_array(metrics, csv_column)

    clean_data = {key: value for key, value in data.items() if not key.startswith("__")}
    savemat(OUTPUT_PATH, clean_data)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
