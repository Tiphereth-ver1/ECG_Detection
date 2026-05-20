import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

data = loadmat("dataSet/ProjectTestDataAnalysis.mat")
predictions = np.load("predictions.npy", allow_pickle=True).item()
metrics = pd.read_csv("outputs/qrs_eval/hrv_metrics.csv")
print(data.keys())
print(metrics.keys())
# Existing scaffold dimensions
n_records = data["QRS"].shape[1]

# Overwrite each existing QRS cell
for i in range(n_records):

    pred = np.asarray(predictions[i + 1], dtype=np.uint16)
    data["QRS"][0, i] = pred.reshape(1, -1)

data["pNN50"] = metrics["pNN50"].to_numpy(dtype=float).reshape(1, -1)
data["RMSSD"] = metrics["RMSSD"].to_numpy(dtype=float).reshape(1, -1)
data["avgRR"] = metrics["avgRR"].to_numpy(dtype=float).reshape(1, -1)
data["sdRR"] = metrics["sdRR"].to_numpy(dtype=float).reshape(1, -1)
data["LF"] = metrics["LF"].to_numpy(dtype=float).reshape(1, -1)
data["HF"] = metrics["HF"].to_numpy(dtype=float).reshape(1, -1)
data["LF_HFratio"] = metrics["LF_HF"].to_numpy(dtype=float).reshape(1, -1)

savemat("dataSet/ProjectTestDataAnalysis_filled.mat",data)