import numpy as np
from scipy.io import loadmat, savemat

data = loadmat("dataSet/ProjectTestDataAnalysis.mat")
predictions = np.load("predictions.npy", allow_pickle=True).item()

# Existing scaffold dimensions
n_records = data["QRS"].shape[1]

# Overwrite each existing QRS cell
for i in range(n_records):

    pred = np.asarray(predictions[i + 1], dtype=np.uint16)
    data["QRS"][0, i] = pred.reshape(1, -1)

savemat("dataSet/ProjectTestDataAnalysis_filled.mat",data)