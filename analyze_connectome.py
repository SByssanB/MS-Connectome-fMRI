# scripts/analyze_connectome.py

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from nilearn import datasets, input_data, plotting, connectome

# Load example functional dataset from Nilearn (can be replaced with MS data)
data = datasets.fetch_development_fmri(n_subjects=1)
fmri_filenames = data.func[0]

# Define masker for brain region extraction
masker = input_data.NiftiLabelsMasker(labels_img=data.labels,
                                      standardize=True, memory='nilearn_cache')

# Transform the fMRI time series
time_series = masker.fit_transform(fmri_filenames)

# Compute correlation matrix
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Graph analysis using NetworkX
graph = nx.from_numpy_array(correlation_matrix)

# Calculate and display global efficiency
global_efficiency = nx.global_efficiency(graph)
print(f"Global Efficiency: {global_efficiency:.4f}")

# Visualize the matrix
plt.imshow(correlation_matrix, interpolation="nearest", cmap="coolwarm")
plt.title("Functional Connectivity Matrix")
plt.colorbar()
plt.show()
