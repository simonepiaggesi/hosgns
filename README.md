# Time-varying graph representation learning via higher-order skip-gram with negative sampling

Repository where we show how the method presented in [1] can be applied to data about face-to-face proximity relations collected by the SocioPatterns collaboration [2].

The `data/` folder contains preprocessed time-varying proximity data (with the nights removed) and corresponding metadata with class labels. The original raw data can be found in http://www.sociopatterns.org/datasets/.

The `code/` folder contains Jupyter notebooks to execute the method on the empirical datasets:
1. [MakeVariousSupraNetworks](code/MakeVariousSupraNetworks.ipynb): To build and save networkx supra-adjacency graphs.
2. [MakeVariousSupraSparseTensors](code/MakeVariousSupraSparseTensors.ipynb): To build co-occurrence tensors from supra-adjacency graphs.


## References
[1]
[2] Cattuto, C., Van den Broeck, W., Barrat, A., Colizza, V., Pinton, J. F., & Vespignani, A. (2010). Dynamics of person-to-person interactions from distributed RFID sensor networks. PloS one, 5(7), e11596.
