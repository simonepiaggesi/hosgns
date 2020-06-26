# Time-varying graph representation learning via higher-order skip-gram with negative sampling

Repository where we show how the method presented in [1] can be applied to data about face-to-face proximity relations collected by the SocioPatterns collaboration [2].

The `data/` folder contains preprocessed time-varying proximity data (with the nights removed) and corresponding metadata with class labels. The original raw data can be found in http://www.sociopatterns.org/datasets/. Furthermore it contains SIR spreading realizations on each presented dataset.

The `code/` folder contains Jupyter notebooks to execute the method on the empirical datasets:
1. [MakeVariousSupraNetworks](code/MakeVariousSupraNetworks.ipynb): To build and save networkx supra-adjacency graphs.
2. [MakeVariousSupraSparseTensors](code/MakeVariousSupraSparseTensors.ipynb): To build co-occurrence tensors from supra-adjacency graphs.
3. - [hosgns_3Way_Stat](code/hosgns_3Way_Stat.ipynb)
   - [hosgns_4Way_Dyn](code/hosgns_4Way_Dyn.ipynb) To embed different higher-order representations and test them in downstream tasks.
   - [hosgns_4Way_StatDyn](code/hosgns_4Way_StatDyn.ipynb) 

## References
[1] Simone Piaggesi and André Panisson (2020). Time-varying Graph Representation Learning via Higher-Order Skip-Gram with Negative Sampling. arXiv preprint arXiv:2006.14330

[2] Cattuto, C., Van den Broeck, W., Barrat, A., Colizza, V., Pinton, J. F., & Vespignani, A. (2010). Dynamics of person-to-person interactions from distributed RFID sensor networks. PloS one, 5(7), e11596.
