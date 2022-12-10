# Time-varying graph representation learning via higher-order skip-gram with negative sampling

Repository where we show how the method HOSGNS (Higher-Order Skip-Gram with Negative Sampling) can be applied to face-to-face proximity data (http://www.sociopatterns.org/), and to synthetic data generated by an agent-based model (https://github.com/BDI-pathogens/OpenABM-Covid19).

If you use the code in this repository, please cite us:
```bibtex
@article{piaggesi2022time,
  title={Time-varying graph representation learning via higher-order skip-gram with negative sampling},
  author={Piaggesi, Simone and Panisson, Andr{\'e}},
  journal={EPJ Data Science},
  volume={11},
  number={1},
  pages={33},
  year={2022},
  publisher={Springer Berlin Heidelberg}
}
```


## Repository organization

The `data/` folder contains preprocessed time-varying proximity data and corresponding metadata with class labels (where they are available). The original raw data can be found in http://www.sociopatterns.org/datasets/. Furthermore it contains SIR spreading realizations on each presented dataset.

The `code/` folder contains Jupyter notebooks to execute the method on presented datasets:
1. [MakeVariousSupraNetworks](code/MakeVariousSupraNetworks.ipynb): To build and save networkx supra-adjacency graphs.
2. [MakeVariousSupraSparseTensors](code/MakeVariousSupraSparseTensors.ipynb): To build co-occurrence tensors from supra-adjacency graphs (not needed for synthetic datasets).
3. [RemoveEvents](code/RemoveEvents.ipynb): To remove events from empirical temporal graphs and save results.
4. To embed different higher-order representations and test them in downstream tasks.
   - [hosgns_3Way_Stat_SocioPatterns](code/hosgns_3Way_Stat_SocioPatterns.ipynb)
   - [hosgns_4Way_Dyn_SocioPatterns](code/hosgns_4Way_Dyn_SocioPatterns.ipynb) 
   - [hosgns_4Way_StatDyn_SocioPatterns](code/hosgns_4Way_StatDyn_SocioPatterns.ipynb) 
   - [hosgns_3Way_Stat_OpenABM](code/hosgns_3Way_Stat_OpenABM.ipynb)
   - [hosgns_4Way_Dyn_OpenABM](code/hosgns_4Way_Dyn_OpenABM.ipynb)

## Requirements

The repository required packages can be installed from `requirements.txt`. To run the code on OpenABM (last two notebooks) the libraries [GEM](https://github.com/palash1992/GEM) and [SNAP](https://github.com/snap-stanford/snap) are also needed. The code has been tested under Python 3.6.

## References

1. Cattuto, C., Van den Broeck, W., Barrat, A., Colizza, V., Pinton, J. F., & Vespignani, A. (2010). Dynamics of person-to-person interactions from distributed RFID sensor networks. PloS one, 5(7), e11596.

2. Hinch, Robert, et al. "OpenABM-Covid19—An agent-based model for non-pharmaceutical interventions against COVID-19 including contact tracing." PLoS computational biology 17.7 (2021): e1009146.
