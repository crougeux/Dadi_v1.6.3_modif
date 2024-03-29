## ∂a∂i
Modified version od ∂a∂i, allowing the use of the `dadi.Inference.optimize_anneal` for the simulated annealing approach.

This modified version includes the 26 demographic models of divergence used in [Rougeux et al. (2017)](https://academic.oup.com/gbe/article-lookup/doi/10.1093/gbe/evx150) and in [Rougeux et al. (2019)](https://onlinelibrary.wiley.com/doi/abs/10.1111/jeb.13482) available in `modeldemo_mis_new_models`.

## Citation
Please cite this paper if you use demographic models from implemented scripts:                                                                  
Rougeux, C., Bernatchez, L., & Gagnaire, P.-A. (2017). Modeling the Multiple Facets of Speciation-with-Gene-Flow toward Inferring the Divergence History of Lake Whitefish Species Pairs (Coregonus clupeaformis). Genome Biology and Evolution, 9(8), 2057–2074.

## Some hints
dadi is a powerful method for fitting demographic models to genetic data.

Binary installers are available for OS X and Windows. If you use a binary installer, we still suggest downloading the source, so as to get the documentation, examples and tests.

To install from source, simply run `python setup.py install`.
After that, run a (growing) series of tests on the installation. Change to the `tests` directory and run `python run_tests.py`
Usage examples are found under the `examples` directory.

To run this pipeline, put your frequency spectrum into the `/Dadi_studied_model/01_Data/` and run the desired models by adjusting scripts to your data from the `/Dadi_studied_model/00_inference/` directory.
Have fun!
