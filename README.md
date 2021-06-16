[![Install & Load](https://github.com/LouisFaure/simpleppt/actions/workflows/install.yml/badge.svg)](https://github.com/LouisFaure/simpleppt/actions/workflows/install.yml)

# SimplePPT
Python implementation of [SimplePPT algorithm](https://doi.org/10.1137/1.9781611974010.89), with GPU acceleration.

Please cite the following paper if you use it:

```
Mao et al. (2015), SimplePPT: A simple principal tree algorithm, SIAM International Conference on Data Mining.
```

GPU dependencies (optional)
---------------------------

If you have a nvidia GPU, simpleppt can leverage CUDA computations for speedup in tree inference. The latest version of rapids framework is required (at least 0.17) it is recommanded to create a new conda environment:

    conda create -n SimplePPT-gpu -c rapidsai -c nvidia -c conda-forge -c defaults \
        rapids=0.19 python=3.8 cudatoolkit=11.0 -y
    conda activate SimplePPT-gpu
    pip install simpleppt
