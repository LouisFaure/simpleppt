|PyPI| |Build & Test| |codecov| |License| |Code style: black|

SimplePPT
=========

A python implementation of SimplePPT algorithm, with GPU acceleration.

Installation
------------

.. code:: bash

    pip install -U simpleppt

Usage
-----

.. code:: python

    from sklearn.datasets import make_classification
    import simpleppt

    X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, n_classes=3)

    SP = simpleppt.ppt(X1,Nodes=30,seed=1,progress=False,lam=10)
    simpleppt.project_ppt(SP, X1, c=Y1)

Citation
--------

Please cite the following paper if you use it::

    Mao et al. (2015), SimplePPT: A simple principal tree algorithm
    SIAM International Conference on Data Mining.

GPU dependencies (optional)
---------------------------

If you have a nvidia GPU, simpleppt can leverage CUDA computations for speedup in tree inference. The latest version of rapids framework is required (at least 0.17) it is recommanded to create a new conda environment::

    conda -n SimplePPT-gpu -c rapidsai -c nvidia -c conda-forge -c defaults cuml=23.04 cugraph=23.04 python=3.8 cudatoolkit=11.8 -y
    conda activate SimplePPT-gpu
    pip install simpleppt


.. |PyPI| image:: https://img.shields.io/pypi/v/simpleppt.svg
   :target: https://pypi.python.org/pypi/simpleppt/
.. |Build & Test| image:: https://github.com/LouisFaure/simpleppt/actions/workflows/test.yml/badge.svg
   :target: https://github.com/LouisFaure/simpleppt/actions/workflows/test.yml
.. |codecov| image:: https://codecov.io/gh/LouisFaure/simpleppt/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/LouisFaure/simpleppt
.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://github.com/LouisFaure/simpleppt/blob/master/LICENSE
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black



.. toctree::
   :maxdepth: 0
   :hidden:

   api
