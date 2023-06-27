<br>

### Development Notes

This repository's virtual environment is created via

```shell
    # Environment
    conda create --prefix .../uncertainty
    conda activate uncertainty
    
    # Mathematics Packages
    conda install -c conda-forge pymc
    pip install numpyro
    pip install blackjax
    
    # Tests & Evaluations Packages
    conda install -c conda-forge pytest coverage pylint pytest-cov flake8
```

The [filter.txt](/docs/filter.txt) document lists the core libraries of [requirements.txt](/requirements.txt).  Thus

```shell
    pip freeze -r docs/filter.txt > requirements.txt
```

Subsequently, retain `numpy` and `pandas` within the second part of `requirements.txt`.  To generate the dotfile that [`pylint`](https://pylint.pycqa.org/en/latest/user_guide/checkers/features.html) - the static code analyser - will use for analysis, run

```shell
    pylint --generate-rcfile > .pylintrc
```

Use the command

```shell
  conda list {regex}
```

to search for the details of a particular package.

<br>
<br>

### References

* [conda](https://docs.conda.io/projects/conda/en/stable/)
    * `conda search -i ....`
* [pip](https://pip.pypa.io/en/stable/)

<br>
<br>

<br>
<br>

<br>
<br>

<br>
<br>
