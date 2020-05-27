# fast sinc transform

A python implementation of the fast sinc-transform described by [Greengard et. al., 2006](https://doi.org/10.2140/camcos.2006.1.121) as implemented by Hannah Lawrence in [fast-sinc-transform](https://fast-sinc-transform.readthedocs.io/en/latest/Overview.html). Utilizes [FINUFFT](https://finufft.readthedocs.io/) and [fastgl](https://people.sc.fsu.edu/~jburkardt/py_src/fastgl/fastgl.html) (modified to use [numba](http://numba.pydata.org/)).

<p float="left" align="middle">
  <img src="https://raw.githubusercontent.com/gauteh/fsinc/master/doc/example_1d.png" width="40%" />
  <img src="https://raw.githubusercontent.com/gauteh/fsinc/master/doc/example_2d.png" width="40%" />
</p>

Theory and details are described in more detail in [doc/fsinc.md](doc/fsinc.md) ([pdf](https://raw.githubusercontent.com/gauteh/fsinc/master/doc/fsinc.pdf)).

There are a couple of examples in [examples/](examples/), tests can be run with `pytest`. To show plots during testing use `pytest --plot`.

## Installation

```sh
pip install .
```

## Building docs

Set up the environment in [doc/environment.yml](doc/environment.yml), install enough of tex-live and run `make` to generate `pdf` using `pandoc`.

# References

Greengard, L., Lee, J. Y., & Inati, S. (2006). The fast sinc transform and image reconstruction from nonuniform samples in k-space. Communications in Applied Mathematics and Computational Science, 1(1), 121–131. [https://doi.org/10.2140/camcos.2006.1.121](https://doi.org/10.2140/camcos.2006.1.121)
