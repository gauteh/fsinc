---
title: Fast sinc-transform for reconstruction of non-uniformely sampled images
author: Gaute Hope
autoEqnLabels: true
bibliography: sincinterp.bib
link-citations: true
csl: https://www.zotero.org/styles/the-journal-of-the-acoustical-society-of-america
---

![Sinc interpolation from non-uniform to uniform grids](example_2d.png)

# Interpolating images using the sinc-interpolation formula

A continuous signal, $s_c$, is sampled at discrete points $x_n$, so than
$s[x_n]$ is the sample values. To re-construct the continuous signal from the
discrete samples the discrete values are convolved with the $sinc$-function
[@Shannon1948; @Oppenheim2014]:

$$ s(x) = \sum \limits^N_{i=1} s[x_i] \cdot sinc\left(\frac{x - x_i}{T}\right) $$ {#eq:sinc-recon}

where $T$ is the sampling-interval (inverse of frequency) so that $x_n = nT$.
This is known as the _Nyquist-Shannon_-interpolation formula. When the sample
rate is sufficiently high, satisfying the Nyquist-criterion of at least two
times the highest frequency in the signal, the signal can be perfectly
reconstructed. Here $sinc$ is the normalized sinc-function:

$$ sinc(x) = \frac{sin(\pi x)}{\pi x} $$ {#eq:sinc}

## The fast sinc transform

The Fourier transform of the sinc-transform is the $\Pi$-function (rectangle).
The Fourier-transform of $sinc^2$ is the $\Lambda$-function (triangle). The
$sinc$-transform is defined as:

$$ Um = \sum \limits^N_{n=1} q_n sinc (\mathbf{k}_n - \mathbf{v}_m) $$ {#eq:sinc-transform}

The (discrete) convolution in @eq:sinc-transform can be calculated quickly
using the _NUFFT_ library since the convolution in $x$ equals a multiplication
in the $k$-domain:

1. Weight $s[x]$ if non-uniformely spaced.
2. Take forward Fourier transform of $s[x]$ to quadrature nodes (e.g. Gauss-Legendre), to get
   $S[k]$.
3. Integrate $S[k]$ from $[-1, 1]$.
4. Take inverse Fourier transform of $\int S[k]$

# Using the fast sinc transform to reconstruct images from non-uniform images

The approximate (inverse or _adjoint_) Fourier transform [@Greengard2006]:

$$ \rho(\mathbf{r}_m) \approx \sum\limits_{n=1}{N} s(n) e^{-2 \pi i \mathbf{k}(n) \cdot \mathbf{r}_m} \cdot w_n $$ {#eq:approx-dft}

## Convention

$$ sinc(\mathbf{k}) = sinc(k_1) \cdot sinc(k_2) \cdot \dots $$

## Sinc-kernel weights

Optimal weights (Sinc-3 in [@Choi1998]):

$$ \frac{1}{w_n} = \sum\limits_{m=1}^{N} sinc^2(\mathbf{k}(m) - \mathbf{k}(n)) $$ {#eq:optimal-weights}

## Jacobian weights

Another choice weights is the difference between samples, Jacobian-weighting,
see Sinc-2 in [@Choi1998], so that densely sampled regions are scaled down
proportionally. For a single-variable scalar function $f(x')$:

$$ \mathbf J = \frac{\partial x'}{\partial x_{uf}} $$

where $x'$ is the non-uniform samples and $x_{uf}$ is an equidistant
monotonically increasing grid.

$$ w_n = x_{n+1} - x_n $$

up to $n = N - 1$, and $w_N = w_{N-1}$.

### Jacobian of a scalar function of two variables

The points are transformed from non-uniform sampling to uniform sampling:

$x' = x$

```{=latex}
  \begin{align*}
    x' &= x(i) \\
    y' &= \theta(y)
  \end{align*}
```

where $x'$ and $y'$ is the uniform grid coordinates. The Jacobian for the set
of equations is:

$$
    \mathbf J = \begin{bmatrix}
                  \frac{\partial x'}{\partial \phi(x)} & \frac{\partial x'}{\partial \theta(y)} \\
                  \frac{\partial y'}{\partial \phi(x)} & \frac{\partial y'}{\partial \theta(y)}
                \end{bmatrix}
$$


## NUFFT Type 3

The non-uniform fast Fourier transform [@Barnett2019], NUFFT, type 3 (most general) computes sums of type:

The forward transform:

$$ G_j = \sum \limits^P_{p=1} g_p e^{-i \mathbf{k}_j \cdot \mathbf{x}_p} $$ {#eq:nufft-3-fwd}

or, the inverse (adjoint) transform:

$$ g_p = \sum \limits^J_{j=1} G_j e^{+i \mathbf{k}_j \cdot \mathbf{x}_p} $$ {#eq:nufft-3-inv}

These use the same implementation, with only the sign of $i$ changed.

## Gauss-Legendre quadrature

Weights are found for nodes on interval $[-1, 1]$ (re-scale input to this
interval), multiply by weights to numerically integrate. This is exact for a
polynominal with degree less or equal to $2n -1$, where $n$ is number of nodes.

$$ \int_{-1}^1 f(x) \approx \sum \limits^{n}_{i=0} w_i \cdot f(x_i) $$

# Interpolating from non-uniform points to a uniform grid

$s[x_i]$ is the $N$ samples at non-uniform points $\mathbf{x}$. We wish to find $s'[x_p]$, where $x_p$ is a regular grid and $max(|s'[x] - s[x]|)$ is minimized (_minimax_).

## Scheme

1. $S_k = \mathcal{F}\left\{ s(x_i) \right\}$, the discrete non-uniform Fourier transform of the samples.

# References
