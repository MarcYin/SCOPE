# Leaf Angle Distribution Functions

The **leaf angle distribution** (LAD), also called the leaf inclination
distribution function (LIDF), describes the probability density of leaf-normal
zenith angles within a canopy.  It is one of the most influential structural
inputs to any turbid-medium radiative transfer model: it controls the canopy
extinction coefficients, the single-scattering phase function, and the
directional gap-fraction profile.

SCOPE-RTM provides two built-in parameterizations — the Verhoef two-parameter
bimodal and the Campbell ellipsoidal — and the experiment script demonstrates
a third (the Beta distribution of Goel & Strebel) for comparison.


## Why does the LAD matter?

In the 4SAIL radiative transfer equations the LAD enters through a single
integration step that averages the angle-dependent volume-scattering function
over all possible leaf orientations:

$$
G_s = \sum_{i} \frac{\chi_s(\theta_l^{(i)})}{\cos\theta_s}\, f(\theta_l^{(i)})
$$

where $\theta_l^{(i)}$ are the discrete leaf inclination bins, $\chi_s$ is the
Ross-Nilson G-function (projected area of a leaf in the solar direction), and
$f(\theta_l)$ is the LAD weight.  An analogous summation produces $G_o$ for
the observer direction.  These two coefficients ($k_s$, $k_o$) propagate into
every reflectance, transmittance, and gap-probability quantity computed by the
model.

**In practice** a planophile canopy (mostly horizontal leaves) has large
extinction at small zenith angles and produces high near-infrared reflectance,
whereas an erectophile canopy (mostly vertical leaves) transmits near-nadir
light deep into the canopy and produces lower reflectance with a weaker
hotspot.


## Distribution shapes at a glance

The figure below shows probability vs. leaf inclination angle for all three
model families.  Higher probability at low angles means more horizontal leaves
(planophile); higher probability near 90 deg means more vertical leaves
(erectophile).

![Leaf angle distribution shapes](assets/lad_distributions.svg)


## 1. Verhoef Two-Parameter Bimodal (1998)

### Theory

Verhoef (1998, ch. 3) proposed a two-parameter family that generates a cumulative
density function on $[0, \pi/2]$ through the implicit equation

$$
a\,\sin x + \tfrac{1}{2}\,b\,\sin 2x = y
$$

where $y = 2\theta/\pi$ (normalized angle), and $a$, $b$ are the shape
parameters.  The equation is solved iteratively (Newton-like half-step) for
each angular bin boundary, and the bin frequencies are obtained by differencing
the cumulative values.

When $a > 1$ the distribution degenerates to the *de Wit cosine* form
$F(\theta) = 1 - \cos\theta$, which is the spherical distribution in the
angle domain (not solid-angle domain).

**Six canonical types** cover the classical leaf-angle archetypes introduced
by de Wit (1965):

| Name          | a      | b      | Description                              |
|---------------|--------|--------|------------------------------------------|
| Planophile    |  1.0   |  0.0   | Horizontal leaves; peak at 0 deg         |
| Erectophile   | -1.0   |  0.0   | Vertical leaves; peak near 90 deg        |
| Plagiophile   |  0.0   | -1.0   | Oblique leaves; peak near 45 deg         |
| Extremophile  |  0.0   |  1.0   | Bimodal: very flat + very steep           |
| Spherical     | -0.35  | -0.15  | Uniform in solid-angle space              |
| Uniform       |  0.0   |  0.0   | Uniform in inclination-angle space        |

### SCOPE-RTM API

```python
from scope import scope_lidf

# Planophile distribution -> 13-element tensor
lidf = scope_lidf(lidfa=1.0, lidfb=0.0)

# Spherical
lidf = scope_lidf(lidfa=-0.35, lidfb=-0.15)
```

The returned tensor has 13 elements corresponding to the angular bins centred
at {5, 15, 25, 35, 45, 55, 65, 75, 81, 83, 85, 87, 89} deg --- eight coarse
10 deg bins followed by five fine 2 deg bins.  The finer discretization near
90 deg captures the rapid variation of near-vertical leaf populations.

### Reference

> Verhoef, W. (1998). *Theory of radiative transfer models applied in optical
> remote sensing of vegetation canopies.* Ph.D. thesis, Wageningen Agricultural
> University.
>
> de Wit, C.T. (1965). *Photosynthesis of leaf canopies.* Agricultural Research
> Reports 663, Pudoc, Wageningen.


## 2. Campbell Ellipsoidal (1986, 1990)

### Theory

Campbell (1986, 1990) models the leaf population as if the leaf normals were
distributed uniformly over the surface of an **ellipsoid of revolution**.  A
single parameter --- the mean leaf inclination angle $\bar\alpha$ (degrees) ---
controls the ellipsoid eccentricity:

$$
\varepsilon = \exp\!\bigl(-1.6184\times10^{-5}\,\bar\alpha^3
                        + 2.1145\times10^{-3}\,\bar\alpha^2
                        - 1.2390\times10^{-1}\,\bar\alpha
                        + 3.2491\bigr)
$$

The probability density for each angular bin $[\theta_1, \theta_2]$ is then
evaluated in three regimes:

| Eccentricity       | Ellipsoid shape     | Integration formula                      |
|--------------------|--------------------|------------------------------------------|
| $\varepsilon = 1$  | Sphere             | $\lvert\cos\theta_1 - \cos\theta_2\rvert$ |
| $\varepsilon > 1$  | Oblate (flat)      | Hyperbolic + logarithmic terms           |
| $\varepsilon < 1$  | Prolate (elongated)| Elliptic + arcsin terms                  |

The key property is that a **single, physically interpretable parameter**
(mean leaf angle) smoothly interpolates between planophile ($\bar\alpha\to 0$)
and erectophile ($\bar\alpha\to 90$), with $\bar\alpha \approx 57$ deg
recovering the spherical distribution.

### SCOPE-RTM API

```python
from scope import campbell_lidf

# Spherical distribution with 13 bins (to match SCOPE litab)
lidf = campbell_lidf(alpha=57.0, n_elements=13)

# Planophile with the default 18-bin discretization
lidf = campbell_lidf(alpha=10.0)
```

The `n_elements` parameter controls the number of uniform bins spanning 0-90 deg.
When used with `FourSAILModel`, set `n_elements` to match the model's
`n_angles` constructor argument (default 13 for SCOPE, 18 for PROSAIL
convention).

### Reference

> Campbell, G.S. (1986). Extinction coefficients for radiation in plant canopies
> calculated using an ellipsoidal inclination angle distribution.
> *Agricultural and Forest Meteorology*, 36(4), 317-321.
> [doi:10.1016/0168-1923(86)90010-9](https://doi.org/10.1016/0168-1923(86)90010-9)
>
> Campbell, G.S. (1990). Derivation of an angle density function for canopies
> with ellipsoidal leaf angle distributions. *Agricultural and Forest
> Meteorology*, 49(3), 173-176.
> [doi:10.1016/0168-1923(90)90030-A](https://doi.org/10.1016/0168-1923(90)90030-A)


## 3. Beta Distribution (Goel & Strebel, 1984)

### Theory

Goel and Strebel (1984) proposed using the **Beta probability density** to
represent leaf angle distributions.  The Beta density on the normalized
interval $t = \theta / (\pi/2)$ is

$$
f(t;\,\mu_b,\,\nu_b) = \frac{t^{\mu_b - 1}\,(1 - t)^{\nu_b - 1}}
                              {B(\mu_b,\,\nu_b)}
$$

where $\mu_b$ and $\nu_b$ are shape parameters derived from the desired mean
leaf angle $\mu$ and a concentration parameter $\nu$:

$$
\mu_b = \frac{\mu}{90}\,\nu, \qquad
\nu_b = \left(1 - \frac{\mu}{90}\right)\nu
$$

**Advantages:**

- Can represent any unimodal distribution, including skewed and narrow peaks
- Mathematically well understood, with closed-form moments

**Disadvantages:**

- Two parameters ($\mu$, $\nu$) are less physically intuitive than Campbell's
  single mean-angle
- Cannot naturally produce the bimodal extremophile shape that Verhoef's
  parameterization supports
- Not commonly used in operational retrieval algorithms

### Experiment API

The Beta LAD is implemented in the comparison script but is not part of the
core SCOPE-RTM library:

```python
from examples.leaf_angle_distribution_comparison import beta_lidf

lidf = beta_lidf(mu=45.0, nu=3.0, n_elements=13)
```

### Reference

> Goel, N.S. & Strebel, D.E. (1984). Simple Beta distribution representation
> of leaf orientation in vegetation canopies. *Agronomy Journal*, 76(5),
> 800-802.
> [doi:10.2134/agronj1984.00021962007600050021x](https://doi.org/10.2134/agronj1984.00021962007600050021x)


---

## Experiment results

The tables and figures below were generated by
`examples/leaf_angle_distribution_comparison.py` with SZA = 30 deg,
VZA = nadir, LAI = 3, hotspot = 0.05, and flat soil reflectance = 0.1.

### Extinction coefficients

The extinction coefficients $k_s$ (solar), $k_o$ (observer), and the
bi-Lambertian scattering factor $bf$ quantify how each LAD shapes canopy
light interception.  Higher $k_s$ means faster beam attenuation; higher $bf$
means more isotropic scattering.

![Extinction coefficients by LAD](assets/lad_extinction_coefficients.svg)

#### Verhoef distributions

| Distribution   | a     | b     | ks     | ko     | bf     |
|----------------|-------|-------|--------|--------|--------|
| Planophile     |  1.0  |  0.0  | 0.9638 | 0.9633 | 0.9358 |
| Erectophile    | -1.0  |  0.0  | 0.4025 | 0.1476 | 0.0614 |
| Plagiophile    |  0.0  | -1.0  | 0.7028 | 0.7005 | 0.5000 |
| Extremophile   |  0.0  |  1.0  | 0.6822 | 0.5349 | 0.4982 |
| Spherical      | -0.35 | -0.15 | 0.5684 | 0.4911 | 0.3227 |
| Uniform        |  0.0  |  0.0  | 0.6870 | 0.6374 | 0.5003 |

**Key observations:**

- Planophile leaves intercept nearly all incoming light at this geometry
  (ks = 0.96), while erectophile leaves allow most light through (ks = 0.40).
- The difference in $k_o$ is even more dramatic: erectophile gives
  ko = 0.15 vs. planophile ko = 0.96 --- a 6x difference that strongly
  affects directional reflectance.
- The extremophile is bimodal yet produces intermediate coefficients because
  its two modes partially cancel.

#### Campbell distributions

| Distribution          | alpha  | ks     | ko     | bf     |
|-----------------------|--------|--------|--------|--------|
| alpha=10 (planophile) | 10 deg | 0.9438 | 0.9380 | 0.9022 |
| alpha=30              | 30 deg | 0.7329 | 0.6819 | 0.5653 |
| alpha=57 (spherical)  | 57 deg | 0.4866 | 0.3178 | 0.1909 |
| alpha=70              | 70 deg | 0.3982 | 0.1372 | 0.0528 |
| alpha=85 (erectophile)| 85 deg | 0.3688 | 0.0308 | 0.0026 |

**Key observations:**

- Campbell smoothly interpolates from planophile to erectophile with a single
  parameter.  The transition is nonlinear: most of the ks change happens
  between 10 deg and 57 deg.
- At alpha = 85 deg, $k_o$ drops to 0.03 --- near-vertical leaves are nearly
  invisible to a nadir observer.
- Campbell alpha = 57 deg produces lower ks (0.49) than Verhoef Spherical
  (0.57) because the two parameterizations define "spherical" differently:
  Campbell uses solid-angle uniformity while Verhoef's (a=-0.35, b=-0.15) is
  an empirical approximation.

#### Beta (Goel-Strebel) distributions

| Distribution       | mu     | nu  | ks     | ko     | bf     |
|--------------------|--------|-----|--------|--------|--------|
| Beta mu=20, nu=3   | 20 deg | 3   | 0.8168 | 0.7933 | 0.7002 |
| Beta mu=45, nu=2   | 45 deg | 2   | 0.5899 | 0.4681 | 0.3494 |
| Beta mu=57, nu=4   | 57 deg | 4   | 0.4616 | 0.2859 | 0.1538 |
| Beta mu=75, nu=3   | 75 deg | 3   | 0.3977 | 0.1283 | 0.0514 |

**Key observations:**

- At matching mean angles, the Beta distribution produces similar coefficients
  to Campbell, but the additional $\nu$ parameter controls the peak width.
- Higher $\nu$ (more peaked) at the same mean angle gives slightly lower
  coefficients because the distribution concentrates weight at a single
  angle rather than spreading it.


### Impact on canopy reflectance (BRF)

The figure below shows bidirectional reflectance factor (BRF) in the principal
plane for representative distributions from each family.  Negative VZA is the
forward-scatter direction; positive VZA is the backscatter direction (toward
the sun).

![BRF comparison across LADs](assets/lad_brf_comparison.svg)

**Key observations:**

- **Red band (670 nm):** LAD has a modest effect on absolute reflectance
  (range ~0.015-0.04) but a strong effect on angular shape.  Planophile
  canopies show a more pronounced hotspot peak and higher overall BRF;
  erectophile canopies are nearly flat in the angular domain.

- **NIR band (800 nm):** LAD effects are much larger in absolute terms
  (range ~0.25-0.55).  Planophile canopies reflect much more NIR because
  horizontal leaves create a dense reflective layer near the top; erectophile
  canopies allow NIR to penetrate deeper, increasing absorption and soil
  interaction.

- **Hotspot asymmetry:** All distributions show a reflectance peak near
  VZA = +30 deg (the hotspot, where the observer looks along the solar beam).
  Planophile canopies have the sharpest hotspot; erectophile canopies have
  the weakest because their gap fraction is already high at nadir.

- **Cross-family agreement:** Verhoef Spherical and Campbell alpha=57 produce
  similar angular shapes but differ in magnitude, confirming that their
  "spherical" definitions are not identical.


---

## Comparison: which LAD to use?

| Criterion                          | Verhoef        | Campbell       | Beta           |
|------------------------------------|----------------|----------------|----------------|
| Number of parameters               | 2 (a, b)       | 1 (alpha)      | 2 (mu, nu)    |
| Bimodal distributions              | Yes            | No             | No             |
| Physical interpretability          | Moderate       | High           | Low            |
| PROSAIL compatibility              | Yes (type 1)   | Yes (type 2)   | No             |
| Canonical archetypes (de Wit)      | Direct mapping | Via alpha only  | Indirect       |
| Inversion / retrieval friendliness | Good           | Best (1 param) | Fair           |

**Recommendations:**

- **For PROSAIL intercomparison** or retrieval studies: use `campbell_lidf`
  with `n_elements=18` and the PROSAIL litab grid.  This matches the standard
  PROSAIL parameterization (`TypeLidf=2`) exactly.

- **For SCOPE workflows** or when you need bimodal leaf populations: use
  `scope_lidf` with the canonical Verhoef parameters.  The 13-element
  discretization matches the upstream SCOPE MATLAB code.

- **For research** exploring novel LAD shapes: the Beta distribution offers
  maximum flexibility for unimodal distributions but requires custom
  integration into the model.


## Interoperability between SCOPE and PROSAIL

The `FourSAILModel` accepts any LAD as a `(batch, n_angles)` tensor, so it is
agnostic to the generation method.  The only constraint is that the model's
`n_angles` constructor argument must match the length of the LIDF vector:

```python
from scope import FourSAILModel, campbell_lidf, scope_lidf, scope_litab

# PROSAIL-style: 18 bins
lidf_prosail = campbell_lidf(alpha=57.0, n_elements=18)
sail_prosail = FourSAILModel(n_angles=18)

# SCOPE-style: 13 bins
lidf_scope = scope_lidf(lidfa=-0.35, lidfb=-0.15)
sail_scope = FourSAILModel(n_angles=13)

# Both models can also use the other's LIDF by matching n_elements:
lidf_campbell_13 = campbell_lidf(alpha=57.0, n_elements=13)
# Use with the default FourSAILModel(n_angles=13) -- works seamlessly
```


## Running the comparison experiment

A self-contained comparison script is provided at
`examples/leaf_angle_distribution_comparison.py`:

```bash
# Basic run (prints tables, saves JSON)
python examples/leaf_angle_distribution_comparison.py --no-plot

# Full run with figures (requires matplotlib)
MPLBACKEND=Agg python examples/leaf_angle_distribution_comparison.py \
    --scope-root upstream/SCOPE

# Generate figures into docs/assets for documentation
MPLBACKEND=Agg python examples/leaf_angle_distribution_comparison.py \
    --scope-root upstream/SCOPE --output-dir docs/assets
```

The script produces:

1. **`lad_distributions.svg`** --- Probability vs. inclination angle for all
   15 LAD variants, grouped by family
2. **`lad_extinction_coefficients.svg`** --- Grouped bar chart of $k_s$,
   $k_o$, and $bf$ across all distributions
3. **`lad_brf_comparison.svg`** --- Principal-plane BRF at 670 nm and 800 nm
   for 8 representative distributions
4. **`leaf_angle_distribution_comparison.json`** --- Machine-readable summary
   of all distributions, parameters, and extinction coefficients

## Per-pixel ALA from Sentinel-2 biophysical retrievals

When using prepared input from archetype-based Sentinel-2 biophysical
retrievals, the dataset contains a per-pixel **ALA** (Average Leaf Angle)
variable.  SCOPE Torch now uses this parameter to compute a
**spatially-varying Campbell LIDF** for each pixel, rather than applying a
single global leaf angle across the entire scene.

### Data flow

```
S2 L2A imagery
  --> SNAP Biophysical Processor / archetype retrieval
  --> ALA (degrees per pixel)
  --> prepare.py: dataset["ala"] = post_bio_da.sel(band="ala")
  --> ScopeGridRunner._resolve_lidf(batch, varmap)
  --> campbell_lidf(ala_batch, n_elements=18)  # vectorized
  --> per-pixel LIDF tensor (batch, n_elements)
  --> FourSAILModel / CanopyReflectanceModel / ...
```

### How it works

1. **Data preparation** (`scope.io.prepare`): The `ala` band is extracted from
   the biophysical retrieval product alongside LAI, Cab, etc. and stored as a
   data variable in the xarray Dataset.

2. **Runner dispatch** (`ScopeGridRunner`): At each batch iteration, the runner
   calls `_resolve_lidf(batch, varmap)`.  When `"ala"` is present in both the
   varmap and the batch, a **per-pixel Campbell LIDF** is computed using the
   vectorized `campbell_lidf()`.  When `"ala"` is absent, the runner falls back
   to the static LIDF set at construction time (typically 57 deg, approximating
   a spherical distribution).

3. **Vectorized LIDF** (`campbell_lidf`): The function now accepts both scalar
   and batched tensor inputs.  A tensor of shape \((B,)\) produces a LIDF of
   shape \((B, N)\), where \(N\) is the number of inclination bins.  The
   output is numerically identical to calling the scalar version for each
   element.

### API usage

```python
import torch
from scope import ScopeGridRunner, campbell_lidf

# Static LIDF for construction (fallback when ALA is absent)
lidf = campbell_lidf(57.0, device=device, dtype=dtype)

runner = ScopeGridRunner.from_scope_assets(
    lidf=lidf,
    default_lidfa=57.0,  # documents the intended default
    scope_root_path="upstream/SCOPE",
    device=device,
    dtype=dtype,
)

# When varmap includes "ala", per-pixel LIDF is computed automatically
varmap = {name: name for name in dataset.data_vars}  # includes "ala"
outputs = runner.run_dataset(module, varmap=varmap)
```

### Per-pixel ALA demo

The example script `examples/per_pixel_ala_demo.py` demonstrates the
isolated effect of ALA on canopy reflectance.  Five ALA values (20, 35,
50, 57, 70 deg) are simulated with all other parameters held constant:

| ALA (deg) | Canopy type  | rsot @ 650 nm | rsot @ 865 nm | rsot @ 1600 nm |
|-----------|--------------|---------------|---------------|----------------|
| 20        | Planophile   | 0.0368        | 0.5531        | 0.3001         |
| 35        | Intermediate | 0.0339        | 0.5035        | 0.2735         |
| 50        | Intermediate | 0.0320        | 0.4462        | 0.2460         |
| 57        | Spherical    | 0.0317        | 0.4111        | 0.2313         |
| 70        | Erectophile  | 0.0358        | 0.3230        | 0.2028         |

Key observations:

- **NIR (865 nm)**: Planophile canopies (low ALA) have much higher
  reflectance (0.55) than erectophile (0.32), because horizontal leaves
  create more multiple scattering.
- **Red (650 nm)**: The effect is smaller but still measurable; planophile
  canopies absorb slightly more due to longer path lengths.
- **SWIR (1600 nm)**: Follows the NIR pattern at reduced magnitude.

Run the demo:

```bash
python examples/per_pixel_ala_demo.py --scope-root upstream/SCOPE
```
