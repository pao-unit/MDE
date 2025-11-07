## Manifold Dimensional Expansion (MDE)
---
Manifold dimensional expansion is a causal discovery and dimensionality reduction technique designed to identify low dimensional maximally predictive _observables_ of a dynamical system with multivariate observations.

The algorithm is based on a greedy implementation of the [generalized](https://doi.org/10.1371%2Fjournal.pone.0018295) Takens embedding theorem. However, instead of using time delays for dimensionality expansion, _observables_ that improve the forecast skill of a target variable are added until no further improvement can be achieved. The default predictor is the [simplex](https://www.nature.com/articles/344734a0) function in [pyEDM](https://pypi.org/project/pyEDM/) providing a fully nonlinear predictor from [Empirical Dynamic Modeling (EDM)](https://en.wikipedia.org/wiki/Empirical_dynamic_modeling). 

Specifically, given a target observable, scan all other observables to find the best 1-D predictor of the target, ensuring the predictor has causal inference with the target. With this 1-D vector scan all remaining observables to find the 2-D embedding with best predictability and causal inference. This greedy algorithm is iterated up to the point that no further prediction skill improvement can be produced. 

Causal inference is performed by default with Convergent Cross Mapping ([CCM](https://science.sciencemag.org/content/338/6106/496)) ensuring the added observable is part of the dynamical system of the interrogated time series. The embedding dimension needed for CCM is automatically determined if parameter `E=0`, the default. Otherwise the specifed value of `E` is used. To account for unobserved variables time delay vectors of the top observables can be added.

Output is a DataFrame with a ranked list of observation vectors and predictive skill satisfying MDE criteria for the target variable.

---

## Installation

`python -m pip install dimx`

## Documentation
Documentation is available at [MDE Docs](https://pao-unit.github.io/MDE_docs/)

---

## Usage
MDE is an object-oriented class implementation with command line interface (CLI) support. CLI parameters are configured through command line arguments, MDE class arguments through the MDE class constuctor.

MDE can be imported as a module and executed with `dimx.Run()` or from the command line with the`ManifoldDimExpand.py` executable as shown below.

CLI example:
```
./ManifoldDimExpand.py -d ../data/Fly80XY_norm_1061.csv 
-rc index FWD Left_Right -D 10 -t FWD -l 1 300 -p 301 600
-C 10 -ccs 0.01 -emin 0.5 -P -title "MDE FWD" -v
```

MDE class constructor API example:
```python
from dimx import MDE
from pandas import read_csv

df = read_csv( './data/Fly80XY_norm_1061.csv' )

mde = MDE( df, target = 'FWD', 
           removeColumns = ['index','FWD','Left_Right'], 
           D = 10, lib = [1,300], pred = [301,600], ccmSeed = 12345,
           cores = 10, plot = True, title = "MDE FWD" )

mde.Run()

mde.MDEOut
  variables       rho
0      TS33  0.652844
1       TS4  0.792290
2      TS17  0.823024
3      TS71  0.840094
4      TS44  0.840958
5      TS37  0.845765
6       TS9  0.846601
7      TS30  0.859614
8      TS47  0.860541
9      TS67  0.860230
```
