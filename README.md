## Manifold Dimensional Expansion (MDE)
---
Manifold dimensional expansion is a dimensionality reduction technique designed to identify low dimensional maximally predictive observables from a high dimensional dynamical system. 

The code is based on [pyEDM](https://pypi.org/project/pyEDM/). 

The algorithm is based on a greedy implementation of the generalized Takens embedding theorem. However, instead of using time delays for dimensionality expansion observables that improve the forecast skill of a target variable are added until no further improvement can be achieved. Forecasts are made from [simplex](https://www.nature.com/articles/344734a0) cross prediction. 

Specifically, given a target observable, scan all other observables to find the best 1-D predictor of the target, ensuring the predictor has causal inference with the target. With this 1-D vector scan all remaining observables to find the 2-D embedding with best predictability and causal inference. This greedy algorithm is iterated up to the point that no further prediction skill improvement can be produced. 

Causal inference is optionally performed with Convergent Cross Mapping ([CCM](https://science.sciencemag.org/content/338/6106/496)) ensuring the added observable is part of the dynamical system of the interrogated time series. There is also an option to add time delay vectors of the top observables to account for unobserved behaviors. 

Output is a ranked list of observation vectors and predictive skill satisfying MDE criteria for the target variable.

---
## Installation

### Python Package pyEDM
MDE is built on pyEDM hosted on PyPI [pyEDM](https://pypi.org/project/pyEDM/) and can be installed via pip: `python -m pip install pyEDM`

---
## Usage
MDE is an object-oriented class implementation with command line interface (CLI) support. CLI parameters are configured through command line arguments, MDE class arguments through the constuctor API. 

CLI example:
```
./ManifoldDimExpand.py -d ../data/Fly80XY_norm_1061.csv -rc index FWD Left_Right -D 10 -t FWD -l 1 300 -p 301 600 -C 10 -ccs 0.01 -emin 0.5 -P -title "MDE FWD" -v
```

MDE class constructor API example:
```python
from MDE import MDE
from pandas import read_csv

df = read_csv( '../data/Fly80XY_norm_1061.csv' )

mde = MDE( df, target = 'FWD', 
           removeColumns = ['index','FWD','Left_Right'], 
           D = 10, lib = [1,300], pred = [301,600], 
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
7      TS22  0.856512
8      TS67  0.858537
9      TS13  0.859161
```
