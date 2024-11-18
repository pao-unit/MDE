## Manifold Dimensional Expansion (MDE)
---
Manifold dimensional expansion is a dimensionality reduction technique designed to identify low dimensional maximally predictive observables in a high dimensional dynamical system. 

The code is based on [pyEDM](https://github.com/SugiharaLab/pyEDM?tab=readme-ov-file#empirical-dynamic-modeling-edm). 

The algorithm is based on a greedy implementation of the generalized Takens embedding theorem that instead of using time delays for dimensionality expansion, employs real observables that improve the forecast skill of the target variable until no further improvement can be achieved. 

Specifically, given a target observable, scan all other observables to find the best 1-D predictor of the target, ensuring the predictor has causal inference with the target.  With this 1-D vector scan all remaining observables to find the 2-D embedding with best predictability and causal inference. This greedy algorithm is iterated up to the point that no further prediction skill improvement can be produced. Causal inference is performed with Convergent Cross Mapping (CCM) ensuring the added observable is part of the dynamical system of the interrogated time series.

Output is a ranked list of observation vectors and predictive skill satisfying MDE criteria for the target variable.

---
## Installation

### Python Package pyEDM
MDE is built on pyEDM hosted on PyPI [pyEDM](https://pypi.org/project/pyEDM/) which can be installed with pip: `python -m pip install pyEDM`

---
## Usage
MDE is a python loadable function module and application with command line interface (CLI) support. All parameters are configured through command line arguments. 

CLI example:
```
./ManifoldDimExpand.py -d ../data/Fly80XY_norm_1061.csv -rc index FWD Left_Right -D 10 -t FWD -l 1 300 -p 301 600 -C 10 -ccs 0.01 -emin 0.5 -P -title "MDE FWD"
```

API example:
```python
from ManifoldDimExpand import ManifoldDimExpand
from pandas import read_csv

df = read_csv( '../data/Fly80XY_norm_1061.csv' )

MDE_df = ManifoldDimExpand( df, target = 'FWD', 
                            removeColumns = ['index','FWD','Left_Right'],
                            D = 10, lib = [1,300], pred = [301,600], 
                            cores = 10, plot = True )
MDE_df
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
