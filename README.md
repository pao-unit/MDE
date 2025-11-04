## Manifold Dimensional Expansion (MDE)
---
Manifold dimensional expansion is a causal discovery and dimensionality reduction technique designed to identify low dimensional maximally predictive _observables_ of a dynamical system with multivariate observations.

The algorithm is based on a greedy implementation of the generalized Takens embedding theorem. However, instead of using time delays for dimensionality expansion, _observables_ that improve the forecast skill of a target variable are added until no further improvement can be achieved. The default predictor is the [simplex](https://www.nature.com/articles/344734a0) function in [pyEDM](https://pypi.org/project/pyEDM/) providing a fully nonlinear predictor from [Empirical Dynamic Modeling (EDM)](https://en.wikipedia.org/wiki/Empirical_dynamic_modeling). 

Specifically, given a target observable, scan all other observables to find the best 1-D predictor of the target, ensuring the predictor has causal inference with the target. With this 1-D vector scan all remaining observables to find the 2-D embedding with best predictability and causal inference. This greedy algorithm is iterated up to the point that no further prediction skill improvement can be produced. 

Causal inference is performed by default with Convergent Cross Mapping ([CCM](https://science.sciencemag.org/content/338/6106/496)) ensuring the added observable is part of the dynamical system of the interrogated time series. The embedding dimension needed for CCM is automatically determined if parameter `E=0`, the default. Otherwise the specifed value of `E` is used. To account for unobserved variables time delay vectors of the top observables can be added.

Output is a DataFrame with a ranked list of observation vectors and predictive skill satisfying MDE criteria for the target variable.

---
## Installation

`python -m pip install dimx`

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
---
## Real World Example
This example finds optimal observables and estimates the dimension of neural data from _Drosophila melanogaster_. 

A fly expressing the calcium indicator GCaMP6f as a measure of neuronal activity was recorded walking on a Styrofoam ball. Neuronal activity across the brain was spatially segmented by independent component analysis (ICA) yielding 80 time series of neural activity from the component brain areas. Two behavioral variables were simultaneously recorded: forward speed (FWD) and left/right turning speed (Left_Right) [Aimon, S. et al. 2019](https://doi.org/10.1371/journal.pbio.2006732). A Jupyter Lab notebook is available at [MDE_Fly_Example](https://github.com/pao-unit/MDE/blob/b0eaca5bebc947676498057c679f4d3864909656/example/MDE_Fly_Example.ipynb).

### Import MDE and Evaluate classes
```python
from dimx import MDE, Evaluate
```
### Load data
```python
from pandas import read_csv
df = read_csv( './data/Fly80XY_norm_1061.csv' )
```
#### Instantiate and Run MDE class objects for FWD & Left_Right targets
We use the first 300 time series rows to create the EDM library, and perform out-of-sample prediction on time series rows 301-600. These indices are Not zero offset. 

`ccmSlope` is the minimum slope of a linear fit to the CCM rho(L) curve to validate a causal driver. L is the vector of CCM library sizes at which CCM is evaluated. Default values for L are percentiles [10,15,85,90] of the number of observations (rows).

```python
Fly_FWD = MDE( df,                     # Pandas DataFrame of observables
               target = 'FWD',         # target behavior variable
               removeColumns = ['index','FWD','Left_Right'], # variables to ignore
               D    = 12,              # Max number of dimensions
               lib  = [1,300],         # EDM library start,stop indices
               pred = [301,600],       # EDM prediction start,stop indices
               ccmSlope = 0.01,        # CCM convergence criteria
               embedDimRhoMin = 0.65,  # Minimum rho for CCM embedding dimension
               crossMapRhoMin = 0.5,   # Minumum rho for cross map of target : variables
               cores = 10,             # Number of cores in CrossMapColumns()
               chunksize = 30,
               plot = False )
Fly_FWD.Run()
```

```python
Fly_LR = MDE( df,
              target = 'Left_Right',
              removeColumns = ['index','FWD','Left_Right'], 
              D    = 12,
              lib  = [1,600],
              pred = [801,1000], 
              ccmSlope = 0.01,
              embedDimRhoMin = 0.2,
              crossMapRhoMin = 0.05,
              cores = 10,
              chunksize = 30,
              plot = False )
Fly_LR.Run()
```

The FWD behavior suggests a dimension of D=5 observables is an appropriate low-dimensional set of obervables to predict FWD movement. 
![MDE Fly](/example/MDE_Fly.png)

---
#### Evaluate MDE components & compare to PCA & Diffusion Map
Here we compare out-of-sample prediction of FWD behavior with the 5 MDE identified observables as well as 5 component PCA and Diffusion Map. 

```python
Fly_FWD_Eval = Evaluate( df, 
                         columns_range = [1,81], # 0-offset range of columns for PCA, DMap
                         mde_columns = ['TS33', 'TS4', 'TS8', 'TS9', 'TS32'],
                         predictVar = 'FWD',
                         library    = [1, 300],   # index start,stop of observations for library
                         prediction = [301, 600], # index start,stop of predictions 
                         components = 5,          # Number of PCA & DMap components
                         dmap_k     = 15,         # diffusion_map k nearest neighbors
                         figsize    = (8,6) )
Fly_FWD_Eval.Run()
Fly_FWD_Eval.Plot()
```
The MDE prediction has the lowest CAE (cumulative absolute error) to the out-of-sample observations. The diffusion map compoents are latent (not observable) and do not correspond in an obvious way to observed neural dynamics. The PCA prediction lumps the majority of the variance into a single component based on a linear decomposition. Both PCA and diffusion map predict activity during times when no FWD movement is observed, while MDE does not. Crucially, MDE predictions are not latent, but actual _observables_ of the system.

![MDE Evaluate](/example/Evaluate_Fly.png)

---
## Parameters
MDE parameters are defined in the MDE constructor. 

| Parameter | Default | Description |
|---|---|---|
| dataFrame | None | Pandas DataFrame : column observation vectors, row observations |
| dataFile | None | Data file name to load |
| dataName | None | Data name in .npz archive |
| removeTime | False | Remove first column from dataFrame |
| noTime | False | First column of dataFrame is not time vector |
| columnNames | [] | dataFrame columns to process |
| initDataColumns | [] | If reading .npz omit these leading columns |
| removeColumns | [] | Columns to ignore |
| D | 3 | Maximum number of MDE dimensions |
| target | None | Target variable |
| lib | [] | EDM library indices. Default to all rows |
| pred | [] | EDM prediction indices. Default to all rows  |
| Tp | 1 | Prediction time interval |
| tau | -1 | CCM embedding time delay |
| exclusionRadius | 0 | CCM library temporal exlcusion radius |
| sample | 20 | CCM random library samples to average |
| pLibSizes | [10, 15, 85, 90] | Percentiles of CCM library sizes |
| noCCM | False | Disable CCM |
| ccmSlope | 0.01 | Slope of CCM(LibSizes) convergence |
| ccmSeed | None | CCM random seed |
| E | 0 | CCM embedding dimension. If 0 compute automatically |
| crossMapRhoMin | 0.5 | Minimum rho for cross map acceptance |
| embedDimRhoMin | 0.5 | Minimum rho for CCM embedding dimension |
| maxE | 15 | Maximum embedding dimension for CCM |
| firstEMax | False | CCM embedding dimension is first local peak in rho(E) |
| timeDelay | 0 | Add N=timeDelay time delays |
| cores | 5 | number of multiprocessing CPU in CrossMapColumns() |
| mpMethod | None | multiprocessing context method in CrossMapColumns() |
| chunksize | 1 | multiprocessing chunksize in CrossMapColumns() |
| outDir | None | Output file directory |
| outFile | None | MDE object pickle file |
| outCSV | None | CSV of MDE output |
| logFile | None | Log file |
| consoleOut | True | Echo output to console |
| verbose | False | Verbose mode |
| debug | False | Debug mode |
| plot | False | Plot MDE result |
| title | None | Plot title |
| args | None | ArgumentParser object from CLI_Parser |
