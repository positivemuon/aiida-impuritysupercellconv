# aiida-impuritysupercellconv

An [AiiDA](www.aiida.net) workflow plugin that allows to obtain  converged supercell size and or the supercell transformation matrix for an interstitial impurity calculation. This is obtained by converging the supercell size with respect to the induced forces at host atomic by an intersitial impurity at a Voronoi site. Forces are obtained from one shot SCF DFT calculations with  the [Quantum-Espresso code using its aiida plugin](https://aiida-quantumespresso.readthedocs.io/en/latest/).

**Please note**: the code supports Quantum ESPRESSO versions higher or equal than v7.1 .

## Dependencies
To run the aiida-IsolatedImpurityWorkChain, [impuritysupercellconv](https://github.com/positivemuon/impuritysupercellconv) or on impuritysupercellconv [pypi](https://pypi.org/project/impuritysupercellconv/0.0.1/) page, aiida-core, plugin installations and aiida-quantum espresso code and computer setups are required.


## Available Workflows
```
aiida_impuritysupercellconv/
└── workflows
    ├── __init__.py
    └── impuritysupercellconv.py
```

## Installation
install this repository as:

```
git clone https://github.com/positivemuon/aiida-impuritysupercellconv.git
cd aiida-impuritysupercellconv/
pip install -e .
```

## Run the workflow following the example as;
```
cd examples/
python run_aiidaimpuritysupercellconv_Si_LiF.py
```
* (caveat: labels of code and pseudo have to be edited)

## Acknowledgements
We acknowledge support from:
* the [NCCR MARVEL](http://nccr-marvel.ch/) funded by the Swiss National Science Foundation;
* the PNRR MUR project [ECS-00000033-ECOSISTER](https://ecosister.it/);

<img src="https://raw.githubusercontent.com/positivemuon/aiida-impuritysupercellconv/main/docs/source/images/MARVEL_logo.png" width="250px" height="131px"/>
<img src="https://raw.githubusercontent.com/positivemuon/aiida-impuritysupercellconv/main/docs/source/images/ecosister_logo.png" width="300px" height="84px"/>
