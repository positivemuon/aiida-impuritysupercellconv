# aiida-musConv

An [AiiDA](www.aiida.net) workflow plugin that allows to obtain  converged supercell size and or the supercell transformation matrix for an interstitial impurity calculation. This is obtained by converging the supercell size with respect to the induced forces at host atomic by an intersitial impurity at a Voronoi site. Forces are obtained from one shot SCF DFT calculations with  the [Quantum-Espresso code using its aiida plugin](https://aiida-quantumespresso.readthedocs.io/en/latest/).


## Depend
To run the aiida-musConvworkschain, [musConv](https://github.com/positivemuon/musConv) or on musConv [pypi](https://pypi.org/project/musConv/0.0.1/) page, aiida-core, plugin installations and aiida-quantum espresso code and computer setups are required.
If everything is up and the musConv package installed. It can be run using the following example
(caveat: labels of code and pseudo have to be edited);




## Available Workflows
```
aiida_musConv/
└── workflows
    ├── __init__.py
    └── musConv.py
```

## Installation 
install this repository as: 

```
git clone https://github.com/positivemuon/aiida-musConv.git
cd aiida-musConv/
pip install -e .
```

## Run the workflow as;
```
cd examples/
python run_aiidamusconv_Si_LiF.py
```
* (caveat: labels of code and pseudo have to be edited)


