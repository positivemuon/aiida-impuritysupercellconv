# aiida-musConv

#1-->aiida_musConv/workflows/musConv.py

This script contains the musConvworkschain class to be used in [AiiDA](www.aiida.net)
for generating converged supercell structure and/or its transformation matrix. It does this
by performing supercell size convergence checks against unrelaxed atomic forces; SCF DFT calculations
are done with the [Quantum-Espresso code using its aiida plugin](https://aiida-quantumespresso.readthedocs.io/en/latest/).

To run the aiida-musConvworkschain, [musConv](https://github.com/positivemuon/musConv) or on musConv [pypi](https://pypi.org/project/musConv/0.0.1/) page, aiida-core, plugin installations and aiida-quantum espresso code and computer setups are required.
If everything is up and the musConv package installed. It can be run using the following example
(caveat: labels of code and pseudo have to be edited);

```python examples/run_aiidamusconv_Si_LiF.py```



#Available Workflows
```
aiida_musConv/
└── workflows
    ├── __init__.py
    └── musConv.py
```

#Installation 
install this repository as: 

```
git clone https://github.com/positivemuon/aiida-musConv.git
cd aiida-musConv/
pip install -e .
```

#********
