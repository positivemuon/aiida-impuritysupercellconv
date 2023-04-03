# aiida-musConv

#1-->aiida_musConv/workflows/musConvWorkChain.py

This script contains the musConvworkschain class to be used in [AiiDA](www.aiida.net)
for generating converged supercell structure and/or its transformation matrix. It does this
by performing supercell size convergence checks against unrelaxed atomic forces; SCF DFT calculations
are done with the [Quantum-Espresso code using its aiida plugin](https://aiida-quantumespresso.readthedocs.io/en/latest/).

To run the the aiida-musConvworkschain, aiida-core, plugin installations and setups are required.
If everything is up and the musConv package installed. It can be run using the following example
(caveat: labels of code and pseudo have to be edited);

```python examples/run_aiidamusconv_Si_LiF.py```



#--------------------------------------------------------------------------------------------------------
