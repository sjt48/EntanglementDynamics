# EntanglementDynamics

This repository contains code used for the project ``Measuring out quasi-local integrals of motion from entanglement" by B. Lu, C. Bertoni, S. J. Thomson and J. Eisert.

The code in this repository makes use of the [QUIMB package](https://quimb.readthedocs.io/en/latest/) for quantum information calculations in many-body quantum systems. The Python script ``entanglement.py" generates the data used in this project. It takes three command line arguments, for the system size, disorder strength and maximum bond dimension respectively. For example, to run a simulation for L=12 at disorder strength d=9 and bond dimension chi=128, the command is the following:

```
python entanglement.py 12 9 128
```

By default, 100 disorder realisations are used, but this can be changed in the script, along with further details such as the timestep and SVD cutoff. Note that QUIMB uses an internal grid of linearly spaced timesteps, but accepts a logarithmically spaced grid of timesteps as an input: time evolution proceeds along the internal linear grid until it reaches a timestep close to one of the desired logarithmic timesteps, at which point QUIMB will automatically adjust the internal timestep to obtain the evolved state at the desired time. Observables are calculated only at logarithmically-spaced timesteps, not at every linear timestep, for more efficient use of computational resources.

The Jupyter notebook ``figures.ipynb" contains the code used to process the data and generate the figures shown in the manuscript. The data corresponding to this project may be found on [Zenodo](https://doi.org/10.5281/zenodo.7322988).
