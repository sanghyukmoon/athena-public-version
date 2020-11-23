athena
======
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Athena++ radiation MHD code

Open BC self-gravity module
======
* This branch implements the James algorithm for self-gravity with open (vacuum) boundary condition in Cartesian and cylindrical coordinates.
* **WARNING: this module is built on Athena++ v1.1.1 released on Aug 2018.**
* In Cartesian coordinates, the gravitational force is added as a momentum flux.
* In cylindrical coordinates, the gravitational force is added as a source term.
* If you use this module in publications, please kindly cite **[Moon, Kim, & Ostriker (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...24M/abstract)**.

#### Usage
* To activate open BC self-gravity, compile with `--grav=obc -fft`

#### Test problem
* $./configure.py --prob=cylgrav_test --coord=cylindrical --grav=obc -fft -mpi
* $mpirun -np 4 bin/athena -i inputs/hydro/athinput.cylgrav
