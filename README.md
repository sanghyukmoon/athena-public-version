athena
======
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4455880.svg)](https://doi.org/10.5281/zenodo.4455880) <!-- v21.0, not Concept DOI that tracks the "latest" version (erroneously sorted by DOI creation date on Zenodo). 10.5281/zenodo.4455879 -->
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Athena++ GRMHD code and adaptive mesh refinement (AMR) framework

(**2021-05-10**) We have opened the main repository to the public:

https://github.com/PrincetonUniversity/athena

James Open BC self-gravity
======

Implementation of the James algorithm on the public version of Athena++.

Enables self-gravity with open boundary condition in either Cartesian or **cylindrical** coordinates (in 3-D).

Method paper:  **[Moon, Kim, & Ostriker (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...24M/abstract)**.

### Usage                                                                                  
* To activate open BC self-gravity, compile with `--grav=obc -fft -mpi`                     
* Note that the serial version is not implemented; if you want to run with single core, do $mpirun -np 1
