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

This folk implements the James algorithm based on [Moon, Kim, & Ostriker (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJS..241...24M/abstract) to enable self-gravity with open boundary conditions in three-dimensional Cartesian or cylindrical coordinates.

### Usage                                                                                  
* To activate open BC self-gravity, compile with `--grav=obc -fft -mpi`                     
* Serial version is not implemented. Compile with `-mpi` flag even when you use just a single core.

### Restrictions
Because the current parallel FFT interface requires Mesh to be evenly divisible for both the block and pencil decompositions, the solver may not work for certain number of cells or MeshBlock decompositions. In addition, the James algorithm involves FFTs acting only on surfaces, which add complications on the possible decompositions. **The easiest way to meet all the requirements is to set the number of cells and MeshBlocks in each direction as powers of two.**
