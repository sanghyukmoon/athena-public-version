athena
======
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Athena++ radiation MHD code

open BC self-gravity module
======

* Includes self-gravity with vacuum (open) boundary condition via James algorithm **[(Moon, Kim, & Ostriker 2019, ApJS, 241, 24)](http://adsabs.harvard.edu/abs/2019ApJS..241...24M)**
* Supports Cartesian and cylindrical coordinates.
* Self gravity is added as a momentum flux in Cartesian coordinates and as a source term in cylindrical coordinates (to be updated).

#### Usage
compile with `--grav=obc -fft`
