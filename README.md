athena
======
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Athena++ radiation MHD code

open BC self-gravity module
======

* Includes self-gravity with vacuum (open) boundary condition via James algorithm **(Moon, Kim, & Ostriker, in press., https://arxiv.org/abs/1902.08369 )**
* Supports Cartesian and cylindrical coordinates.
* Currently (cylindrical coordinates only), gravitational acceleration is added as a source term, rather than as a momentum flux using gravitational tensor.

#### Usage
compile with `--grav=obc -fft`
