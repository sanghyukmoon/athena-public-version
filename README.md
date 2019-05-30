athena
======
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Athena++ radiation MHD code

Open BC self-gravity module
======
* This is an open BC self-gravity module built on the [public Athena++ code](https://github.com/PrincetonUniversity/athena-public-version)
* Activates self-gravity with vacuum (open) boundary condition in Cartesian and cylindrical coordinates.
* Self-gravitational acceleration is added as a momentum flux in Cartesian coordinates and as a source term in cylindrical coordinates (In the future update, self gravity will be added as a momentum flux in cylindrical coordinates as well).
* If you use this module in a publication, please cite **[Moon, Kim, & Ostriker (2019)](http://adsabs.harvard.edu/abs/2019ApJS..241...24M)**.

#### Usage
* For using Athena++ code, consult [Athena++ Wiki](https://github.com/PrincetonUniversity/athena-public-version/wiki)
* To activate open BC self-gravity, compile with `--grav=obc -fft`
