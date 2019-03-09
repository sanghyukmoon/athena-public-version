//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file poly_equil.cpp
//  \brief equilibrium of n=1 polytrope

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../fft/athena_fft.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../gravity/fftgravity.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/mggravity.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../multigrid/multigrid.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#if !SELF_GRAVITY_ENABLED
#error "This problem generator requires self-gravity"
#endif

#if !NON_BAROTROPIC_EOS
#error "This problem generator requires a non-barotropic EOS"
#endif

void Mesh::InitUserMeshData(ParameterInput *pin) {
  Real G = pin->GetReal("problem","grav_const");
  SetFourPiG(4.0*PI*G);
  SetMeanDensity(0.0);
  return;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // constants
  Real G     = pin->GetReal("problem","grav_const");
  Real gm1   = peos->GetGamma() - 1.0;

  // ambient density and pressure
  Real damb  = pin->GetReal("problem", "damb");
  Real pamb  = pin->GetReal("problem", "pamb");

  // center coordinates
  Real x1c   = pin->GetReal("problem","x1c");
  Real x2c   = pin->GetReal("problem","x2c");
  Real x3c   = pin->GetReal("problem","x3c");

  // polytrope structure
  Real rhoc  = pin->GetReal("problem","rhoc");
  Real pc    = pin->GetReal("problem","pc");

  // polytrope radius
  Real rsurf = pin->GetReal("problem","rsurf");

  // polytrope setup
  Real den, pres, v1, v2, v3;
  for (int k=ks; k<=ke; ++k) {
  for (int j=js; j<=je; ++j) {
  for (int i=is; i<=ie; ++i) {
    Real x1 = pcoord->x1v(i);
    Real x2 = pcoord->x2v(j);
    Real x3 = pcoord->x3v(k);

    Real rad   = std::sqrt(SQR(x1-x1c)
                         + SQR(x2-x2c)
                         + SQR(x3-x3c));

    if (rad < rsurf) {
      den   = (std::sqrt(pc)*std::sin((std::sqrt(G)
             * std::sqrt(2.*PI)*rad*rhoc)
             / std::sqrt(pc)))
             /(std::sqrt(G)*std::sqrt(2.*PI)*rad);
      pres  = pc/pow(rhoc,2.0)*pow(den, 2.0);

      v1 = pin->GetReal("problem","v1");
      v2 = pin->GetReal("problem","v2");
      v3 = pin->GetReal("problem","v3");
    } else { // atmosphere
      den  = damb;
      pres = pamb;

      v1 = 0.0;
      v2 = 0.0;
      v3 = 0.0;
    }

    //set conserved variables
    phydro->u(IDN,k,j,i) = den;
    phydro->u(IEN,k,j,i) = pres/gm1;
    phydro->u(IM1,k,j,i) = den*v1;
    phydro->u(IM2,k,j,i) = den*v2;
    phydro->u(IM3,k,j,i) = den*v3;

    phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i))
                               + SQR(phydro->u(IM2,k,j,i))
                               + SQR(phydro->u(IM3,k,j,i)))
                               / phydro->u(IDN,k,j,i);
  }}}
}
