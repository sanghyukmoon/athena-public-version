//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cylgrav.cpp
//  \brief Problem generator for cylindrical Poisson solver. 
// Problem generator for test problems for the cylindrical Poisson solver.
// TODO Description for the test problems goes here.
//========================================================================================

// C++ headers
#include <iostream>  // cout, endl
#include <iomanip>   // setprecision, scientific
#include <sstream>
#include <cmath>
#include <ctime>
#include <mpi.h>
// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../mesh/mesh.hpp"
#include "../defs.hpp"
#include "../hydro/hydro.hpp"
#include "../field/field.hpp"
#include "../gravity/gravity.hpp"
#include "../gravity/obcgravity.hpp"
#include "../gravity/fftgravity.hpp"
#include "../gravity/mggravity.hpp"
#include "../eos/eos.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"
#include "../fft/plimpton/remap_3d.h"
#include "../fft/plimpton/remap_2d.h"
#include "../fft/plimpton/pack_3d.h"
#include "../fft/plimpton/pack_2d.h"


#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void InnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void InnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void OuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  Real four_pi_G = pin->GetReal("problem","four_pi_G");
  Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
  SetFourPiG(four_pi_G);
  SetGravityThreshold(eps);
  SetMeanDensity(0.0);

//  if(mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(INNER_X1, InnerX1);
//  }
//  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(OUTER_X1, OuterX1);
//  }
//  if(mesh_bcs[INNER_X2] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(INNER_X2, InnerX2);
//  }
//  if(mesh_bcs[OUTER_X2] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(OUTER_X2, OuterX2);
//  }
//  if(mesh_bcs[INNER_X3] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(INNER_X3, InnerX3);
//  }
//  if(mesh_bcs[OUTER_X3] == GetBoundaryFlag("user")) {
//    EnrollUserBoundaryFunction(OUTER_X3, OuterX3);
//  }
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  Real gamma = peos->GetGamma();
  Real gm1 = gamma - 1.0;
  if (gamma != 2) {
    std::stringstream msg;
    msg << "### FATAL ERROR in polytrope.cpp ProblemGenerator" << std::endl
        << "invalid gamma " << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  Real four_pi_G = pin->GetReal("problem", "four_pi_G");
  Real rhoc  = pin->GetReal("problem","rhoc");
  Real damb  = pin->GetReal("problem","damb");
  Real pamb  = pin->GetReal("problem","pamb");
  Real rsurf = pin->GetReal("problem","rsurf");
  Real Pc = four_pi_G*SQR(rhoc*rsurf)/SQR(PI)/2;
  Real xi,x,y,z,den,prs;
  Real vx=0, vy=0, vz=0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        x = pcoord->x1v(i);
        y = pcoord->x2v(j);
        z = pcoord->x3v(k);
        xi = PI*sqrt(SQR(x)+SQR(y)+SQR(z))/rsurf;
        if (xi < PI) {
          den = rhoc*sin(xi)/xi;
          prs = Pc*SQR(sin(xi)/xi);
        }
        else {
          den = damb;
          prs = pamb;
        }
        phydro->u(IDN,k,j,i) = den;
        phydro->u(IM1,k,j,i) = phydro->u(IDN,k,j,i)*vx;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*vy;
        phydro->u(IM3,k,j,i) = phydro->u(IDN,k,j,i)*vz;
        phydro->u(IEN,k,j,i) = 0.5*phydro->u(IDN,k,j,i)*(SQR(vx)+SQR(vy)+SQR(vz))
          + prs/gm1;
      }
    }
  }
  return;
}
