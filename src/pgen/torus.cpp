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
#include <fstream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <mpi.h>
#include "fftw3.h"
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
#include "../eos/eos.hpp"
#include "../coordinates/coordinates.hpp"
#include "../globals.hpp"

#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================
static AthenaArray<Real> initdens;
static AthenaArray<Real> initvel2;
static AthenaArray<Real> initprs;
static Real omg, omg0;

void DiskInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void DiskOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,FaceField &b,
  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh);
void ExternalGravity(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  Real four_pi_G = pin->GetReal("problem","four_pi_G");
  Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
  SetFourPiG(four_pi_G);
  SetGravityThreshold(eps);
  SetMeanDensity(0.0);
  if(mesh_bcs[INNER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X1, DiskInnerX1);
  }
  if(mesh_bcs[OUTER_X1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X1, DiskOuterX1);
  }
  if(mesh_bcs[INNER_X3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(INNER_X3, DiskInnerX3);
  }
  if(mesh_bcs[OUTER_X3] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(OUTER_X3, DiskOuterX3);
  }
  EnrollUserExplicitSourceFunction(ExternalGravity);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  initdens.NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  initvel2.NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  initprs.NewAthenaArray(block_size.nx3+2*NGHOST, block_size.nx2+2*NGHOST, block_size.nx1+2*NGHOST);
  omg  = pin->GetReal("problem","omg");
  omg0 = pin->GetReal("problem","omg0");
  Real alpha2 = pin->GetReal("problem","alpha2");
  Real four_pi_G = pin->GetReal("problem","four_pi_G");
  int mperturb = pin->GetInteger("problem","mperturb");
  Real gm1 = pin->GetReal("hydro","gamma") - 1.0;
  // open initial condition files
  std::ifstream scf_dens_file ("scf_dens.txt"); 
  std::ifstream scf_prs_file ("scf_prs.txt"); 
  std::ifstream scf_r_file ("scf_r.txt"); 
  std::ifstream scf_mu_file ("scf_mu.txt"); 
  std::ifstream scf_phi_file ("scf_Phi.txt"); 
  if (!scf_dens_file.is_open()) {std::cout << "error opening file\n"; return;}
  if (!scf_prs_file.is_open()) {std::cout << "error opening file\n"; return;}
  if (!scf_r_file.is_open()) {std::cout << "error opening file\n"; return;}
  if (!scf_mu_file.is_open()) {std::cout << "error opening file\n"; return;}
  if (!scf_phi_file.is_open()) {std::cout << "error opening file\n"; return;}
  AthenaArray<Real> scf_dens;
  AthenaArray<Real> scf_prs;
  AthenaArray<Real> scf_r;
  AthenaArray<Real> scf_mu;
  AthenaArray<Real> scf_phi;
  scf_dens.NewAthenaArray(2049, 2049); // in R, cos(theta) plane
  scf_prs.NewAthenaArray(2049, 2049);
  scf_r.NewAthenaArray(2049);
  scf_mu.NewAthenaArray(2049);
  scf_phi.NewAthenaArray(2049, 2049);
  // read from files
  for (int i=0;i<2049;++i) {
    for (int j=0;j<2049;++j) {
      scf_dens_file >> scf_dens(i,j);
      scf_prs_file >> scf_prs(i,j);
      scf_phi_file >> scf_phi(i,j);
    }
    scf_r_file >> scf_r(i);
    scf_mu_file >> scf_mu(i);
  }
  scf_dens_file.close();
  scf_prs_file.close();
  scf_r_file.close();
  scf_mu_file.close();
  scf_phi_file.close();
  int il, iu, jl, ju, im, jm;
  Real R, z, r, mu, t, u;
  for (int k=ks-NGHOST; k<=ke+NGHOST; ++k) {
    for (int j=js-NGHOST; j<=je+NGHOST; ++j) {
      for (int i=is-NGHOST; i<=ie+NGHOST; ++i) {
        R = pcoord->x1v(i);
        z = fabs(pcoord->x3v(k));
        r = sqrt(SQR(R)+SQR(z));
        mu = z/r;
        il = 0;
        iu = 2048;
        jl = 0;
        ju = 2048; 
        while (iu - il > 1) {
          im = (iu + il) >> 1;
          if (r > scf_r(im)) il = im;
          else iu = im;
        }
        while (ju - jl > 1) {
          jm = (ju + jl) >> 1;
          if (mu > scf_mu(jm)) jl = jm;
          else ju = jm;
        }
        t = (r - scf_r(il)) / (scf_r(iu) - scf_r(il));
        u = (mu - scf_mu(jl)) / (scf_mu(ju) - scf_mu(jl));
        // bilinear interpolation
        phydro->u(IDN,k,j,i) = (1.-t)*(1.-u)*scf_dens(il,jl) + t*(1.-u)*scf_dens(iu,jl)
                              + (1.-t)*u*scf_dens(il,ju) + t*u*scf_dens(iu,ju);
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = phydro->u(IDN,k,j,i)*R*omg0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = (1.-t)*(1.-u)*scf_prs(il,jl) + t*(1.-u)*scf_prs(iu,jl)
                              + (1.-t)*u*scf_prs(il,ju) + t*u*scf_prs(iu,ju);
        phydro->u(IEN,k,j,i) = phydro->u(IEN,k,j,i)/gm1 + 0.5*phydro->u(IDN,k,j,i)*SQR(R*omg0);       
        initdens(k,j,i) = phydro->u(IDN,k,j,i);
        initvel2(k,j,i) = R*omg0;
        initprs(k,j,i) = gm1*(phydro->u(IEN,k,j,i) - 0.5*phydro->u(IDN,k,j,i)*SQR(R*omg0));
        // add single mode perturbation
        if (mperturb != 0)
          phydro->u(IDN,k,j,i) += ((1e-3*phydro->u(IDN,k,j,i))*sin(mperturb*pcoord->x2v(j)));
      }
    }
  }

  // add white noise perturbation
//  int N = 1024;
//  int hN = 512;
//  fftw_complex *in =  fftw_alloc_complex(hN+1);
//  fftw_plan c2r = fftw_plan_dft_c2r_1d(N, in, (double*)in, FFTW_MEASURE);
//  in[0][0] = 0;
//  in[0][1] = 0;
//  in[hN][0] = 1;
//  in[hN][1] = 0;
//  double phase;
//  for (int j=1;j<hN;++j) {
//    phase = 2*PI*(double)rand()/RAND_MAX;
//    in[j][0] = cos(phase);
//    in[j][1] = sin(phase);
//  }
//  fftw_execute(c2r);
//
//  for (int k=ks; k<=ke; ++k) {
//    for (int j=js; j<=je; ++j) {
//      for (int i=is; i<=ie; ++i) {
//        phydro->u(IDN,k,j,i) += ((1e-5*phydro->u(IDN,k,j,i))*((double*)in)[j-js]/N);
//      }
//    }
//  }
//  fftw_destroy_plan(c2r);
//  fftw_free(in);


  return;
}

void DiskInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,is-i) = initdens(k,j,is-i);
        prim(IM1,k,j,is-i) = 0;
        prim(IM2,k,j,is-i) = initvel2(k,j,is-i);
        prim(IM3,k,j,is-i) = 0;
        prim(IEN,k,j,is-i) = initprs(k,j,is-i);
      }
    }
  }
}
void DiskOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,ie+i) = initdens(k,j,ie+i);
        prim(IM1,k,j,ie+i) = 0;
        prim(IM2,k,j,ie+i) = initvel2(k,j,ie+i);
        prim(IM3,k,j,ie+i) = 0;
        prim(IEN,k,j,ie+i) = initprs(k,j,ie+i);
      }
    }
  }
}
void DiskInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ks-k,j,i) = initdens(ks-k,j,i);
        prim(IM1,ks-k,j,i) = 0;
        prim(IM2,ks-k,j,i) = initvel2(ks-k,j,i);
        prim(IM3,ks-k,j,i) = 0;
        prim(IEN,ks-k,j,i) = initprs(ks-k,j,i);
      }
    }
  }
}
void DiskOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke, int ngh)
{
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        prim(IDN,ke+k,j,i) = initdens(ke+k,j,i);
        prim(IM1,ke+k,j,i) = 0;
        prim(IM2,ke+k,j,i) = initvel2(ke+k,j,i);
        prim(IM3,ke+k,j,i) = 0;
        prim(IEN,ke+k,j,i) = initprs(ke+k,j,i);
      }
    }
  }
}

void ExternalGravity(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{
  Real omge2 = SQR(omg0) - SQR(omg);
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real src = -dt*prim(IDN,k,j,i)*(pmb->pcoord->x1v(i))*omge2;
        cons(IM1,k,j,i) += src;
        if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) += src*prim(IVX,k,j,i);
      }
    }
  }
}
