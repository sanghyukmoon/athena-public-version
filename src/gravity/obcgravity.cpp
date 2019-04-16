//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file obcgravity.cpp
//  \brief implementation of functions in class OBCGravityCar and OBCGravityCyl

// Purpose    : Self gravity with vacuum (open) boundary condition.
// Method     : James algorithm (R. A. James, 1977, JCoPh).
// Reference  : Moon, Kim, & Ostriker (2019), in press.
// History    : Written by Sanghyuk Moon, Jan 2019.

// C/C++ headers
#include <iostream>   // endl
#include <sstream>    // sstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>
#include <fstream>
#include <iomanip>
#include <fenv.h>

// Athena++ headers
#include "obcgravity.hpp"
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../coordinates/coordinates.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../task_list/grav_task_list.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef FFT
#include "fftw3.h"
#ifdef MPI_PARALLEL
#include "../fft/plimpton/remap_3d.h"
#include "../fft/plimpton/remap_2d.h"
#include "../fft/plimpton/pack_3d.h"
#include "../fft/plimpton/pack_2d.h"
#endif
#endif

#define MAX(A,B) ((A) > (B)) ? (A) : (B)

OBCGravityDriver::OBCGravityDriver(Mesh *pm, ParameterInput *pin)
{
  pmy_mesh_ = pm;
  if (COORDINATE_SYSTEM=="cartesian") {
    pmy_og_car = new OBCGravityCar(this, pm->pblock, pin);
    pmy_og_cyl = NULL;
  }
  else if (COORDINATE_SYSTEM=="cylindrical") {
    pmy_og_car = NULL;
    pmy_og_cyl = new OBCGravityCyl(this, pm->pblock, pin);
  }
  else {
    std::stringstream msg;
    msg << "### FATAL ERROR in OBCGravityDriver::OBCGravityDriver" << std::endl
         << "Currently, James solver only works in cartesian or cylindrical coordinates." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (pmy_mesh_->nbtotal != Globals::nranks) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OBCGravityDriver::OBCGravityDriver" << std::endl
         << "Number of MeshBlocks should be equal to the number of processors" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  gtlist_ = new GravitySolverTaskList(pin, pm);
}

// destructor
OBCGravityDriver::~OBCGravityDriver()
{
  if (COORDINATE_SYSTEM=="cartesian")
    delete pmy_og_car;
  else if (COORDINATE_SYSTEM=="cylindrical")
    delete pmy_og_cyl;
  else {
    std::stringstream msg;
    msg << "### FATAL ERROR in OBCGravityDriver::OBCGravityDriver" << std::endl
         << "Currently, James solver only works in cartesian or cylindrical coordinates." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  delete gtlist_;
}

void OBCGravityDriver::Solve(int stage)
{
  if (COORDINATE_SYSTEM=="cartesian") {
    MeshBlock *pmb=pmy_mesh_->pblock;
    pmy_og_car->LoadSource(pmb->phydro->u);

    // Step 1: Solve Poisson equation with zero boundary condition.
    pmy_og_car->SolveZeroBC();

    // Step 2: Calculate screening charges at the domain boundary.
    pmy_og_car->CalcBndCharge();
    pmy_og_car->LoadSource(pmb->phydro->u);

    // Step 3: Calculate gravitationl potential from the screening charges.
    pmy_og_car->CalcBndPot();

    // Step 4: Solve Poisson equation with correct boundary condition.
    pmy_og_car->SolveZeroBC();
    pmy_og_car->RetrieveResult(pmb->pgrav->phi);
    gtlist_->DoTaskListOneStage(pmy_mesh_, stage);
//    std::cout << std::scientific << std::setprecision(16);
//    if (Globals::my_rank==1) {
//      std::cout << "ghost cell at proc 0 = " << pmb->pgrav->phi(pmb->ks+32,pmb->js+63,pmb->is-1) << std::endl;
//    }
//    if (Globals::my_rank==3) {
//      std::cout << "first active cell at proc 1 = " << pmb->pgrav->phi(pmb->ks,pmb->js+63,pmb->is-1) << std::endl;
//    }
//    if (pmb->loc.lx1==0) {
//      for (int k=pmb->ks-1;k<=pmb->ke+1;++k) {
//        for (int j=pmb->js-1;j<=pmb->je+1;++j) {
//          for (int i=pmb->ie;i>=pmb->is;--i) {
//            pmb->pgrav->phi(k,j,i) = pmb->pgrav->phi(k,j,i-1);
//          }
//        }
//      }
//    }
  }
  else if (COORDINATE_SYSTEM=="cylindrical") {
    MeshBlock *pmb=pmy_mesh_->pblock;
    pmy_og_cyl->LoadSource(pmb->phydro->u);

    // Step 1: Solve Poisson equation with zero boundary condition.
    pmy_og_cyl->SolveZeroBC();

    // Step 2: Calculate screening charges at the domain boundary.
    pmy_og_cyl->CalcBndCharge();
    pmy_og_cyl->LoadSource(pmb->phydro->u);

    // Step 3: Calculate gravitationl potential from the screening charges.
    pmy_og_cyl->CalcBndPot();

    // Step 4: Solve Poisson equation with correct boundary condition.
    pmy_og_cyl->SolveZeroBC();
    pmy_og_cyl->RetrieveResult(pmb->pgrav->phi);
    gtlist_->DoTaskListOneStage(pmy_mesh_, stage);
  }
  else {
    std::stringstream msg;
    msg << "### FATAL ERROR in OBCGravityDriver::OBCGravityDriver" << std::endl
         << "Currently, James method only works in cartesian or cylindrical coordinates." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
}


// OBCGravityCar constructor
OBCGravityCar::OBCGravityCar(OBCGravityDriver *pod, MeshBlock *pmb, ParameterInput *pin)
 : Gravity(pmb, pin)
{
  ngh_ = 1;
  /* The number of extra ghost cells for the DGF calculation */
  ngh_grf_ = 16;
  pmy_driver_ = pod;
  Coordinates *pcrd = pmy_block->pcoord;
  RegionSize& mesh_size = pmy_block->pmy_mesh->mesh_size;
  RegionSize& block_size = pmy_block->block_size;
  Mesh *pm = pmy_block->pmy_mesh;
  Nx1 = mesh_size.nx1;
  Nx2 = mesh_size.nx2;
  Nx3 = mesh_size.nx3;
  nx1 = block_size.nx1;
  nx2 = block_size.nx2;
  nx3 = block_size.nx3;
  np1 = Nx1/nx1;
  np2 = Nx2/nx2;
  np3 = Nx3/nx3;

  dx1_ = pcrd->dx1v(NGHOST);
  dx2_ = pcrd->dx2v(NGHOST);
  dx3_ = pcrd->dx3v(NGHOST);

  // create plan for remap and FFT
  int np2d1,np2d2;
  bifactor(Globals::nranks, &np2d2, &np2d1);
  int ip1 = Globals::my_rank % np2d1; // ip1 is fast index
  int ip2 = Globals::my_rank / np2d1; // ip2 is slow index

  for (int i=CXB;i<=CE3P;++i) {
    dcmps[i].is = -1;
    dcmps[i].ie = -1;
    dcmps[i].js = -1;
    dcmps[i].je = -1;
    dcmps[i].ks = -1;
    dcmps[i].ke = -1;
    dcmps[i].nx1 = -1;
    dcmps[i].nx2 = -1;
    dcmps[i].nx3 = -1;
    dcmps[i].block_size = -1;
  }

  // block decomposition
  dcmps[CXB].is = (pmy_block->loc.lx1)*nx1;
  dcmps[CXB].ie = dcmps[CXB].is + nx1 - 1;
  dcmps[CXB].js = (pmy_block->loc.lx2)*nx2;
  dcmps[CXB].je = dcmps[CXB].js + nx2 - 1;
  dcmps[CXB].ks = (pmy_block->loc.lx3)*nx3;
  dcmps[CXB].ke = dcmps[CXB].ks + nx3 - 1;

  // x-pencil decomposition
  dcmps[CX1P].is = 0;
  dcmps[CX1P].ie = Nx1 - 1;
  dcmps[CX1P].js = ip2*Nx2/np2d2;
  dcmps[CX1P].je = (ip2+1)*Nx2/np2d2 - 1;
  dcmps[CX1P].ks = ip1*Nx3/np2d1;
  dcmps[CX1P].ke = (ip1+1)*Nx3/np2d1 - 1;

  // y-pencil decomposition
  dcmps[CX2P].is = ip1*Nx1/np2d1;
  dcmps[CX2P].ie = (ip1+1)*Nx1/np2d1 - 1;
  dcmps[CX2P].js = 0;
  dcmps[CX2P].je = Nx2 - 1;
  dcmps[CX2P].ks = ip2*Nx3/np2d2;
  dcmps[CX2P].ke = (ip2+1)*Nx3/np2d2 - 1;

  // z-pencil decomposition
  dcmps[CX3P].is = ip1*Nx1/np2d1;
  dcmps[CX3P].ie = (ip1+1)*Nx1/np2d1 - 1;
  dcmps[CX3P].js = ip2*Nx2/np2d2;
  dcmps[CX3P].je = (ip2+1)*Nx2/np2d2 - 1;
  dcmps[CX3P].ks = 0;
  dcmps[CX3P].ke = Nx3 - 1;

  // For the sine/cosine transform in boundary solver, include single layer of
  // ghost cells.
  // block decomposition
  dcmps[PB].is = (pmy_block->loc.lx1)*nx1 + ngh_;
  dcmps[PB].ie = dcmps[PB].is + nx1 - 1;
  dcmps[PB].js = (pmy_block->loc.lx2)*nx2 + ngh_;
  dcmps[PB].je = dcmps[PB].js + nx2 - 1;
  dcmps[PB].ks = (pmy_block->loc.lx3)*nx3 + ngh_;
  dcmps[PB].ke = dcmps[PB].ks + nx3 - 1;
  if (pmy_block->loc.lx1==0) dcmps[PB].is -= ngh_;
  if (pmy_block->loc.lx2==0) dcmps[PB].js -= ngh_;
  if (pmy_block->loc.lx3==0) dcmps[PB].ks -= ngh_;
  if (pmy_block->loc.lx1==np1-1) dcmps[PB].ie += ngh_;
  if (pmy_block->loc.lx2==np2-1) dcmps[PB].je += ngh_;
  if (pmy_block->loc.lx3==np3-1) dcmps[PB].ke += ngh_;

  // x-pencil decomposition
  dcmps[P1P].is = 0;
  dcmps[P1P].ie = Nx1 - 1 + 2*ngh_;
  dcmps[P1P].js = ip2*Nx2/np2d2 + ngh_;
  dcmps[P1P].je = (ip2+1)*Nx2/np2d2 - 1 + ngh_;
  dcmps[P1P].ks = ip1*Nx3/np2d1 + ngh_;
  dcmps[P1P].ke = (ip1+1)*Nx3/np2d1 - 1 + ngh_;
  if (ip1==0) dcmps[P1P].ks-=ngh_;
  if (ip1==np2d1-1) dcmps[P1P].ke+=ngh_;
  if (ip2==0) dcmps[P1P].js-=ngh_;
  if (ip2==np2d2-1) dcmps[P1P].je+=ngh_;

  // y-pencil decomposition
  dcmps[P2P].is = ip1*Nx1/np2d1 + ngh_;
  dcmps[P2P].ie = (ip1+1)*Nx1/np2d1 - 1 + ngh_;
  dcmps[P2P].js = 0;
  dcmps[P2P].je = Nx2 - 1 + 2*ngh_;
  dcmps[P2P].ks = ip2*Nx3/np2d2 + ngh_;
  dcmps[P2P].ke = (ip2+1)*Nx3/np2d2 - 1 + ngh_;
  if (ip1==0) dcmps[P2P].is-=ngh_;
  if (ip1==np2d1-1) dcmps[P2P].ie+=ngh_;
  if (ip2==0) dcmps[P2P].ks-=ngh_;
  if (ip2==np2d2-1) dcmps[P2P].ke+=ngh_;

  // z-pencil decomposition
  dcmps[P3P].is = ip1*Nx1/np2d1 + ngh_;
  dcmps[P3P].ie = (ip1+1)*Nx1/np2d1 - 1 + ngh_;
  dcmps[P3P].js = ip2*Nx2/np2d2 + ngh_;
  dcmps[P3P].je = (ip2+1)*Nx2/np2d2 - 1 + ngh_;
  dcmps[P3P].ks = 0;
  dcmps[P3P].ke = Nx3 - 1 + 2*ngh_;
  if (ip1==0) dcmps[P3P].is-=ngh_;
  if (ip1==np2d1-1) dcmps[P3P].ie+=ngh_;
  if (ip2==0) dcmps[P3P].js-=ngh_;
  if (ip2==np2d2-1) dcmps[P3P].je+=ngh_;

  // For the calculation of discrete Green's function (DGF), use extended
  // domain.
  // block decomposition
  dcmps[CEB].is = (pmy_block->loc.lx1)*nx1 + ngh_+ngh_grf_;
  dcmps[CEB].ie = dcmps[CEB].is + nx1 - 1;
  dcmps[CEB].js = (pmy_block->loc.lx2)*nx2 + ngh_+ngh_grf_;
  dcmps[CEB].je = dcmps[CEB].js + nx2 - 1;
  dcmps[CEB].ks = (pmy_block->loc.lx3)*nx3 + ngh_+ngh_grf_;
  dcmps[CEB].ke = dcmps[CEB].ks + nx3 - 1;
  if (pmy_block->loc.lx1==0) dcmps[CEB].is -= ngh_+ngh_grf_;
  if (pmy_block->loc.lx2==0) dcmps[CEB].js -= ngh_+ngh_grf_;
  if (pmy_block->loc.lx3==0) dcmps[CEB].ks -= ngh_+ngh_grf_;
  if (pmy_block->loc.lx1==np1-1) dcmps[CEB].ie += ngh_;
  if (pmy_block->loc.lx2==np2-1) dcmps[CEB].je += ngh_;
  if (pmy_block->loc.lx3==np3-1) dcmps[CEB].ke += ngh_;

  // x-pencil decomposition
  dcmps[CE1P].is = 0;
  dcmps[CE1P].ie = Nx1 - 1 + 2*ngh_+ngh_grf_;
  dcmps[CE1P].js = ip2*Nx2/np2d2 + ngh_+ngh_grf_;
  dcmps[CE1P].je = (ip2+1)*Nx2/np2d2 - 1 + ngh_+ngh_grf_;
  dcmps[CE1P].ks = ip1*Nx3/np2d1 + ngh_+ngh_grf_;
  dcmps[CE1P].ke = (ip1+1)*Nx3/np2d1 - 1 + ngh_+ngh_grf_;
  if (ip1==0) dcmps[CE1P].ks -= ngh_+ngh_grf_;
  if (ip1==np2d1-1) dcmps[CE1P].ke += ngh_;
  if (ip2==0) dcmps[CE1P].js -= ngh_+ngh_grf_;
  if (ip2==np2d2-1) dcmps[CE1P].je += ngh_;

  // y-pencil decomposition
  dcmps[CE2P].is = ip1*Nx1/np2d1 + ngh_+ngh_grf_;
  dcmps[CE2P].ie = (ip1+1)*Nx1/np2d1 - 1 + ngh_+ngh_grf_;
  dcmps[CE2P].js = 0;
  dcmps[CE2P].je = Nx2 - 1 + 2*ngh_+ngh_grf_;
  dcmps[CE2P].ks = ip2*Nx3/np2d2 + ngh_+ngh_grf_;
  dcmps[CE2P].ke = (ip2+1)*Nx3/np2d2 - 1 + ngh_+ngh_grf_;
  if (ip1==0) dcmps[CE2P].is -= ngh_+ngh_grf_;
  if (ip1==np2d1-1) dcmps[CE2P].ie += ngh_;
  if (ip2==0) dcmps[CE2P].ks -= ngh_+ngh_grf_;
  if (ip2==np2d2-1) dcmps[CE2P].ke += ngh_;

  // z-pencil decomposition
  dcmps[CE3P].is = ip1*Nx1/np2d1 + ngh_+ngh_grf_;
  dcmps[CE3P].ie = (ip1+1)*Nx1/np2d1 - 1 + ngh_+ngh_grf_;
  dcmps[CE3P].js = ip2*Nx2/np2d2 + ngh_+ngh_grf_;
  dcmps[CE3P].je = (ip2+1)*Nx2/np2d2 - 1 + ngh_+ngh_grf_;
  dcmps[CE3P].ks = 0;
  dcmps[CE3P].ke = Nx3 - 1 + 2*ngh_+ngh_grf_;
  if (ip1==0) dcmps[CE3P].is -= ngh_+ngh_grf_;
  if (ip1==np2d1-1) dcmps[CE3P].ie += ngh_;
  if (ip2==0) dcmps[CE3P].js -= ngh_+ngh_grf_;
  if (ip2==np2d2-1) dcmps[CE3P].je += ngh_;

  for (int i=CXB;i<=CE3P;++i) {
    dcmps[i].nx1 = dcmps[i].ie - dcmps[i].is + 1;
    dcmps[i].nx2 = dcmps[i].je - dcmps[i].js + 1;
    dcmps[i].nx3 = dcmps[i].ke - dcmps[i].ks + 1;
    dcmps[i].block_size = dcmps[i].nx1*dcmps[i].nx2*dcmps[i].nx3;
  }

  // communicator for the 2D FFT on the boundary
  int color;
  color = (dcmps[CXB].is == 0) ? dcmps[CXB].is : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[STH]);
  color = (dcmps[CXB].ie == Nx1-1) ? dcmps[CXB].ie : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[NTH]);
  color = (dcmps[CXB].js == 0) ? dcmps[CXB].js : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[WST]);
  color = (dcmps[CXB].je == Nx2-1) ? dcmps[CXB].je : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[EST]);
  color = (dcmps[CXB].ks == 0) ? dcmps[CXB].ks : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[CBOT]);
  color = (dcmps[CXB].ke == Nx3-1) ? dcmps[CXB].ke : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[CTOP]);
  MPI_Comm_split(MPI_COMM_WORLD, dcmps[CXB].js+Nx2*dcmps[CXB].ks, dcmps[CXB].is, &x1comm);
  MPI_Comm_split(MPI_COMM_WORLD, dcmps[CXB].is+Nx1*dcmps[CXB].ks, dcmps[CXB].js, &x2comm);
  MPI_Comm_split(MPI_COMM_WORLD, dcmps[CXB].js+Nx2*dcmps[CXB].is, dcmps[CXB].ks, &x3comm);
  MPI_Comm_size(x1comm, &x1comm_size);
  MPI_Comm_size(x2comm, &x2comm_size);
  MPI_Comm_size(x3comm, &x3comm_size);
  MPI_Comm_rank(x1comm, &x1rank);
  MPI_Comm_rank(x2comm, &x2rank);
  MPI_Comm_rank(x3comm, &x3rank);

  for (int i=STH;i<=CTOP;++i ) {
    for (int j=CBLOCK;j<=FFT_SECOND;++j) {
      bndry_dcmps[i][j].is = 0;
      bndry_dcmps[i][j].ie = -1;
      bndry_dcmps[i][j].js = 0;
      bndry_dcmps[i][j].je = -1;
      bndry_dcmps[i][j].ks = 0;
      bndry_dcmps[i][j].ke = -1;
    }
  }
  int bndrank, bndsize, nx;
  if (dcmps[CXB].is==0) { // STH boundary
    bndry_dcmps[STH][CBLOCK].is = 0;
    bndry_dcmps[STH][CBLOCK].ie = 0;
    bndry_dcmps[STH][CBLOCK].js = dcmps[PB].js;
    bndry_dcmps[STH][CBLOCK].je = dcmps[PB].je;
    bndry_dcmps[STH][CBLOCK].ks = dcmps[PB].ks;
    bndry_dcmps[STH][CBLOCK].ke = dcmps[PB].ke;
    MPI_Comm_rank(bndcomm[STH], &bndrank);
    MPI_Comm_size(bndcomm[STH], &bndsize);
    nx = Nx3/np2/np3;
    bndry_dcmps[STH][FFT_FIRST].is = 0;
    bndry_dcmps[STH][FFT_FIRST].ie = 0;
    bndry_dcmps[STH][FFT_FIRST].js = 0;
    bndry_dcmps[STH][FFT_FIRST].je = Nx2-1+2*ngh_;
    bndry_dcmps[STH][FFT_FIRST].ks = bndrank * nx + ngh_;
    bndry_dcmps[STH][FFT_FIRST].ke = (bndrank+1) * nx - 1 + ngh_;
    if (bndrank==0) bndry_dcmps[STH][FFT_FIRST].ks-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[STH][FFT_FIRST].ke+=ngh_;
    nx = Nx2/np2/np3;
    bndry_dcmps[STH][FFT_SECOND].is = 0;
    bndry_dcmps[STH][FFT_SECOND].ie = 0;
    bndry_dcmps[STH][FFT_SECOND].js = bndrank * nx + ngh_;
    bndry_dcmps[STH][FFT_SECOND].je = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[STH][FFT_SECOND].ks = 0;
    bndry_dcmps[STH][FFT_SECOND].ke = Nx3-1+2*ngh_;
    if (bndrank==0) bndry_dcmps[STH][FFT_SECOND].js-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[STH][FFT_SECOND].je+=ngh_;
  }
  if (dcmps[CXB].ie==Nx1-1) { // NTH boundary
    bndry_dcmps[NTH][CBLOCK].is = Nx1-1+2*ngh_;
    bndry_dcmps[NTH][CBLOCK].ie = Nx1-1+2*ngh_;
    bndry_dcmps[NTH][CBLOCK].js = dcmps[PB].js;
    bndry_dcmps[NTH][CBLOCK].je = dcmps[PB].je;
    bndry_dcmps[NTH][CBLOCK].ks = dcmps[PB].ks;
    bndry_dcmps[NTH][CBLOCK].ke = dcmps[PB].ke;
    MPI_Comm_rank(bndcomm[NTH], &bndrank);
    MPI_Comm_size(bndcomm[NTH], &bndsize);
    nx = Nx3/np2/np3;
    bndry_dcmps[NTH][FFT_FIRST].is = Nx1-1+2*ngh_;
    bndry_dcmps[NTH][FFT_FIRST].ie = Nx1-1+2*ngh_;
    bndry_dcmps[NTH][FFT_FIRST].js = 0;
    bndry_dcmps[NTH][FFT_FIRST].je = Nx2-1+2*ngh_;
    bndry_dcmps[NTH][FFT_FIRST].ks = bndrank * nx + ngh_;
    bndry_dcmps[NTH][FFT_FIRST].ke = (bndrank+1) * nx - 1 + ngh_;
    if (bndrank==0) bndry_dcmps[NTH][FFT_FIRST].ks-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[NTH][FFT_FIRST].ke+=ngh_;
    nx = Nx2/np2/np3;
    bndry_dcmps[NTH][FFT_SECOND].is = Nx1-1+2*ngh_;
    bndry_dcmps[NTH][FFT_SECOND].ie = Nx1-1+2*ngh_;
    bndry_dcmps[NTH][FFT_SECOND].js = bndrank * nx + ngh_;
    bndry_dcmps[NTH][FFT_SECOND].je = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[NTH][FFT_SECOND].ks = 0;
    bndry_dcmps[NTH][FFT_SECOND].ke = Nx3-1+2*ngh_;
    if (bndrank==0) bndry_dcmps[NTH][FFT_SECOND].js-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[NTH][FFT_SECOND].je+=ngh_;
  }
  if (dcmps[CXB].js==0) { // WST boundary
    bndry_dcmps[WST][CBLOCK].is = dcmps[PB].is;
    bndry_dcmps[WST][CBLOCK].ie = dcmps[PB].ie;
    bndry_dcmps[WST][CBLOCK].js = 0;
    bndry_dcmps[WST][CBLOCK].je = 0;
    bndry_dcmps[WST][CBLOCK].ks = dcmps[PB].ks;
    bndry_dcmps[WST][CBLOCK].ke = dcmps[PB].ke;
    MPI_Comm_rank(bndcomm[WST], &bndrank);
    MPI_Comm_size(bndcomm[WST], &bndsize);
    nx = Nx3/np3/np1;
    bndry_dcmps[WST][FFT_FIRST].is = 0;
    bndry_dcmps[WST][FFT_FIRST].ie = Nx1-1+2*ngh_;
    bndry_dcmps[WST][FFT_FIRST].js = 0;
    bndry_dcmps[WST][FFT_FIRST].je = 0;
    bndry_dcmps[WST][FFT_FIRST].ks = bndrank * nx + ngh_;
    bndry_dcmps[WST][FFT_FIRST].ke = (bndrank+1) * nx - 1 + ngh_;
    if (bndrank==0) bndry_dcmps[WST][FFT_FIRST].ks-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[WST][FFT_FIRST].ke+=ngh_;
    nx = Nx1/np3/np1;
    bndry_dcmps[WST][FFT_SECOND].is = bndrank * nx + ngh_;
    bndry_dcmps[WST][FFT_SECOND].ie = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[WST][FFT_SECOND].js = 0;
    bndry_dcmps[WST][FFT_SECOND].je = 0;
    bndry_dcmps[WST][FFT_SECOND].ks = 0;
    bndry_dcmps[WST][FFT_SECOND].ke = Nx3-1+2*ngh_;
    if (bndrank==0) bndry_dcmps[WST][FFT_SECOND].is-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[WST][FFT_SECOND].ie+=ngh_;
  }
  if (dcmps[CXB].je==Nx2-1) { // EST boundary
    bndry_dcmps[EST][CBLOCK].is = dcmps[PB].is;
    bndry_dcmps[EST][CBLOCK].ie = dcmps[PB].ie;
    bndry_dcmps[EST][CBLOCK].js = Nx2-1+2*ngh_;
    bndry_dcmps[EST][CBLOCK].je = Nx2-1+2*ngh_;
    bndry_dcmps[EST][CBLOCK].ks = dcmps[PB].ks;
    bndry_dcmps[EST][CBLOCK].ke = dcmps[PB].ke;
    MPI_Comm_rank(bndcomm[EST], &bndrank);
    MPI_Comm_size(bndcomm[EST], &bndsize);
    nx = Nx3/np3/np1;
    bndry_dcmps[EST][FFT_FIRST].is = 0;
    bndry_dcmps[EST][FFT_FIRST].ie = Nx1-1+2*ngh_;
    bndry_dcmps[EST][FFT_FIRST].js = Nx2-1+2*ngh_;
    bndry_dcmps[EST][FFT_FIRST].je = Nx2-1+2*ngh_;
    bndry_dcmps[EST][FFT_FIRST].ks = bndrank * nx + ngh_;
    bndry_dcmps[EST][FFT_FIRST].ke = (bndrank+1) * nx - 1 + ngh_;
    if (bndrank==0) bndry_dcmps[EST][FFT_FIRST].ks-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[EST][FFT_FIRST].ke+=ngh_;
    nx = Nx1/np3/np1;
    bndry_dcmps[EST][FFT_SECOND].is = bndrank * nx + ngh_;
    bndry_dcmps[EST][FFT_SECOND].ie = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[EST][FFT_SECOND].js = Nx2-1+2*ngh_;
    bndry_dcmps[EST][FFT_SECOND].je = Nx2-1+2*ngh_;
    bndry_dcmps[EST][FFT_SECOND].ks = 0;
    bndry_dcmps[EST][FFT_SECOND].ke = Nx3-1+2*ngh_;
    if (bndrank==0) bndry_dcmps[EST][FFT_SECOND].is-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[EST][FFT_SECOND].ie+=ngh_;
  }
  if (dcmps[CXB].ks==0) { // CBOT boundary
    bndry_dcmps[CBOT][CBLOCK].is = dcmps[PB].is;
    bndry_dcmps[CBOT][CBLOCK].ie = dcmps[PB].ie;
    bndry_dcmps[CBOT][CBLOCK].js = dcmps[PB].js;
    bndry_dcmps[CBOT][CBLOCK].je = dcmps[PB].je;
    bndry_dcmps[CBOT][CBLOCK].ks = 0;
    bndry_dcmps[CBOT][CBLOCK].ke = 0;
    MPI_Comm_rank(bndcomm[CBOT], &bndrank);
    MPI_Comm_size(bndcomm[CBOT], &bndsize);
    nx = Nx2/np1/np2;
    bndry_dcmps[CBOT][FFT_FIRST].is = 0;
    bndry_dcmps[CBOT][FFT_FIRST].ie = Nx1-1+2*ngh_;
    bndry_dcmps[CBOT][FFT_FIRST].js = bndrank * nx + ngh_;
    bndry_dcmps[CBOT][FFT_FIRST].je = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[CBOT][FFT_FIRST].ks = 0;
    bndry_dcmps[CBOT][FFT_FIRST].ke = 0;
    if (bndrank==0) bndry_dcmps[CBOT][FFT_FIRST].js-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[CBOT][FFT_FIRST].je+=ngh_;
    nx = Nx1/np1/np2;
    bndry_dcmps[CBOT][FFT_SECOND].is = bndrank * nx + ngh_;
    bndry_dcmps[CBOT][FFT_SECOND].ie = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[CBOT][FFT_SECOND].js = 0;
    bndry_dcmps[CBOT][FFT_SECOND].je = Nx2-1+2*ngh_;
    bndry_dcmps[CBOT][FFT_SECOND].ks = 0;
    bndry_dcmps[CBOT][FFT_SECOND].ke = 0;
    if (bndrank==0) bndry_dcmps[CBOT][FFT_SECOND].is-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[CBOT][FFT_SECOND].ie+=ngh_;
  }
  if (dcmps[CXB].ke==Nx3-1) { // CTOP boundary
    bndry_dcmps[CTOP][CBLOCK].is = dcmps[PB].is;
    bndry_dcmps[CTOP][CBLOCK].ie = dcmps[PB].ie;
    bndry_dcmps[CTOP][CBLOCK].js = dcmps[PB].js;
    bndry_dcmps[CTOP][CBLOCK].je = dcmps[PB].je;
    bndry_dcmps[CTOP][CBLOCK].ks = Nx3-1+2*ngh_;
    bndry_dcmps[CTOP][CBLOCK].ke = Nx3-1+2*ngh_;
    MPI_Comm_rank(bndcomm[CTOP], &bndrank);
    MPI_Comm_size(bndcomm[CTOP], &bndsize);
    nx = Nx2/np1/np2;
    bndry_dcmps[CTOP][FFT_FIRST].is = 0;
    bndry_dcmps[CTOP][FFT_FIRST].ie = Nx1-1+2*ngh_;
    bndry_dcmps[CTOP][FFT_FIRST].js = bndrank * nx + ngh_;
    bndry_dcmps[CTOP][FFT_FIRST].je = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[CTOP][FFT_FIRST].ks = Nx3-1+2*ngh_;
    bndry_dcmps[CTOP][FFT_FIRST].ke = Nx3-1+2*ngh_;
    if (bndrank==0) bndry_dcmps[CTOP][FFT_FIRST].js-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[CTOP][FFT_FIRST].je+=ngh_;
    nx = Nx1/np1/np2;
    bndry_dcmps[CTOP][FFT_SECOND].is = bndrank * nx + ngh_;
    bndry_dcmps[CTOP][FFT_SECOND].ie = (bndrank+1) * nx - 1 + ngh_;
    bndry_dcmps[CTOP][FFT_SECOND].js = 0;
    bndry_dcmps[CTOP][FFT_SECOND].je = Nx2-1+2*ngh_;
    bndry_dcmps[CTOP][FFT_SECOND].ks = Nx3-1+2*ngh_;
    bndry_dcmps[CTOP][FFT_SECOND].ke = Nx3-1+2*ngh_;
    if (bndrank==0) bndry_dcmps[CTOP][FFT_SECOND].is-=ngh_;
    if (bndrank==bndsize-1) bndry_dcmps[CTOP][FFT_SECOND].ie+=ngh_;
  }

  for (int i=STH;i<=CTOP;++i) {
    for (int j=CBLOCK;j<=FFT_SECOND;++j) {
      bndry_dcmps[i][j].nx1 = bndry_dcmps[i][j].ie - bndry_dcmps[i][j].is + 1;
      bndry_dcmps[i][j].nx2 = bndry_dcmps[i][j].je - bndry_dcmps[i][j].js + 1;
      bndry_dcmps[i][j].nx3 = bndry_dcmps[i][j].ke - bndry_dcmps[i][j].ks + 1;
      bndry_dcmps[i][j].block_size = 
        bndry_dcmps[i][j].nx1*bndry_dcmps[i][j].nx2*bndry_dcmps[i][j].nx3;
    }
  }

  if ((Nx2 % np2d2 != 0)||(Nx3 % np2d2 != 0)||(Nx1 % np2d1 != 0)
      ||(Nx3 % np2d1 != 0)||(nx3 % np1 != 0)||(nx1 % np3 != 0)
      ||(nx2 % np1 != 0)||(nx1 % np2 != 0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in OBCGravityCar" << std::endl
         << "domain decomposition failed in obcgrav" << std::endl
         << "np2d1 = " << np2d1 << " np2d2 = " << np2d2 << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // allocate arrays
  int arrsize, first_size, second_size, third_size;
  arrsize = dcmps[CXB].block_size;
  in_  = fftw_alloc_real(arrsize);
  out_  = fftw_alloc_real(arrsize);
  arrsize = MAX(dcmps[CEB].block_size, dcmps[CE1P].block_size);
  arrsize = MAX(arrsize, dcmps[CE2P].block_size);
  arrsize = MAX(arrsize, dcmps[CE3P].block_size);
  in2_ = fftw_alloc_real(arrsize);
  buf_ = fftw_alloc_real(arrsize);

  first_size = MAX(dcmps[PB].nx2*dcmps[PB].nx3, (Nx2+2*ngh_)*(Nx3/np2/np3+2*ngh_));
  second_size = MAX((Nx2+2*ngh_)*(Nx3/np2/np3+2*ngh_), (Nx3+2*ngh_)*(Nx2/np2/np3+2*ngh_));
  third_size = MAX((Nx3+2*ngh_)*(Nx2/np2/np3+2*ngh_), dcmps[PB].nx2*dcmps[PB].nx3);
  for (int i=STH;i<=NTH;++i) {
    sigma[i] = fftw_alloc_real(first_size);
    for (int j=C;j<=S;++j) {
      sigma_mid[i][j] = fftw_alloc_real(second_size);
      for (int k=C;k<=S;++k ) {
        sigma_fft[i][j][k] = fftw_alloc_real(third_size);
      }
    }
  }
  first_size = MAX(dcmps[PB].nx1*dcmps[PB].nx3, (Nx1+2*ngh_)*(Nx3/np3/np1+2*ngh_));
  second_size = MAX((Nx1+2*ngh_)*(Nx3/np3/np1+2*ngh_), (Nx3+2*ngh_)*(Nx1/np3/np1+2*ngh_));
  third_size = MAX((Nx3+2*ngh_)*(Nx1/np3/np1+2*ngh_), dcmps[PB].nx1*dcmps[PB].nx3);
  for (int i=WST;i<=EST;++i) {
    sigma[i] = fftw_alloc_real(first_size);
    for (int j=C;j<=S;++j) {
      sigma_mid[i][j] = fftw_alloc_real(second_size);
      for (int k=C;k<=S;++k ) {
        sigma_fft[i][j][k] = fftw_alloc_real(third_size);
      }
    }
  }
  first_size = MAX(dcmps[PB].nx1*dcmps[PB].nx2, (Nx1+2*ngh_)*(Nx2/np1/np2+2*ngh_));
  second_size = MAX((Nx1+2*ngh_)*(Nx2/np1/np2+2*ngh_), (Nx2+2*ngh_)*(Nx1/np1/np2+2*ngh_));
  third_size = MAX((Nx2+2*ngh_)*(Nx1/np1/np2+2*ngh_), dcmps[PB].nx1*dcmps[PB].nx2);
  for (int i=CBOT;i<=CTOP;++i) {
    sigma[i] = fftw_alloc_real(first_size);
    for (int j=C;j<=S;++j) {
      sigma_mid[i][j] = fftw_alloc_real(second_size);
      for (int k=C;k<=S;++k ) {
        sigma_fft[i][j][k] = fftw_alloc_real(third_size);
      }
    }
  }
  for (int i=C;i<=S;++i) {
    for (int j=C;j<=S;++j) {
      for (int k=C;k<=S;++k) {
        sigfft[i][j][k] = fftw_alloc_real(dcmps[PB].block_size);
      }
    }
  }
  arrsize = MAX(dcmps[PB].block_size, dcmps[P1P].block_size);
  arrsize = MAX(arrsize, dcmps[P2P].block_size);
  arrsize = MAX(arrsize, dcmps[P3P].block_size);
  grf = fftw_alloc_real(arrsize);

  lambda3_.NewAthenaArray(dcmps[CX1P].nx3);
  lambda2_.NewAthenaArray(dcmps[CX1P].nx2);
  lambda1_.NewAthenaArray(dcmps[CX1P].nx1);
  lambda33_.NewAthenaArray(dcmps[CE1P].nx3);
  lambda22_.NewAthenaArray(dcmps[CE1P].nx2);
  lambda11_.NewAthenaArray(dcmps[CE1P].nx1);

  for (int k=0;k<dcmps[CX1P].nx3;++k) {
    lambda3_(k) = -4*SQR(sin(0.5*PI*(Real)(dcmps[CX1P].ks+k+1)/((Real)((Nx3)+1))))/SQR(dx3_);
  }
  for (int j=0;j<dcmps[CX1P].nx2;++j) {
    lambda2_(j) = -4*SQR(sin(0.5*PI*(Real)(dcmps[CX1P].js+j+1)/((Real)((Nx2)+1))))/SQR(dx2_);
  }
  for (int i=0;i<dcmps[CX1P].nx1;++i) {
    lambda1_(i) = -4*SQR(sin(0.5*PI*(Real)(dcmps[CX1P].is+i+1)/((Real)((Nx1)+1))))/SQR(dx1_);
  }
  for (int k=0;k<dcmps[CE1P].nx3;++k) {
    lambda33_(k) = -4*SQR(sin(0.5*PI*(Real)(dcmps[CE1P].ks+k+1)/((Real)((Nx3+2*ngh_+ngh_grf_)+1))))/SQR(dx3_);
  }
  for (int j=0;j<dcmps[CE1P].nx2;++j) {
    lambda22_(j) = -4*SQR(sin(0.5*PI*(Real)(dcmps[CE1P].js+j+1)/((Real)((Nx2+2*ngh_+ngh_grf_)+1))))/SQR(dx2_);
  }
  for (int i=0;i<dcmps[CE1P].nx1;++i) {
    lambda11_(i) = -4*SQR(sin(0.5*PI*(Real)(dcmps[CE1P].is+i+1)/((Real)((Nx1+2*ngh_+ngh_grf_)+1))))/SQR(dx1_);
  }

  fftw_r2r_kind DST[] = {FFTW_RODFT00};
  fftw_r2r_kind DCT[] = {FFTW_REDFT00};
  fft_plan_r2r_[0] = fftw_plan_many_r2r(1, &(dcmps[CX2P].nx2),
    dcmps[CX2P].block_size/dcmps[CX2P].nx2, in_, NULL, 1, dcmps[CX2P].nx2,
    in_, NULL, 1, dcmps[CX2P].nx2, DST, FFTW_MEASURE);
  fft_plan_r2r_[1] = fftw_plan_many_r2r(1, &(dcmps[CX3P].nx3),
    dcmps[CX3P].block_size/dcmps[CX3P].nx3, in_, NULL, 1, dcmps[CX3P].nx3,
    in_, NULL, 1, dcmps[CX3P].nx3, DST, FFTW_MEASURE);
  fft_plan_r2r_[2] = fftw_plan_many_r2r(1, &(dcmps[CX1P].nx1),
    dcmps[CX1P].block_size/dcmps[CX1P].nx1, in_, NULL, 1, dcmps[CX1P].nx1,
    in_, NULL, 1, dcmps[CX1P].nx1, DST, FFTW_MEASURE);
  fft_plan_r2r_[3] = fftw_plan_many_r2r(1, &(dcmps[CX2P].nx2),
    dcmps[CX2P].block_size/dcmps[CX2P].nx2, out_, NULL, 1, dcmps[CX2P].nx2,
    out_, NULL, 1, dcmps[CX2P].nx2, DST, FFTW_MEASURE);
  fft_plan_r2r_[4] = fftw_plan_many_r2r(1, &(dcmps[CX3P].nx3),
    dcmps[CX3P].block_size/dcmps[CX3P].nx3, out_, NULL, 1, dcmps[CX3P].nx3,
    out_, NULL, 1, dcmps[CX3P].nx3, DST, FFTW_MEASURE);
  fft_plan_r2r_[5] = fftw_plan_many_r2r(1, &(dcmps[CX1P].nx1),
    dcmps[CX1P].block_size/dcmps[CX1P].nx1, out_, NULL, 1, dcmps[CX1P].nx1,
    out_, NULL, 1, dcmps[CX1P].nx1, DST, FFTW_MEASURE);
  // plan for grf (padded block)
  fft_plan_r2r_[6] = fftw_plan_many_r2r(1, &(dcmps[P2P].nx2),
    dcmps[P2P].block_size/dcmps[P2P].nx2, grf, NULL, 1, dcmps[P2P].nx2,
    grf, NULL, 1, dcmps[P2P].nx2, DCT, FFTW_MEASURE);
  fft_plan_r2r_[7] = fftw_plan_many_r2r(1, &(dcmps[P3P].nx3),
    dcmps[P3P].block_size/dcmps[P3P].nx3, grf, NULL, 1, dcmps[P3P].nx3,
    grf, NULL, 1, dcmps[P3P].nx3, DCT, FFTW_MEASURE);
  fft_plan_r2r_[8] = fftw_plan_many_r2r(1, &(dcmps[P1P].nx1),
    dcmps[P1P].block_size/dcmps[P1P].nx1, grf, NULL, 1, dcmps[P1P].nx1,
    grf, NULL, 1, dcmps[P1P].nx1, DCT, FFTW_MEASURE);
  // plan for extended block
  fft_plan_r2r_[9] = fftw_plan_many_r2r(1, &(dcmps[CE2P].nx2),
    dcmps[CE2P].block_size/dcmps[CE2P].nx2, in2_, NULL, 1, dcmps[CE2P].nx2,
    in2_, NULL, 1, dcmps[CE2P].nx2, DST, FFTW_MEASURE);
  fft_plan_r2r_[10] = fftw_plan_many_r2r(1, &(dcmps[CE3P].nx3),
    dcmps[CE3P].block_size/dcmps[CE3P].nx3, in2_, NULL, 1, dcmps[CE3P].nx3,
    in2_, NULL, 1, dcmps[CE3P].nx3, DST, FFTW_MEASURE);
  fft_plan_r2r_[11] = fftw_plan_many_r2r(1, &(dcmps[CE1P].nx1),
    dcmps[CE1P].block_size/dcmps[CE1P].nx1, in2_, NULL, 1, dcmps[CE1P].nx1,
    in2_, NULL, 1, dcmps[CE1P].nx1, DST, FFTW_MEASURE);
  fft_plan_r2r_[12] = fftw_plan_many_r2r(1, &(dcmps[CE2P].nx2),
    dcmps[CE2P].block_size/dcmps[CE2P].nx2, in2_, NULL, 1, dcmps[CE2P].nx2,
    in2_, NULL, 1, dcmps[CE2P].nx2, DST, FFTW_MEASURE);
  fft_plan_r2r_[13] = fftw_plan_many_r2r(1, &(dcmps[CE3P].nx3),
    dcmps[CE3P].block_size/dcmps[CE3P].nx3, in2_, NULL, 1, dcmps[CE3P].nx3,
    in2_, NULL, 1, dcmps[CE3P].nx3, DST, FFTW_MEASURE);
  fft_plan_r2r_[14] = fftw_plan_many_r2r(1, &(dcmps[CE1P].nx1),
    dcmps[CE1P].block_size/dcmps[CE1P].nx1, in2_, NULL, 1, dcmps[CE1P].nx1,
    in2_, NULL, 1, dcmps[CE1P].nx1, DST, FFTW_MEASURE);


  RmpPlan[CXB][CX2P] = remap_3d_create_plan(MPI_COMM_WORLD,dcmps[CXB].is,
    dcmps[CXB].ie, dcmps[CXB].js, dcmps[CXB].je, dcmps[CXB].ks, dcmps[CXB].ke,
    dcmps[CX2P].is, dcmps[CX2P].ie, dcmps[CX2P].js, dcmps[CX2P].je,
    dcmps[CX2P].ks, dcmps[CX2P].ke, 1, 1, 0, 2);
  RmpPlan[CX2P][CX3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CX2P].js,
    dcmps[CX2P].je, dcmps[CX2P].ks, dcmps[CX2P].ke, dcmps[CX2P].is, dcmps[CX2P].ie,
    dcmps[CX3P].js, dcmps[CX3P].je, dcmps[CX3P].ks, dcmps[CX3P].ke, dcmps[CX3P].is,
    dcmps[CX3P].ie, 1, 1, 0, 2);
  RmpPlan[CX3P][CX1P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CX3P].ks,
    dcmps[CX3P].ke, dcmps[CX3P].is, dcmps[CX3P].ie, dcmps[CX3P].js, dcmps[CX3P].je,
    dcmps[CX1P].ks, dcmps[CX1P].ke, dcmps[CX1P].is, dcmps[CX1P].ie, dcmps[CX1P].js,
    dcmps[CX1P].je, 1, 1, 0, 2);
  RmpPlan[CX1P][CX3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CX1P].is,
    dcmps[CX1P].ie, dcmps[CX1P].js, dcmps[CX1P].je, dcmps[CX1P].ks, dcmps[CX1P].ke,
    dcmps[CX3P].is, dcmps[CX3P].ie, dcmps[CX3P].js, dcmps[CX3P].je, dcmps[CX3P].ks,
    dcmps[CX3P].ke, 1, 2, 0, 2);
  RmpPlan[CX3P][CX2P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CX3P].ks,
    dcmps[CX3P].ke, dcmps[CX3P].is, dcmps[CX3P].ie, dcmps[CX3P].js, dcmps[CX3P].je,
    dcmps[CX2P].ks, dcmps[CX2P].ke, dcmps[CX2P].is, dcmps[CX2P].ie, dcmps[CX2P].js,
    dcmps[CX2P].je, 1, 2, 0, 2);
  RmpPlan[CX2P][CXB] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CX2P].js,
    dcmps[CX2P].je, dcmps[CX2P].ks, dcmps[CX2P].ke, dcmps[CX2P].is,
    dcmps[CX2P].ie, dcmps[CXB].js, dcmps[CXB].je, dcmps[CXB].ks, dcmps[CXB].ke,
    dcmps[CXB].is, dcmps[CXB].ie, 1, 2, 0, 2);

  RmpPlan[PB][P2P] = remap_3d_create_plan(MPI_COMM_WORLD,dcmps[PB].is,
    dcmps[PB].ie, dcmps[PB].js, dcmps[PB].je, dcmps[PB].ks, dcmps[PB].ke,
    dcmps[P2P].is, dcmps[P2P].ie, dcmps[P2P].js, dcmps[P2P].je,
    dcmps[P2P].ks, dcmps[P2P].ke, 1, 1, 0, 2);
  RmpPlan[P2P][P3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[P2P].js,
    dcmps[P2P].je, dcmps[P2P].ks, dcmps[P2P].ke, dcmps[P2P].is, dcmps[P2P].ie,
    dcmps[P3P].js, dcmps[P3P].je, dcmps[P3P].ks, dcmps[P3P].ke, dcmps[P3P].is,
    dcmps[P3P].ie, 1, 1, 0, 2);
  RmpPlan[P3P][P1P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[P3P].ks,
    dcmps[P3P].ke, dcmps[P3P].is, dcmps[P3P].ie, dcmps[P3P].js, dcmps[P3P].je,
    dcmps[P1P].ks, dcmps[P1P].ke, dcmps[P1P].is, dcmps[P1P].ie, dcmps[P1P].js,
    dcmps[P1P].je, 1, 1, 0, 2);
  RmpPlan[P1P][P3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[P1P].is,
    dcmps[P1P].ie, dcmps[P1P].js, dcmps[P1P].je, dcmps[P1P].ks, dcmps[P1P].ke,
    dcmps[P3P].is, dcmps[P3P].ie, dcmps[P3P].js, dcmps[P3P].je, dcmps[P3P].ks,
    dcmps[P3P].ke, 1, 2, 0, 2);
  RmpPlan[P3P][P2P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[P3P].ks,
    dcmps[P3P].ke, dcmps[P3P].is, dcmps[P3P].ie, dcmps[P3P].js, dcmps[P3P].je,
    dcmps[P2P].ks, dcmps[P2P].ke, dcmps[P2P].is, dcmps[P2P].ie, dcmps[P2P].js,
    dcmps[P2P].je, 1, 2, 0, 2);
  RmpPlan[P2P][PB] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[P2P].js,
    dcmps[P2P].je, dcmps[P2P].ks, dcmps[P2P].ke, dcmps[P2P].is,
    dcmps[P2P].ie, dcmps[PB].js, dcmps[PB].je, dcmps[PB].ks, dcmps[PB].ke,
    dcmps[PB].is, dcmps[PB].ie, 1, 2, 0, 2);

  RmpPlan[CEB][CE2P] = remap_3d_create_plan(MPI_COMM_WORLD,dcmps[CEB].is,
    dcmps[CEB].ie, dcmps[CEB].js, dcmps[CEB].je, dcmps[CEB].ks, dcmps[CEB].ke,
    dcmps[CE2P].is, dcmps[CE2P].ie, dcmps[CE2P].js, dcmps[CE2P].je,
    dcmps[CE2P].ks, dcmps[CE2P].ke, 1, 1, 0, 2);
  RmpPlan[CE2P][CE3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CE2P].js,
    dcmps[CE2P].je, dcmps[CE2P].ks, dcmps[CE2P].ke, dcmps[CE2P].is, dcmps[CE2P].ie,
    dcmps[CE3P].js, dcmps[CE3P].je, dcmps[CE3P].ks, dcmps[CE3P].ke, dcmps[CE3P].is,
    dcmps[CE3P].ie, 1, 1, 0, 2);
  RmpPlan[CE3P][CE1P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CE3P].ks,
    dcmps[CE3P].ke, dcmps[CE3P].is, dcmps[CE3P].ie, dcmps[CE3P].js, dcmps[CE3P].je,
    dcmps[CE1P].ks, dcmps[CE1P].ke, dcmps[CE1P].is, dcmps[CE1P].ie, dcmps[CE1P].js,
    dcmps[CE1P].je, 1, 1, 0, 2);
  RmpPlan[CE1P][CE3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CE1P].is,
    dcmps[CE1P].ie, dcmps[CE1P].js, dcmps[CE1P].je, dcmps[CE1P].ks, dcmps[CE1P].ke,
    dcmps[CE3P].is, dcmps[CE3P].ie, dcmps[CE3P].js, dcmps[CE3P].je, dcmps[CE3P].ks,
    dcmps[CE3P].ke, 1, 2, 0, 2);
  RmpPlan[CE3P][CE2P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CE3P].ks,
    dcmps[CE3P].ke, dcmps[CE3P].is, dcmps[CE3P].ie, dcmps[CE3P].js, dcmps[CE3P].je,
    dcmps[CE2P].ks, dcmps[CE2P].ke, dcmps[CE2P].is, dcmps[CE2P].ie, dcmps[CE2P].js,
    dcmps[CE2P].je, 1, 2, 0, 2);
  RmpPlan[CE2P][CEB] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[CE2P].js,
    dcmps[CE2P].je, dcmps[CE2P].ks, dcmps[CE2P].ke, dcmps[CE2P].is,
    dcmps[CE2P].ie, dcmps[CEB].js, dcmps[CEB].je, dcmps[CEB].ks, dcmps[CEB].ke,
    dcmps[CEB].is, dcmps[CEB].ie, 1, 2, 0, 2);

  for (int i=STH;i<=NTH;++i) {
    BndryRmpPlan[i][CBLOCK][FFT_FIRST] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][CBLOCK].js, bndry_dcmps[i][CBLOCK].je,
      bndry_dcmps[i][CBLOCK].ks, bndry_dcmps[i][CBLOCK].ke,
      bndry_dcmps[i][FFT_FIRST].js, bndry_dcmps[i][FFT_FIRST].je,
      bndry_dcmps[i][FFT_FIRST].ks, bndry_dcmps[i][FFT_FIRST].ke, 1, 0, 0, 2);
    BndryRmpPlan[i][FFT_FIRST][FFT_SECOND] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_FIRST].js, bndry_dcmps[i][FFT_FIRST].je,
      bndry_dcmps[i][FFT_FIRST].ks, bndry_dcmps[i][FFT_FIRST].ke,
      bndry_dcmps[i][FFT_SECOND].js, bndry_dcmps[i][FFT_SECOND].je,
      bndry_dcmps[i][FFT_SECOND].ks, bndry_dcmps[i][FFT_SECOND].ke, 1, 1, 0, 2);
    BndryRmpPlan[i][FFT_SECOND][CBLOCK] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_SECOND].ks, bndry_dcmps[i][FFT_SECOND].ke,
      bndry_dcmps[i][FFT_SECOND].js, bndry_dcmps[i][FFT_SECOND].je,
      bndry_dcmps[i][CBLOCK].ks, bndry_dcmps[i][CBLOCK].ke,
      bndry_dcmps[i][CBLOCK].js, bndry_dcmps[i][CBLOCK].je, 1, 1, 0, 2);
    int Nx2C = Nx2+2*ngh_;
    int Nx2S = Nx2+2*ngh_-2;
    int Nx3C = Nx3+2*ngh_;
    int Nx3S = Nx3+2*ngh_-2;
    fft_2d_plan[i][0] = fftw_plan_many_r2r(1, &(Nx2C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx2C,
      sigma[i], NULL, 1, Nx2C, sigma_mid[i][C], NULL,
      1, Nx2C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][1] = fftw_plan_many_r2r(1, &(Nx2S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx2C,
      sigma[i]+1, NULL, 1, Nx2C, sigma_mid[i][S]+1, NULL,
      1, Nx2C, DST, FFTW_MEASURE);
    fft_2d_plan[i][2] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][C], NULL, 1, Nx3C,
      sigma_fft[i][C][C], NULL, 1, Nx3C, DCT,
      FFTW_MEASURE);
    fft_2d_plan[i][3] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][C]+1, NULL, 1, Nx3C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx3C, DST,
      FFTW_MEASURE);
    fft_2d_plan[i][4] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][S], NULL, 1, Nx3C,
      sigma_fft[i][S][C], NULL, 1, Nx3C, DCT,
      FFTW_MEASURE);
    fft_2d_plan[i][5] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][S]+1, NULL, 1, Nx3C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx3C, DST,
      FFTW_MEASURE);
    // backward
    fft_2d_plan[i][6] = fftw_plan_many_r2r(1, &(Nx2C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx2C,
      sigma_fft[i][C][C], NULL, 1, Nx2C,
      sigma_fft[i][C][C], NULL, 1, Nx2C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][7] = fftw_plan_many_r2r(1, &(Nx2C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx2C,
      sigma_fft[i][C][S], NULL, 1, Nx2C,
      sigma_fft[i][C][S], NULL, 1, Nx2C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][8] = fftw_plan_many_r2r(1, &(Nx2S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx2C,
      sigma_fft[i][S][C]+1, NULL, 1, Nx2C,
      sigma_fft[i][S][C]+1, NULL, 1, Nx2C, DST, FFTW_MEASURE);
    fft_2d_plan[i][9] = fftw_plan_many_r2r(1, &(Nx2S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx2C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx2C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx2C, DST, FFTW_MEASURE);
    fft_2d_plan[i][10] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][C][C], NULL, 1, Nx3C,
      sigma_fft[i][C][C], NULL, 1, Nx3C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][11] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][S][C], NULL, 1, Nx3C,
      sigma_fft[i][S][C], NULL, 1, Nx3C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][12] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx3C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx3C, DST, FFTW_MEASURE);
    fft_2d_plan[i][13] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx3C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx3C, DST, FFTW_MEASURE);
  }
  for (int i=WST;i<=EST;++i) {
    BndryRmpPlan[i][CBLOCK][FFT_FIRST] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][CBLOCK].is, bndry_dcmps[i][CBLOCK].ie,
      bndry_dcmps[i][CBLOCK].ks, bndry_dcmps[i][CBLOCK].ke,
      bndry_dcmps[i][FFT_FIRST].is, bndry_dcmps[i][FFT_FIRST].ie,
      bndry_dcmps[i][FFT_FIRST].ks, bndry_dcmps[i][FFT_FIRST].ke, 1, 0, 0, 2);
    BndryRmpPlan[i][FFT_FIRST][FFT_SECOND] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_FIRST].is, bndry_dcmps[i][FFT_FIRST].ie,
      bndry_dcmps[i][FFT_FIRST].ks, bndry_dcmps[i][FFT_FIRST].ke,
      bndry_dcmps[i][FFT_SECOND].is, bndry_dcmps[i][FFT_SECOND].ie,
      bndry_dcmps[i][FFT_SECOND].ks, bndry_dcmps[i][FFT_SECOND].ke, 1, 1, 0, 2);
    BndryRmpPlan[i][FFT_SECOND][CBLOCK] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_SECOND].ks, bndry_dcmps[i][FFT_SECOND].ke,
      bndry_dcmps[i][FFT_SECOND].is, bndry_dcmps[i][FFT_SECOND].ie,
      bndry_dcmps[i][CBLOCK].ks, bndry_dcmps[i][CBLOCK].ke,
      bndry_dcmps[i][CBLOCK].is, bndry_dcmps[i][CBLOCK].ie, 1, 1, 0, 2);
    int Nx1C = Nx1+2*ngh_;
    int Nx1S = Nx1+2*ngh_-2;
    int Nx3C = Nx3+2*ngh_;
    int Nx3S = Nx3+2*ngh_-2;
    fft_2d_plan[i][0] = fftw_plan_many_r2r(1, &(Nx1C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma[i], NULL, 1, Nx1C, sigma_mid[i][C], NULL,
      1, Nx1C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][1] = fftw_plan_many_r2r(1, &(Nx1S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma[i]+1, NULL, 1, Nx1C, sigma_mid[i][S]+1, NULL,
      1, Nx1C, DST, FFTW_MEASURE);
    fft_2d_plan[i][2] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][C], NULL, 1, Nx3C,
      sigma_fft[i][C][C], NULL, 1, Nx3C, DCT,
      FFTW_MEASURE);
    fft_2d_plan[i][3] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][C]+1, NULL, 1, Nx3C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx3C, DST,
      FFTW_MEASURE);
    fft_2d_plan[i][4] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][S], NULL, 1, Nx3C,
      sigma_fft[i][S][C], NULL, 1, Nx3C, DCT,
      FFTW_MEASURE);
    fft_2d_plan[i][5] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_mid[i][S]+1, NULL, 1, Nx3C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx3C, DST,
      FFTW_MEASURE);
    // backward
    fft_2d_plan[i][6] = fftw_plan_many_r2r(1, &(Nx1C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][C][C], NULL, 1, Nx1C,
      sigma_fft[i][C][C], NULL, 1, Nx1C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][7] = fftw_plan_many_r2r(1, &(Nx1C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][C][S], NULL, 1, Nx1C,
      sigma_fft[i][C][S], NULL, 1, Nx1C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][8] = fftw_plan_many_r2r(1, &(Nx1S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][S][C]+1, NULL, 1, Nx1C,
      sigma_fft[i][S][C]+1, NULL, 1, Nx1C, DST, FFTW_MEASURE);
    fft_2d_plan[i][9] = fftw_plan_many_r2r(1, &(Nx1S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx1C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx1C, DST, FFTW_MEASURE);
    fft_2d_plan[i][10] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][C][C], NULL, 1, Nx3C,
      sigma_fft[i][C][C], NULL, 1, Nx3C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][11] = fftw_plan_many_r2r(1, &(Nx3C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][S][C], NULL, 1, Nx3C,
      sigma_fft[i][S][C], NULL, 1, Nx3C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][12] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx3C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx3C, DST, FFTW_MEASURE);
    fft_2d_plan[i][13] = fftw_plan_many_r2r(1, &(Nx3S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx3C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx3C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx3C, DST, FFTW_MEASURE);
  }
  for (int i=CBOT;i<=CTOP;++i) {
    BndryRmpPlan[i][CBLOCK][FFT_FIRST] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][CBLOCK].is, bndry_dcmps[i][CBLOCK].ie,
      bndry_dcmps[i][CBLOCK].js, bndry_dcmps[i][CBLOCK].je,
      bndry_dcmps[i][FFT_FIRST].is, bndry_dcmps[i][FFT_FIRST].ie,
      bndry_dcmps[i][FFT_FIRST].js, bndry_dcmps[i][FFT_FIRST].je, 1, 0, 0, 2);
    BndryRmpPlan[i][FFT_FIRST][FFT_SECOND] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_FIRST].is, bndry_dcmps[i][FFT_FIRST].ie,
      bndry_dcmps[i][FFT_FIRST].js, bndry_dcmps[i][FFT_FIRST].je,
      bndry_dcmps[i][FFT_SECOND].is, bndry_dcmps[i][FFT_SECOND].ie,
      bndry_dcmps[i][FFT_SECOND].js, bndry_dcmps[i][FFT_SECOND].je, 1, 1, 0, 2);
    BndryRmpPlan[i][FFT_SECOND][CBLOCK] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_SECOND].js, bndry_dcmps[i][FFT_SECOND].je,
      bndry_dcmps[i][FFT_SECOND].is, bndry_dcmps[i][FFT_SECOND].ie,
      bndry_dcmps[i][CBLOCK].js, bndry_dcmps[i][CBLOCK].je,
      bndry_dcmps[i][CBLOCK].is, bndry_dcmps[i][CBLOCK].ie, 1, 1, 0, 2);
    int Nx1C = Nx1+2*ngh_;
    int Nx1S = Nx1+2*ngh_-2;
    int Nx2C = Nx3+2*ngh_;
    int Nx2S = Nx3+2*ngh_-2;
    fft_2d_plan[i][0] = fftw_plan_many_r2r(1, &(Nx1C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma[i], NULL, 1, Nx1C, sigma_mid[i][C], NULL,
      1, Nx1C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][1] = fftw_plan_many_r2r(1, &(Nx1S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma[i]+1, NULL, 1, Nx1C, sigma_mid[i][S]+1, NULL,
      1, Nx1C, DST, FFTW_MEASURE);
    fft_2d_plan[i][2] = fftw_plan_many_r2r(1, &(Nx2C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_mid[i][C], NULL, 1, Nx2C,
      sigma_fft[i][C][C], NULL, 1, Nx2C, DCT,
      FFTW_MEASURE);
    fft_2d_plan[i][3] = fftw_plan_many_r2r(1, &(Nx2S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_mid[i][C]+1, NULL, 1, Nx2C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx2C, DST,
      FFTW_MEASURE);
    fft_2d_plan[i][4] = fftw_plan_many_r2r(1, &(Nx2C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_mid[i][S], NULL, 1, Nx2C,
      sigma_fft[i][S][C], NULL, 1, Nx2C, DCT,
      FFTW_MEASURE);
    fft_2d_plan[i][5] = fftw_plan_many_r2r(1, &(Nx2S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_mid[i][S]+1, NULL, 1, Nx2C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx2C, DST,
      FFTW_MEASURE);
    // backward
    fft_2d_plan[i][6] = fftw_plan_many_r2r(1, &(Nx1C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][C][C], NULL, 1, Nx1C,
      sigma_fft[i][C][C], NULL, 1, Nx1C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][7] = fftw_plan_many_r2r(1, &(Nx1C),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][C][S], NULL, 1, Nx1C,
      sigma_fft[i][C][S], NULL, 1, Nx1C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][8] = fftw_plan_many_r2r(1, &(Nx1S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][S][C]+1, NULL, 1, Nx1C,
      sigma_fft[i][S][C]+1, NULL, 1, Nx1C, DST, FFTW_MEASURE);
    fft_2d_plan[i][9] = fftw_plan_many_r2r(1, &(Nx1S),
      bndry_dcmps[i][FFT_FIRST].block_size/Nx1C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx1C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx1C, DST, FFTW_MEASURE);
    fft_2d_plan[i][10] = fftw_plan_many_r2r(1, &(Nx2C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_fft[i][C][C], NULL, 1, Nx2C,
      sigma_fft[i][C][C], NULL, 1, Nx2C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][11] = fftw_plan_many_r2r(1, &(Nx2C),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_fft[i][S][C], NULL, 1, Nx2C,
      sigma_fft[i][S][C], NULL, 1, Nx2C, DCT, FFTW_MEASURE);
    fft_2d_plan[i][12] = fftw_plan_many_r2r(1, &(Nx2S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx2C,
      sigma_fft[i][C][S]+1, NULL, 1, Nx2C, DST, FFTW_MEASURE);
    fft_2d_plan[i][13] = fftw_plan_many_r2r(1, &(Nx2S),
      bndry_dcmps[i][FFT_SECOND].block_size/Nx2C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx2C,
      sigma_fft[i][S][S]+1, NULL, 1, Nx2C, DST, FFTW_MEASURE);
  }
//  FillCntGrf();
  FillDscGrf();
}

// OBCGravityCar destructor
OBCGravityCar::~OBCGravityCar()
{
  for (int i=STH;i<=CTOP;++i) {
    fftw_free(sigma[i]);
    for (int j=C;j<=S;++j) {
      fftw_free(sigma_mid[i][j]);
      for (int k=C;k<=S;++k ) {
        fftw_free(sigma_fft[i][j][k]);
      }
    }
  }
  for (int i=C;i<=S;++i) {
    for (int j=C;j<=S;++j) {
      for (int k=C;k<=S;++k) {
        fftw_free(sigfft[i][j][k]);
      }
    }
  }
  fftw_free(grf);
  lambda1_.DeleteAthenaArray();
  lambda2_.DeleteAthenaArray();
  lambda3_.DeleteAthenaArray();
  lambda11_.DeleteAthenaArray();
  lambda22_.DeleteAthenaArray();
  lambda33_.DeleteAthenaArray();
  for (int i=0;i<15;++i) fftw_destroy_plan(fft_plan_r2r_[i]);
  fftw_free(in_);
  fftw_free(in2_);
  fftw_free(out_);
  fftw_free(buf_);
  remap_3d_destroy_plan(RmpPlan[CXB][CX2P]);
  remap_3d_destroy_plan(RmpPlan[CX2P][CX3P]);
  remap_3d_destroy_plan(RmpPlan[CX3P][CX1P]);
  remap_3d_destroy_plan(RmpPlan[CX1P][CX3P]);
  remap_3d_destroy_plan(RmpPlan[CX3P][CX2P]);
  remap_3d_destroy_plan(RmpPlan[CX2P][CXB]);
  remap_3d_destroy_plan(RmpPlan[PB][P2P]);
  remap_3d_destroy_plan(RmpPlan[P2P][P3P]);
  remap_3d_destroy_plan(RmpPlan[P3P][P1P]);
  remap_3d_destroy_plan(RmpPlan[P1P][P3P]);
  remap_3d_destroy_plan(RmpPlan[P3P][P2P]);
  remap_3d_destroy_plan(RmpPlan[P2P][PB]);
  remap_3d_destroy_plan(RmpPlan[CEB][CE2P]);
  remap_3d_destroy_plan(RmpPlan[CE2P][CE3P]);
  remap_3d_destroy_plan(RmpPlan[CE3P][CE1P]);
  remap_3d_destroy_plan(RmpPlan[CE1P][CE3P]);
  remap_3d_destroy_plan(RmpPlan[CE3P][CE2P]);
  remap_3d_destroy_plan(RmpPlan[CE2P][CEB]);


  for (int i=STH;i<=CTOP;++i) {
    remap_2d_destroy_plan(BndryRmpPlan[i][CBLOCK][FFT_FIRST]);
    remap_2d_destroy_plan(BndryRmpPlan[i][FFT_FIRST][FFT_SECOND]);
    remap_2d_destroy_plan(BndryRmpPlan[i][FFT_SECOND][CBLOCK]);
    for (int j=0;j<14;++j) {
      fftw_destroy_plan(fft_2d_plan[i][j]);
    }
  }
  for (int i=STH;i<=CTOP;++i) {
    if (bndcomm[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&(bndcomm[i]));
    }
  }
  MPI_Comm_free(&x1comm);
  MPI_Comm_free(&x2comm);
  MPI_Comm_free(&x3comm);
}

void OBCGravityCar::BndFFTBackward(int first_nslow, int first_nfast, int second_nslow, int second_nfast, int B)
{
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      remap_2d(sigma_fft[B][p1][p2], sigma_fft[B][p1][p2], buf_, BndryRmpPlan[B][CBLOCK][FFT_FIRST]);
      for (int k=0;k<first_nslow;++k) {
        for (int j=1;j<first_nfast-1;++j) {
          int idx = j + first_nfast*k;
          sigma_fft[B][p1][p2][idx] *= 0.5;
        }
      }
    }
  }
  fftw_execute(fft_2d_plan[B][6]);
  fftw_execute(fft_2d_plan[B][7]);
  fftw_execute(fft_2d_plan[B][8]);
  fftw_execute(fft_2d_plan[B][9]);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      remap_2d(sigma_fft[B][p1][p2], sigma_fft[B][p1][p2], buf_, BndryRmpPlan[B][FFT_FIRST][FFT_SECOND]);
      for (int j=0;j<second_nslow;++j) {
        for (int k=1;k<second_nfast-1;++k) {
          int idx = k + second_nfast*j;
          sigma_fft[B][p1][p2][idx] *= 0.5;
        }
      }
    }
  }
  fftw_execute(fft_2d_plan[B][10]);
  fftw_execute(fft_2d_plan[B][11]);
  fftw_execute(fft_2d_plan[B][12]);
  fftw_execute(fft_2d_plan[B][13]);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      remap_2d(sigma_fft[B][p1][p2], sigma_fft[B][p1][p2], buf_, BndryRmpPlan[B][FFT_SECOND][CBLOCK]);
    }
  }
}


void OBCGravityCar::BndFFTForward(int first_nslow, int first_nfast, int second_nslow, int second_nfast, int B)
{
  remap_2d(sigma[B], sigma[B], buf_, BndryRmpPlan[B][CBLOCK][FFT_FIRST]);
  for (int k=0;k<first_nslow;++k) {
    for (int j=1;j<first_nfast-1;++j) {
      int idx = j + first_nfast*k;
      sigma[B][idx] *= 0.5;
    }
  }
  fftw_execute(fft_2d_plan[B][0]);
  fftw_execute(fft_2d_plan[B][1]);
  for (int k=0;k<first_nslow;++k) {
    int idx1 = 0       + first_nfast*k;
    int idx2 = first_nfast-1 + first_nfast*k;
    sigma_mid[B][S][idx1] = 0;
    sigma_mid[B][S][idx2] = 0;
  }
  for (int p1=C;p1<=S;++p1) {
    remap_2d(sigma_mid[B][p1], sigma_mid[B][p1], buf_, BndryRmpPlan[B][FFT_FIRST][FFT_SECOND]);
    for (int j=0;j<second_nslow;++j) {
      for (int k=1;k<second_nfast-1;++k) {
        int idx = k + second_nfast*j;
        sigma_mid[B][p1][idx] *= 0.5;
      }
    }
  }
  fftw_execute(fft_2d_plan[B][2]);
  fftw_execute(fft_2d_plan[B][3]);
  fftw_execute(fft_2d_plan[B][4]);
  fftw_execute(fft_2d_plan[B][5]);
  for (int p1=C;p1<=S;++p1) {
    for (int j=0;j<second_nslow;++j) {
      int idx1 = 0       + second_nfast*j;
      int idx2 = second_nfast-1 + second_nfast*j;
      sigma_fft[B][p1][S][idx1] = 0;
      sigma_fft[B][p1][S][idx2] = 0;
    }
    for (int p2=C;p2<=S;++p2) {
      remap_2d(sigma_fft[B][p1][p2], sigma_fft[B][p1][p2], buf_, BndryRmpPlan[B][FFT_SECOND][CBLOCK]);
    }
  }
}

void OBCGravityCar::FillDscGrf()
{
  std::cout << "Populating discrete Green's function..." << std::endl;
  Real x,y,z,rds,vol,psi,normfac;
  vol = dx1_*dx2_*dx3_;
  int ishift = 0;
  int jshift = 0;
  int kshift = 0;
  if (dcmps[CXB].is == 0) ishift=ngh_grf_;
  if (dcmps[CXB].js == 0) jshift=ngh_grf_;
  if (dcmps[CXB].ks == 0) kshift=ngh_grf_;

  // Add point charge
  for (int k=0;k<dcmps[CEB].nx3;++k) {
    for (int j=0;j<dcmps[CEB].nx2;++j) {
      for (int i=0;i<dcmps[CEB].nx1;++i) {
        int idx = i + dcmps[CEB].nx1*(j + dcmps[CEB].nx2*k);
        int gi = dcmps[CEB].is + i;
        int gj = dcmps[CEB].js + j;
        int gk = dcmps[CEB].ks + k;
        if ((gi==ngh_grf_)&&(gj==ngh_grf_)&&(gk==ngh_grf_)) {
          in2_[idx] = 1.0;
        }
        else {
          in2_[idx] = 0.0;
        }
      }
    }
  }
  // Add boundary charge
  for (int k=0;k<dcmps[CEB].nx3;++k) {
    int gk = dcmps[CEB].ks + k;
    for (int j=0;j<dcmps[CEB].nx2;++j) {
      int gj = dcmps[CEB].js + j;
      if (dcmps[CXB].is == 0) {
        int i = 0;
        int gi = dcmps[CEB].is-1;
        int idx = i + dcmps[CEB].nx1*(j + dcmps[CEB].nx2*k);
        x = (gi-ngh_grf_)*dx1_;
        y = (gj-ngh_grf_)*dx2_;
        z = (gk-ngh_grf_)*dx3_;
        rds = sqrt(x*x+y*y+z*z);
        psi = -(four_pi_G/4.0/PI)*vol/rds;
        in2_[idx] -= psi/four_pi_G/SQR(dx1_);
      }
      if (dcmps[CXB].ie == Nx1-1) {
        int i = dcmps[CEB].nx1-1;
        int gi = dcmps[CEB].ie+1;
        int idx = i + dcmps[CEB].nx1*(j + dcmps[CEB].nx2*k);
        x = (gi-ngh_grf_)*dx1_;
        y = (gj-ngh_grf_)*dx2_;
        z = (gk-ngh_grf_)*dx3_;
        rds = sqrt(x*x+y*y+z*z);
        psi = -(four_pi_G/4.0/PI)*vol/rds;
        in2_[idx] -= psi/four_pi_G/SQR(dx1_);
      }
    }
  }
  for (int k=0;k<dcmps[CEB].nx3;++k) {
    int gk = dcmps[CEB].ks + k;
    for (int i=0;i<dcmps[CEB].nx1;++i) {
      int gi = dcmps[CEB].is + i;
      if (dcmps[CXB].js == 0) {
        int j = 0;
        int gj = dcmps[CEB].js-1;
        int idx = i + dcmps[CEB].nx1*(j + dcmps[CEB].nx2*k);
        x = (gi-ngh_grf_)*dx1_;
        y = (gj-ngh_grf_)*dx2_;
        z = (gk-ngh_grf_)*dx3_;
        rds = sqrt(x*x+y*y+z*z);
        psi = -(four_pi_G/4.0/PI)*vol/rds;
        in2_[idx] -= psi/four_pi_G/SQR(dx2_);
      }
      if (dcmps[CXB].je == Nx2-1) {
        int j = dcmps[CEB].nx2-1;
        int gj = dcmps[CEB].je+1;
        int idx = i + dcmps[CEB].nx1*(j + dcmps[CEB].nx2*k);
        x = (gi-ngh_grf_)*dx1_;
        y = (gj-ngh_grf_)*dx2_;
        z = (gk-ngh_grf_)*dx3_;
        rds = sqrt(x*x+y*y+z*z);
        psi = -(four_pi_G/4.0/PI)*vol/rds;
        in2_[idx] -= psi/four_pi_G/SQR(dx2_);
      }
    }
  }
  for (int j=0;j<dcmps[CEB].nx2;++j) {
    int gj = dcmps[CEB].js + j;
    for (int i=0;i<dcmps[CEB].nx1;++i) {
      int gi = dcmps[CEB].is + i;
      if (dcmps[CXB].ks == 0) {
        int k = 0;
        int gk = dcmps[CEB].ks-1;
        int idx = i + dcmps[CEB].nx1*(j + dcmps[CEB].nx2*k);
        x = (gi-ngh_grf_)*dx1_;
        y = (gj-ngh_grf_)*dx2_;
        z = (gk-ngh_grf_)*dx3_;
        rds = sqrt(x*x+y*y+z*z);
        psi = -(four_pi_G/4.0/PI)*vol/rds;
        in2_[idx] -= psi/four_pi_G/SQR(dx3_);
      }
      if (dcmps[CXB].ke == Nx3-1) {
        int k = dcmps[CEB].nx3-1;
        int gk = dcmps[CEB].ke+1;
        int idx = i + dcmps[CEB].nx1*(j + dcmps[CEB].nx2*k);
        x = (gi-ngh_grf_)*dx1_;
        y = (gj-ngh_grf_)*dx2_;
        z = (gk-ngh_grf_)*dx3_;
        rds = sqrt(x*x+y*y+z*z);
        psi = -(four_pi_G/4.0/PI)*vol/rds;
        in2_[idx] -= psi/four_pi_G/SQR(dx3_);
      }
    }
  }

  remap_3d(in2_, in2_, buf_, RmpPlan[CEB][CE2P]);
  fftw_execute(fft_plan_r2r_[9]);
  remap_3d(in2_, in2_, buf_, RmpPlan[CE2P][CE3P]);
  fftw_execute(fft_plan_r2r_[10]);
  remap_3d(in2_, in2_, buf_, RmpPlan[CE3P][CE1P]);
  fftw_execute(fft_plan_r2r_[11]);
  /* apply kernel */
  for (int k=0;k<dcmps[CE1P].nx3;++k) {
    for (int j=0;j<dcmps[CE1P].nx2;++j) {
      for (int i=0;i<dcmps[CE1P].nx1;++i) {
        int idx = i + dcmps[CE1P].nx1*(j + dcmps[CE1P].nx2*k);
        in2_[idx] *= four_pi_G/(lambda11_(i)+lambda22_(j)+lambda33_(k));
      }
    }
  }
  fftw_execute(fft_plan_r2r_[12]);
  remap_3d(in2_, in2_, buf_, RmpPlan[CE1P][CE3P]);
  fftw_execute(fft_plan_r2r_[13]);
  remap_3d(in2_, in2_, buf_, RmpPlan[CE3P][CE2P]);
  fftw_execute(fft_plan_r2r_[14]);
  remap_3d(in2_, in2_, buf_, RmpPlan[CE2P][CEB]);
  normfac = 1.0 / (2*(Nx3+1+2*ngh_+ngh_grf_)) / (2*(Nx2+1+2*ngh_+ngh_grf_)) / (2*(Nx1+1+2*ngh_+ngh_grf_));
  for (int k=0;k<dcmps[PB].nx3;++k) {
    for (int j=0;j<dcmps[PB].nx2;++j) {
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
        int idx2 = (i+ishift) + dcmps[CEB].nx1*((j+jshift) + dcmps[CEB].nx2*(k+kshift));
        grf[idx] = in2_[idx2]*normfac;
      }
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[PB][P2P]);
  fftw_execute(fft_plan_r2r_[6]);
  for (int i=0;i<dcmps[P2P].nx1;++i) {
    for (int k=0;k<dcmps[P2P].nx3;++k) {
      int idx = 0 + dcmps[P2P].nx2*(k + dcmps[P2P].nx3*i);
      grf[idx] *= 0.5;
      idx = dcmps[P2P].nx2-1 + dcmps[P2P].nx2*(k + dcmps[P2P].nx3*i);
      grf[idx] *= 0.5;
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[P2P][P3P]);
  fftw_execute(fft_plan_r2r_[7]);
  for (int j=0;j<dcmps[P3P].nx2;++j) {
    for (int i=0;i<dcmps[P3P].nx1;++i) {
      int idx = 0 + dcmps[P3P].nx3*(i + dcmps[P3P].nx1*j);
      grf[idx] *= 0.5;
      idx = dcmps[P3P].nx3-1 + dcmps[P3P].nx3*(i + dcmps[P3P].nx1*j);
      grf[idx] *= 0.5;
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[P3P][P1P]);
  fftw_execute(fft_plan_r2r_[8]);
  for (int k=0;k<dcmps[P1P].nx3;++k) {
    for (int j=0;j<dcmps[P1P].nx2;++j) {
      int idx = 0 + dcmps[P1P].nx1*(j + dcmps[P1P].nx2*k);
      grf[idx] *= 0.5;
      idx = dcmps[P1P].nx1-1 + dcmps[P1P].nx1*(j + dcmps[P1P].nx2*k);
      grf[idx] *= 0.5;
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[P1P][P3P]);
  remap_3d(grf, grf, buf_, RmpPlan[P3P][P2P]);
  remap_3d(grf, grf, buf_, RmpPlan[P2P][PB]);
  normfac = 1.0 / (Nx1-1+2*ngh_) / (Nx2-1+2*ngh_) / (Nx3-1+2*ngh_);
  for (int k=0;k<dcmps[PB].nx3;++k) {
    for (int j=0;j<dcmps[PB].nx2;++j) {
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
        grf[idx] *= normfac;
      }
    }
  }
  std::cout << "FillDscGrf is done" << std::endl;
}

void OBCGravityCar::FillCntGrf()
{
  std::cout << "Populating continuous Green's function..." << std::endl;
  Coordinates *pcoord = pmy_block->pcoord;
  Real x,y,z,rds,vol;
  int gi,gj,gk;

  vol = dx1_*dx2_*dx3_;

  for (int k=0;k<dcmps[PB].nx3;++k) {
    for (int j=0;j<dcmps[PB].nx2;++j) {
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
        gi = dcmps[PB].is + i;
        gj = dcmps[PB].js + j;
        gk = dcmps[PB].ks + k;
        x = gi*dx1_;
        y = gj*dx2_;
        z = gk*dx3_;
        rds = sqrt(x*x+y*y+z*z);
        if ((gi==0)&&(gj==0)&&(gk==0)) {
          grf[idx] = 0;
        }
        else {
          grf[idx] = -(four_pi_G/4.0/PI)*vol/rds;
        }
      }
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[PB][P2P]);
  fftw_execute(fft_plan_r2r_[6]);
  for (int i=0;i<dcmps[P2P].nx1;++i) {
    for (int k=0;k<dcmps[P2P].nx3;++k) {
      int idx = 0 + dcmps[P2P].nx2*(k + dcmps[P2P].nx3*i);
      grf[idx] *= 0.5;
      idx = dcmps[P2P].nx2-1 + dcmps[P2P].nx2*(k + dcmps[P2P].nx3*i);
      grf[idx] *= 0.5;
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[P2P][P3P]);
  fftw_execute(fft_plan_r2r_[7]);
  for (int j=0;j<dcmps[P3P].nx2;++j) {
    for (int i=0;i<dcmps[P3P].nx1;++i) {
      int idx = 0 + dcmps[P3P].nx3*(i + dcmps[P3P].nx1*j);
      grf[idx] *= 0.5;
      idx = dcmps[P3P].nx3-1 + dcmps[P3P].nx3*(i + dcmps[P3P].nx1*j);
      grf[idx] *= 0.5;
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[P3P][P1P]);
  fftw_execute(fft_plan_r2r_[8]);
  for (int k=0;k<dcmps[P1P].nx3;++k) {
    for (int j=0;j<dcmps[P1P].nx2;++j) {
      int idx = 0 + dcmps[P1P].nx1*(j + dcmps[P1P].nx2*k);
      grf[idx] *= 0.5;
      idx = dcmps[P1P].nx1-1 + dcmps[P1P].nx1*(j + dcmps[P1P].nx2*k);
      grf[idx] *= 0.5;
    }
  }
  remap_3d(grf, grf, buf_, RmpPlan[P1P][P3P]);
  remap_3d(grf, grf, buf_, RmpPlan[P3P][P2P]);
  remap_3d(grf, grf, buf_, RmpPlan[P2P][PB]);
  Real normfac = 1.0 / (Nx1-1+2*ngh_) / (Nx2-1+2*ngh_) / (Nx3-1+2*ngh_);
  for (int k=0;k<dcmps[PB].nx3;++k) {
    for (int j=0;j<dcmps[PB].nx2;++j) {
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
        grf[idx] *= normfac;
      }
    }
  }
  std::cout << "FillCntGrf is done" << std::endl;
}

void OBCGravityCar::LoadSource(const AthenaArray<Real> &src)
{
  int is, ie, js, je, ks, ke;
  int idx;
  is = pmy_block->is; js = pmy_block->js; ks = pmy_block->ks;
  ie = pmy_block->ie; je = pmy_block->je; ke = pmy_block->ke;
  for (int k=ks;k<=ke;++k) {
    for (int j=js;j<=je;++j) {
      for (int i=is;i<=ie;++i) {
        idx = (i-is) + nx1*((j-js) + nx2*(k-ks));
        in_[idx] = src(IDN,k,j,i);
      }
    }
  }
}

void OBCGravityCar::SolveZeroBC()
{
  remap_3d(in_, in_, buf_, RmpPlan[CXB][CX2P]);
  fftw_execute(fft_plan_r2r_[0]);
  remap_3d(in_, in_, buf_, RmpPlan[CX2P][CX3P]);
  fftw_execute(fft_plan_r2r_[1]);
  remap_3d(in_, in_, buf_, RmpPlan[CX3P][CX1P]);
  fftw_execute(fft_plan_r2r_[2]);
  /* apply kernel */
  int idx;
  for (int k=0;k<dcmps[CX1P].nx3;++k) {
    for (int j=0;j<dcmps[CX1P].nx2;++j) {
      for (int i=0;i<dcmps[CX1P].nx1;++i) {
        idx = i + dcmps[CX1P].nx1*(j + dcmps[CX1P].nx2*k);
        out_[idx] = four_pi_G*in_[idx]/(lambda1_(i)+lambda2_(j)+lambda3_(k));
      }
    }
  }
  fftw_execute(fft_plan_r2r_[3]);
  remap_3d(out_, out_, buf_, RmpPlan[CX1P][CX3P]);
  fftw_execute(fft_plan_r2r_[4]);
  remap_3d(out_, out_, buf_, RmpPlan[CX3P][CX2P]);
  fftw_execute(fft_plan_r2r_[5]);
  remap_3d(out_, out_, buf_, RmpPlan[CX2P][CXB]);
  int is,ie,js,je,ks,ke;
  is = pmy_block->is; js = pmy_block->js; ks = pmy_block->ks;
  ie = pmy_block->ie; je = pmy_block->je; ke = pmy_block->ke;
  Real normfac = 1.0 / (2*(Nx3+1)) / (2*(Nx2+1)) / (2*(Nx1+1));
  for (int k=ks;k<=ke;++k) {
    for (int j=js;j<=je;++j) {
      for (int i=is;i<=ie;++i) {
        idx = (i-is) + dcmps[CXB].nx1*((j-js) + dcmps[CXB].nx2*(k-ks));
        out_[idx] *= normfac;
      }
    }
  }
}

void OBCGravityCar::CalcBndCharge()
{
  RegionSize& mesh_size = pmy_block->pmy_mesh->mesh_size;
  RegionSize& block_size = pmy_block->block_size;
  Real normfac1 = 1.0 / (four_pi_G*dx1_*dx1_);
  Real normfac2 = 1.0 / (four_pi_G*dx2_*dx2_);
  Real normfac3 = 1.0 / (four_pi_G*dx3_*dx3_);
  int gis,gie,gjs,gje,gks,gke;
  gis = (pmy_block->loc.lx1)*nx1;
  gie = gis + nx1 - 1;
  gjs = (pmy_block->loc.lx2)*nx2;
  gje = gjs + nx2 - 1;
  gks = (pmy_block->loc.lx3)*nx3;
  gke = gks + nx3 - 1;

  // calculate boundary charges and store in the procs responsible for the boundary
  if (gis == 0) {
    int jshift=0, kshift=0;
    if (bndry_dcmps[STH][CBLOCK].js==0) jshift=ngh_;
    if (bndry_dcmps[STH][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[STH][CBLOCK].nx3;++k) {
      for (int j=0;j<bndry_dcmps[STH][CBLOCK].nx2;++j) {
        int idx2 = j + bndry_dcmps[STH][CBLOCK].nx2*k;
        sigma[STH][idx2] = 0;
      }
    }
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int j=0;j<dcmps[CXB].nx2;++j) {
        int idx2 = (j+jshift) + bndry_dcmps[STH][CBLOCK].nx2*(k+kshift);
        int idx = 0 + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*k);
        sigma[STH][idx2] = out_[idx]*normfac1;
      }
    }
  }
  if (gie == Nx1-1) {
    int jshift=0, kshift=0;
    if (bndry_dcmps[NTH][CBLOCK].js==0) jshift=ngh_;
    if (bndry_dcmps[NTH][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[NTH][CBLOCK].nx3;++k) {
      for (int j=0;j<bndry_dcmps[NTH][CBLOCK].nx2;++j) {
        int idx2 = j + bndry_dcmps[NTH][CBLOCK].nx2*k;
        sigma[NTH][idx2] = 0;
      }
    }
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int j=0;j<dcmps[CXB].nx2;++j) {
        int idx2 = (j+jshift) + bndry_dcmps[NTH][CBLOCK].nx2*(k+kshift);
        int idx = dcmps[CXB].nx1-1 + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*k);
        sigma[NTH][idx2] = out_[idx]*normfac1;
      }
    }
  }
  if (gjs == 0) {
    int ishift=0, kshift=0;
    if (bndry_dcmps[WST][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[WST][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[WST][CBLOCK].nx3;++k) {
      for (int i=0;i<bndry_dcmps[WST][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[WST][CBLOCK].nx1*k;
        sigma[WST][idx2] = 0;
      }
    }
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[WST][CBLOCK].nx1*(k+kshift);
        int idx = i + dcmps[CXB].nx1*(0 + dcmps[CXB].nx2*k);
        sigma[WST][idx2] = out_[idx]*normfac2;
      }
    }
  }
  if (gje == Nx2-1) {
    int ishift=0, kshift=0;
    if (bndry_dcmps[EST][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[EST][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[EST][CBLOCK].nx3;++k) {
      for (int i=0;i<bndry_dcmps[EST][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[EST][CBLOCK].nx1*k;
        sigma[EST][idx2] = 0;
      }
    }
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[EST][CBLOCK].nx1*(k+kshift);
        int idx = i + dcmps[CXB].nx1*(dcmps[CXB].nx2-1 + dcmps[CXB].nx2*k);
        sigma[EST][idx2] = out_[idx]*normfac2;
      }
    }
  }
  if (gks == 0) {
    int ishift=0, jshift=0;
    if (bndry_dcmps[CBOT][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[CBOT][CBLOCK].js==0) jshift=ngh_;
    for (int j=0;j<bndry_dcmps[CBOT][CBLOCK].nx2;++j) {
      for (int i=0;i<bndry_dcmps[CBOT][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[CBOT][CBLOCK].nx1*j;
        sigma[CBOT][idx2] = 0;
      }
    }
    for (int j=0;j<dcmps[CXB].nx2;++j) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[CBOT][CBLOCK].nx1*(j+jshift);
        int idx = i + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*(0));
        sigma[CBOT][idx2] = out_[idx]*normfac3;
      }
    }
  }
  if (gke == Nx3-1) {
    int ishift=0, jshift=0;
    if (bndry_dcmps[CTOP][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[CTOP][CBLOCK].js==0) jshift=ngh_;
    for (int j=0;j<bndry_dcmps[CTOP][CBLOCK].nx2;++j) {
      for (int i=0;i<bndry_dcmps[CTOP][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[CTOP][CBLOCK].nx1*j;
        sigma[CTOP][idx2] = 0;
      }
    }
    for (int j=0;j<dcmps[CXB].nx2;++j) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[CTOP][CBLOCK].nx1*(j+jshift);
        int idx = i + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*(dcmps[CXB].nx3-1));
        sigma[CTOP][idx2] = out_[idx]*normfac3;
      }
    }
  }
}

void OBCGravityCar::CalcBndPot()
{
  MPI_Request req[6];
  BndFFTForward(bndry_dcmps[STH][FFT_FIRST].nx3, bndry_dcmps[STH][FFT_FIRST].nx2, 
    bndry_dcmps[STH][FFT_SECOND].nx2, bndry_dcmps[STH][FFT_SECOND].nx3, STH);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
//      MPI_Ibcast(sigma_fft[STH][p1][p2], dcmps[PB].nx2*dcmps[PB].nx3,
//       MPI_DOUBLE, 0, x1comm, &req[STH]);
      MPI_Bcast(sigma_fft[STH][p1][p2], dcmps[PB].nx2*dcmps[PB].nx3,
       MPI_DOUBLE, 0, x1comm);
    }
  }
  BndFFTForward(bndry_dcmps[NTH][FFT_FIRST].nx3, bndry_dcmps[NTH][FFT_FIRST].nx2,
    bndry_dcmps[NTH][FFT_SECOND].nx2, bndry_dcmps[NTH][FFT_SECOND].nx3, NTH);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
//      MPI_Ibcast(sigma_fft[NTH][p1][p2], dcmps[PB].nx2*dcmps[PB].nx3,
//       MPI_DOUBLE, x1comm_size-1, x1comm, &req[NTH]);
      MPI_Bcast(sigma_fft[NTH][p1][p2], dcmps[PB].nx2*dcmps[PB].nx3,
       MPI_DOUBLE, x1comm_size-1, x1comm);
    }
  }
  BndFFTForward(bndry_dcmps[WST][FFT_FIRST].nx3, bndry_dcmps[WST][FFT_FIRST].nx1,
    bndry_dcmps[WST][FFT_SECOND].nx1, bndry_dcmps[WST][FFT_SECOND].nx3, WST);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
//      MPI_Ibcast(sigma_fft[WST][p1][p2], dcmps[PB].nx1*dcmps[PB].nx3,
//       MPI_DOUBLE, 0, x2comm, &req[WST]);
      MPI_Bcast(sigma_fft[WST][p1][p2], dcmps[PB].nx1*dcmps[PB].nx3,
       MPI_DOUBLE, 0, x2comm);
    }
  }
  BndFFTForward(bndry_dcmps[EST][FFT_FIRST].nx3, bndry_dcmps[EST][FFT_FIRST].nx1,
    bndry_dcmps[EST][FFT_SECOND].nx1, bndry_dcmps[EST][FFT_SECOND].nx3, EST);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
//      MPI_Ibcast(sigma_fft[EST][p1][p2], dcmps[PB].nx1*dcmps[PB].nx3,
//       MPI_DOUBLE, x2comm_size-1, x2comm, &req[EST]);
      MPI_Bcast(sigma_fft[EST][p1][p2], dcmps[PB].nx1*dcmps[PB].nx3,
       MPI_DOUBLE, x2comm_size-1, x2comm);
    }
  }
  BndFFTForward(bndry_dcmps[CBOT][FFT_FIRST].nx2, bndry_dcmps[CBOT][FFT_FIRST].nx1,
    bndry_dcmps[CBOT][FFT_SECOND].nx1, bndry_dcmps[CBOT][FFT_SECOND].nx2, CBOT);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
//      MPI_Ibcast(sigma_fft[CBOT][p1][p2], dcmps[PB].nx1*dcmps[PB].nx2,
//       MPI_DOUBLE, 0, x3comm, &req[CBOT]);
      MPI_Bcast(sigma_fft[CBOT][p1][p2], dcmps[PB].nx1*dcmps[PB].nx2,
       MPI_DOUBLE, 0, x3comm);
    }
  }
  BndFFTForward(bndry_dcmps[CTOP][FFT_FIRST].nx2, bndry_dcmps[CTOP][FFT_FIRST].nx1,
    bndry_dcmps[CTOP][FFT_SECOND].nx1, bndry_dcmps[CTOP][FFT_SECOND].nx2, CTOP);
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
//      MPI_Ibcast(sigma_fft[CTOP][p1][p2], dcmps[PB].nx1*dcmps[PB].nx2,
//       MPI_DOUBLE, x3comm_size-1, x3comm, &req[CTOP]);
      MPI_Bcast(sigma_fft[CTOP][p1][p2], dcmps[PB].nx1*dcmps[PB].nx2,
       MPI_DOUBLE, x3comm_size-1, x3comm);
    }
  }
//  MPI_Waitall(6, req, MPI_STATUSES_IGNORE);

  int sgn1, sgn2, sgn3;
  for (int k=0;k<dcmps[PB].nx3;++k) {
    sgn3 = ((dcmps[PB].ks+k)%2)*2 - 1;
    for (int j=0;j<dcmps[PB].nx2;++j) {
      sgn2 = ((dcmps[PB].js+j)%2)*2 - 1;
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
        int idx1 = j + dcmps[PB].nx2*k;
        int idx2 = i + dcmps[PB].nx1*k;
        int idx3 = i + dcmps[PB].nx1*j;
        sgn1 = ((dcmps[PB].is+i)%2)*2 - 1;
        sigfft[C][C][C][idx] = sigma_fft[STH][C][C][idx1] - sgn1*sigma_fft[NTH][C][C][idx1]
                             + sigma_fft[WST][C][C][idx2] - sgn2*sigma_fft[EST][C][C][idx2]
                             + sigma_fft[CBOT][C][C][idx3] - sgn3*sigma_fft[CTOP][C][C][idx3];
        sigfft[C][C][S][idx] = sigma_fft[STH][C][S][idx1] - sgn1*sigma_fft[NTH][C][S][idx1]
                             + sigma_fft[WST][C][S][idx2] - sgn2*sigma_fft[EST][C][S][idx2];
        sigfft[C][S][C][idx] = sigma_fft[STH][S][C][idx1] - sgn1*sigma_fft[NTH][S][C][idx1]
                             + sigma_fft[CBOT][C][S][idx3] - sgn3*sigma_fft[CTOP][C][S][idx3];
        sigfft[S][C][C][idx] = sigma_fft[WST][S][C][idx2] - sgn2*sigma_fft[EST][S][C][idx2]
                             + sigma_fft[CBOT][S][C][idx3] - sgn3*sigma_fft[CTOP][S][C][idx3];
        sigfft[C][S][S][idx] = sigma_fft[STH][S][S][idx1] - sgn1*sigma_fft[NTH][S][S][idx1];
        sigfft[S][C][S][idx] = sigma_fft[WST][S][S][idx2] - sgn2*sigma_fft[EST][S][S][idx2];
        sigfft[S][S][C][idx] = sigma_fft[CBOT][S][S][idx3] - sgn3*sigma_fft[CTOP][S][S][idx3];
      }
    }
  }

  // multiply Green's function in Fourier space
  for (int k=0;k<dcmps[PB].nx3;++k) {
    for (int j=0;j<dcmps[PB].nx2;++j) {
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
        sigfft[C][C][C][idx] *= grf[idx]; // this is psifft
        sigfft[C][C][S][idx] *= grf[idx]; // this is psifft
        sigfft[C][S][C][idx] *= grf[idx]; // this is psifft
        sigfft[S][C][C][idx] *= grf[idx]; // this is psifft    
        sigfft[C][S][S][idx] *= grf[idx]; // this is psifft
        sigfft[S][C][S][idx] *= grf[idx]; // this is psifft
        sigfft[S][S][C][idx] *= grf[idx]; // this is psifft
      }
    }
  }

  // Perform inverse transform to obtain 2D FFTs of STH/NTH boundary potential
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      for (int k=0;k<dcmps[PB].nx3;++k) {
        for (int j=0;j<dcmps[PB].nx2;++j) {
          int idx2 = j + dcmps[PB].nx2*k;
          sigma_fft[STH][p1][p2][idx2] = 0.0;
          sigma_fft[NTH][p1][p2][idx2] = 0.0;
          for (int i=0;i<dcmps[PB].nx1;++i) {
            int idx1 = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
            sgn1 = ((dcmps[PB].is+i)%2)*2 - 1;
            sigma_fft[STH][p1][p2][idx2] += sigfft[C][p1][p2][idx1];
            sigma_fft[NTH][p1][p2][idx2] -= sgn1*sigfft[C][p1][p2][idx1];
          }
        }
      }
    }
  }
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      if (x1rank == 0) 
//        MPI_Ireduce(MPI_IN_PLACE, sigma_fft[STH][p1][p2],
//         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         0, x1comm, &req[STH]);
        MPI_Reduce(MPI_IN_PLACE, sigma_fft[STH][p1][p2],
         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         0, x1comm);
      else
//        MPI_Ireduce(sigma_fft[STH][p1][p2], sigma_fft[STH][p1][p2],
//         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         0, x1comm, &req[STH]);
        MPI_Reduce(sigma_fft[STH][p1][p2], sigma_fft[STH][p1][p2],
         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         0, x1comm);
      if (x1rank == x1comm_size-1) 
//        MPI_Ireduce(MPI_IN_PLACE, sigma_fft[NTH][p1][p2],
//         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         x1comm_size-1, x1comm, &req[NTH]);
        MPI_Reduce(MPI_IN_PLACE, sigma_fft[NTH][p1][p2],
         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         x1comm_size-1, x1comm);
      else
//        MPI_Ireduce(sigma_fft[NTH][p1][p2], sigma_fft[NTH][p1][p2],
//         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         x1comm_size-1, x1comm, &req[NTH]);
        MPI_Reduce(sigma_fft[NTH][p1][p2], sigma_fft[NTH][p1][p2],
         dcmps[PB].nx2*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         x1comm_size-1, x1comm);
    }
  }
  // Perform inverse transform to obtain 2D FFTs of WST/EST boundary potential
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      for (int k=0;k<dcmps[PB].nx3;++k) {
        for (int i=0;i<dcmps[PB].nx1;++i) {
          int idx2 = i + dcmps[PB].nx1*k;
          sigma_fft[WST][p1][p2][idx2] = 0.0;
          sigma_fft[EST][p1][p2][idx2] = 0.0;
        }
      }
    }
  }
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      for (int k=0;k<dcmps[PB].nx3;++k) {
        for (int j=0;j<dcmps[PB].nx2;++j) {
          sgn2 = ((dcmps[PB].js+j)%2)*2 - 1;
          for (int i=0;i<dcmps[PB].nx1;++i) {
            int idx2 = i + dcmps[PB].nx1*k;
            int idx1 = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
            sigma_fft[WST][p1][p2][idx2] += sigfft[p1][C][p2][idx1];
            sigma_fft[EST][p1][p2][idx2] -= sgn2*sigfft[p1][C][p2][idx1];
          }
        }
      }
    }
  }
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      if (x2rank == 0) 
//        MPI_Ireduce(MPI_IN_PLACE, sigma_fft[WST][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         0, x2comm, &req[WST]);
        MPI_Reduce(MPI_IN_PLACE, sigma_fft[WST][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         0, x2comm);
      else
//        MPI_Ireduce(sigma_fft[WST][p1][p2], sigma_fft[WST][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         0, x2comm, &req[WST]);
        MPI_Reduce(sigma_fft[WST][p1][p2], sigma_fft[WST][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         0, x2comm);
      if (x2rank == x2comm_size-1) 
//        MPI_Ireduce(MPI_IN_PLACE, sigma_fft[EST][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         x2comm_size-1, x2comm, &req[EST]);
        MPI_Reduce(MPI_IN_PLACE, sigma_fft[EST][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         x2comm_size-1, x2comm);
      else
//        MPI_Ireduce(sigma_fft[EST][p1][p2], sigma_fft[EST][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
//         x2comm_size-1, x2comm, &req[EST]);
        MPI_Reduce(sigma_fft[EST][p1][p2], sigma_fft[EST][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx3, MPI_DOUBLE, MPI_SUM,
         x2comm_size-1, x2comm);
    }
  }

  // Perform inverse transform to obtain 2D FFTs of CBOT/CTOP boundary potential
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      for (int j=0;j<dcmps[PB].nx2;++j) {
        for (int i=0;i<dcmps[PB].nx1;++i) {
          int idx2 = i + dcmps[PB].nx1*j;
          sigma_fft[CBOT][p1][p2][idx2] = 0.0;
          sigma_fft[CTOP][p1][p2][idx2] = 0.0;
        }
      }
    }
  }
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      for (int k=0;k<dcmps[PB].nx3;++k) {
        sgn3 = ((dcmps[PB].ks+k)%2)*2 - 1;
        for (int j=0;j<dcmps[PB].nx2;++j) {
          for (int i=0;i<dcmps[PB].nx1;++i) {
            int idx2 = i + dcmps[PB].nx1*j;
            int idx1 = i + dcmps[PB].nx1*(j + dcmps[PB].nx2*k);
            sigma_fft[CBOT][p1][p2][idx2] += sigfft[p1][p2][C][idx1];
            sigma_fft[CTOP][p1][p2][idx2] -= sgn3*sigfft[p1][p2][C][idx1];
          }
        }
      }
    }
  }
  for (int p1=C;p1<=S;++p1) {
    for (int p2=C;p2<=S;++p2) {
      if (x3rank == 0) 
//        MPI_Ireduce(MPI_IN_PLACE, sigma_fft[CBOT][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
//         0, x3comm, &req[CBOT]);
        MPI_Reduce(MPI_IN_PLACE, sigma_fft[CBOT][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
         0, x3comm);
      else
//        MPI_Ireduce(sigma_fft[CBOT][p1][p2], sigma_fft[CBOT][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
//         0, x3comm, &req[CBOT]);
        MPI_Reduce(sigma_fft[CBOT][p1][p2], sigma_fft[CBOT][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
         0, x3comm);
      if (x3rank == x3comm_size-1) 
//        MPI_Ireduce(MPI_IN_PLACE, sigma_fft[CTOP][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
//         x3comm_size-1, x3comm, &req[CTOP]);
        MPI_Reduce(MPI_IN_PLACE, sigma_fft[CTOP][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
         x3comm_size-1, x3comm);
      else
//        MPI_Ireduce(sigma_fft[CTOP][p1][p2], sigma_fft[CTOP][p1][p2],
//         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
//         x3comm_size-1, x3comm, &req[CTOP]);
        MPI_Reduce(sigma_fft[CTOP][p1][p2], sigma_fft[CTOP][p1][p2],
         dcmps[PB].nx1*dcmps[PB].nx2, MPI_DOUBLE, MPI_SUM,
         x3comm_size-1, x3comm);
    }
  }
//  MPI_Wait(&req[STH], MPI_STATUS_IGNORE);
  BndFFTBackward(bndry_dcmps[STH][FFT_FIRST].nx3, bndry_dcmps[STH][FFT_FIRST].nx2, 
    bndry_dcmps[STH][FFT_SECOND].nx2, bndry_dcmps[STH][FFT_SECOND].nx3, STH);
//  MPI_Wait(&req[NTH], MPI_STATUS_IGNORE);
  BndFFTBackward(bndry_dcmps[NTH][FFT_FIRST].nx3, bndry_dcmps[NTH][FFT_FIRST].nx2,
    bndry_dcmps[NTH][FFT_SECOND].nx2, bndry_dcmps[NTH][FFT_SECOND].nx3, NTH);
//  MPI_Wait(&req[WST], MPI_STATUS_IGNORE);
  BndFFTBackward(bndry_dcmps[WST][FFT_FIRST].nx3, bndry_dcmps[WST][FFT_FIRST].nx1,
    bndry_dcmps[WST][FFT_SECOND].nx1, bndry_dcmps[WST][FFT_SECOND].nx3, WST);
//  MPI_Wait(&req[EST], MPI_STATUS_IGNORE);
  BndFFTBackward(bndry_dcmps[EST][FFT_FIRST].nx3, bndry_dcmps[EST][FFT_FIRST].nx1,
    bndry_dcmps[EST][FFT_SECOND].nx1, bndry_dcmps[EST][FFT_SECOND].nx3, EST);
//  MPI_Wait(&req[CBOT], MPI_STATUS_IGNORE);
  BndFFTBackward(bndry_dcmps[CBOT][FFT_FIRST].nx2, bndry_dcmps[CBOT][FFT_FIRST].nx1,
    bndry_dcmps[CBOT][FFT_SECOND].nx1, bndry_dcmps[CBOT][FFT_SECOND].nx2, CBOT);
//  MPI_Wait(&req[CTOP], MPI_STATUS_IGNORE);
  BndFFTBackward(bndry_dcmps[CTOP][FFT_FIRST].nx2, bndry_dcmps[CTOP][FFT_FIRST].nx1,
    bndry_dcmps[CTOP][FFT_SECOND].nx1, bndry_dcmps[CTOP][FFT_SECOND].nx2, CTOP);

  for (int b=STH;b<=NTH;++b) {
    for (int k=0;k<dcmps[PB].nx3;++k) {
      for (int j=0;j<dcmps[PB].nx2;++j) {
        int idx = j + dcmps[PB].nx2*k;
        sigma[b][idx] = 0;
      }
    }
    for (int p1=C;p1<=S;++p1) {
      for (int p2=C;p2<=S;++p2) {
        for (int k=0;k<dcmps[PB].nx3;++k) {
          for (int j=0;j<dcmps[PB].nx2;++j) {
            int idx = j + dcmps[PB].nx2*k;
            sigma[b][idx] += sigma_fft[b][p1][p2][idx];
          }
        }
      }
    }
  }
  for (int b=WST;b<=EST;++b) {
    for (int k=0;k<dcmps[PB].nx3;++k) {
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*k;
        sigma[b][idx] = 0;
      }
    }
    for (int p1=C;p1<=S;++p1) {
      for (int p2=C;p2<=S;++p2) {
        for (int k=0;k<dcmps[PB].nx3;++k) {
          for (int i=0;i<dcmps[PB].nx1;++i) {
            int idx = i + dcmps[PB].nx1*k;
            sigma[b][idx] += sigma_fft[b][p1][p2][idx];
          }
        }
      }
    }
  }
  for (int b=CBOT;b<=CTOP;++b) {
    for (int j=0;j<dcmps[PB].nx2;++j) {
      for (int i=0;i<dcmps[PB].nx1;++i) {
        int idx = i + dcmps[PB].nx1*j;
        sigma[b][idx] = 0;
      }
    }
    for (int p1=C;p1<=S;++p1) {
      for (int p2=C;p2<=S;++p2) {
        for (int j=0;j<dcmps[PB].nx2;++j) {
          for (int i=0;i<dcmps[PB].nx1;++i) {
            int idx = i + dcmps[PB].nx1*j;
            sigma[b][idx] += sigma_fft[b][p1][p2][idx];
          }
        }
      }
    }
  }

  Real normfac1 = 1.0 / (four_pi_G*dx1_*dx1_);
  Real normfac2 = 1.0 / (four_pi_G*dx2_*dx2_);
  Real normfac3 = 1.0 / (four_pi_G*dx3_*dx3_);

  if (dcmps[CXB].is == 0) {
    int jshift=0, kshift=0;
    if (bndry_dcmps[STH][CBLOCK].js==0) jshift=ngh_;
    if (bndry_dcmps[STH][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int j=0;j<dcmps[CXB].nx2;++j) {
        int idx2 = (j+jshift) + bndry_dcmps[STH][CBLOCK].nx2*(k+kshift);
        int idx = 0 + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*k);
        in_[idx] += sigma[STH][idx2]*normfac1;
      }
    }
  }
  if (dcmps[CXB].ie == Nx1-1) {
    int jshift=0, kshift=0;
    if (bndry_dcmps[NTH][CBLOCK].js==0) jshift=ngh_;
    if (bndry_dcmps[NTH][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int j=0;j<dcmps[CXB].nx2;++j) {
        int idx2 = (j+jshift) + bndry_dcmps[NTH][CBLOCK].nx2*(k+kshift);
        int idx = dcmps[CXB].nx1-1 + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*k);
        in_[idx] += sigma[NTH][idx2]*normfac1;
      }
    }
  }
  if (dcmps[CXB].js == 0) {
    int ishift=0, kshift=0;
    if (bndry_dcmps[WST][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[WST][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[WST][CBLOCK].nx1*(k+kshift);
        int idx = i + dcmps[CXB].nx1*(0 + dcmps[CXB].nx2*k);
        in_[idx] += sigma[WST][idx2]*normfac2;
      }
    }
  }
  if (dcmps[CXB].je == Nx2-1) {
    int ishift=0, kshift=0;
    if (bndry_dcmps[EST][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[EST][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<dcmps[CXB].nx3;++k) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[EST][CBLOCK].nx1*(k+kshift);
        int idx = i + dcmps[CXB].nx1*(dcmps[CXB].nx2-1 + dcmps[CXB].nx2*k);
        in_[idx] += sigma[EST][idx2]*normfac2;
      }
    }
  }
  if (dcmps[CXB].ks == 0) {
    int ishift=0, jshift=0;
    if (bndry_dcmps[CBOT][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[CBOT][CBLOCK].js==0) jshift=ngh_;
    for (int j=0;j<dcmps[CXB].nx2;++j) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[CBOT][CBLOCK].nx1*(j+jshift);
        int idx = i + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*0);
        in_[idx] += sigma[CBOT][idx2]*normfac3;
      }
    }
  }
  if (dcmps[CXB].ke == Nx3-1) {
    int ishift=0, jshift=0;
    if (bndry_dcmps[CTOP][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[CTOP][CBLOCK].js==0) jshift=ngh_;
    for (int j=0;j<dcmps[CXB].nx2;++j) {
      for (int i=0;i<dcmps[CXB].nx1;++i) {
        int idx2 = (i+ishift) + bndry_dcmps[CTOP][CBLOCK].nx1*(j+jshift);
        int idx = i + dcmps[CXB].nx1*(j + dcmps[CXB].nx2*(dcmps[CXB].nx3-1));
        in_[idx] += sigma[CTOP][idx2]*normfac3;
      }
    }
  }
}

void OBCGravityCar::RetrieveResult(AthenaArray<Real> &dst)
{
  int is, ie, js, je, ks, ke;
  int gis,gie,gjs,gje,gks,gke;
  int idx;
  is = pmy_block->is; js = pmy_block->js; ks = pmy_block->ks;
  ie = pmy_block->ie; je = pmy_block->je; ke = pmy_block->ke;
  gis = (pmy_block->loc.lx1)*nx1;
  gie = gis + nx1 - 1;
  gjs = (pmy_block->loc.lx2)*nx2;
  gje = gjs + nx2 - 1;
  gks = (pmy_block->loc.lx3)*nx3;
  gke = gks + nx3 - 1;

  for (int k=ks;k<=ke;++k) {
    for (int j=js;j<=je;++j) {
      for (int i=is;i<=ie;++i) {
        idx = (i-is) + nx1*((j-js) + nx2*(k-ks));
        dst(k,j,i) = out_[idx];
      }
    }
  }
  if (gis == 0) {
    int jshift=0, kshift=0;
    if (bndry_dcmps[STH][CBLOCK].js==0) jshift=ngh_;
    if (bndry_dcmps[STH][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[STH][CBLOCK].nx3;++k) {
      for (int j=0;j<bndry_dcmps[STH][CBLOCK].nx2;++j) {
        int idx2 = j + bndry_dcmps[STH][CBLOCK].nx2*k;
        dst(k+ks-kshift,j+js-jshift,is-1) = -sigma[STH][idx2];
      }
    }
  }
  if (gie == Nx1-1) {
    int jshift=0, kshift=0;
    if (bndry_dcmps[NTH][CBLOCK].js==0) jshift=ngh_;
    if (bndry_dcmps[NTH][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[NTH][CBLOCK].nx3;++k) {
      for (int j=0;j<bndry_dcmps[NTH][CBLOCK].nx2;++j) {
        int idx2 = j + bndry_dcmps[NTH][CBLOCK].nx2*k;
        dst(k+ks-kshift,j+js-jshift,ie+1) = -sigma[NTH][idx2];
      }
    }
  }
  if (gjs == 0) {
    int ishift=0, kshift=0;
    if (bndry_dcmps[WST][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[WST][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[WST][CBLOCK].nx3;++k) {
      for (int i=0;i<bndry_dcmps[WST][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[WST][CBLOCK].nx1*k;
        dst(k+ks-kshift,js-1,i+is-ishift) = -sigma[WST][idx2];
      }
    }
  }
  if (gje == Nx2-1) {
    int ishift=0, kshift=0;
    if (bndry_dcmps[EST][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[EST][CBLOCK].ks==0) kshift=ngh_;
    for (int k=0;k<bndry_dcmps[EST][CBLOCK].nx3;++k) {
      for (int i=0;i<bndry_dcmps[EST][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[EST][CBLOCK].nx1*k;
        dst(k+ks-kshift,je+1,i+is-ishift) = -sigma[EST][idx2];
      }
    }
  }
  if (gks == 0) {
    int ishift=0, jshift=0;
    if (bndry_dcmps[CBOT][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[CBOT][CBLOCK].js==0) jshift=ngh_;
    for (int j=0;j<bndry_dcmps[CBOT][CBLOCK].nx2;++j) {
      for (int i=0;i<bndry_dcmps[CBOT][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[CBOT][CBLOCK].nx1*j;
        dst(ks-1,j+js-jshift,i+is-ishift) = -sigma[CBOT][idx2];
      }
    }
  }
  if (gke == Nx3-1) {
    int ishift=0, jshift=0;
    if (bndry_dcmps[CTOP][CBLOCK].is==0) ishift=ngh_;
    if (bndry_dcmps[CTOP][CBLOCK].js==0) jshift=ngh_;
    for (int j=0;j<bndry_dcmps[CTOP][CBLOCK].nx2;++j) {
      for (int i=0;i<bndry_dcmps[CTOP][CBLOCK].nx1;++i) {
        int idx2 = i + bndry_dcmps[CTOP][CBLOCK].nx1*j;
        dst(ke+1,j+js-jshift,i+is-ishift) = -sigma[CTOP][idx2];
      }
    }
  }
}

void tridag(AthenaArray<Real> &a, AthenaArray<Real> &b, AthenaArray<Real> &c,
            AthenaArray<Real> &x, AthenaArray<Real> &r)
{
  int j,N;
  Real bet;
  AthenaArray<Real> gam;

  N = r.GetDim1();
  gam.NewAthenaArray(N);
  if (b(0) == 0.0) throw("Error 1 in tridag");
  x(0) = r(0) / (bet=b(0));
  for (j=1;j<N;j++) {
    gam(j) = c(j-1) / bet;
    bet = b(j) - a(j)*gam(j);
    if (b(0) == 0.0) throw("Error 2 in tridag");
    x(j) = (r(j) - a(j)*x(j-1)) / bet;
  }
  for (j=(N-2);j>=0;j--) {
    x(j) -= gam(j+1)*x(j+1);
  }
  gam.DeleteAthenaArray();
  return;
}

// OBCGravityCyl constructor
OBCGravityCyl::OBCGravityCyl(OBCGravityDriver *pod, MeshBlock *pmb, ParameterInput *pin)
 : Gravity(pmb, pin)
{
  int arrsize;
  pmy_driver_ = pod;
  Coordinates *pcrd = pmy_block->pcoord;
  RegionSize& mesh_size = pmy_block->pmy_mesh->mesh_size;
  RegionSize& block_size = pmy_block->block_size;
  Mesh *pm = pmy_block->pmy_mesh;
  Nx1 = mesh_size.nx1;
  Nx2 = mesh_size.nx2;
  Nx3 = mesh_size.nx3;
  nx1 = block_size.nx1;
  nx2 = block_size.nx2;
  nx3 = block_size.nx3;
  lNx1 = 2*Nx1;
  hNx2 = Nx2/2;
  hnx2 = nx2/2;
  np1 = Nx1/nx1;
  np2 = Nx2/nx2;
  np3 = Nx3/nx3;
  rat = mesh_size.x1rat;
  ng_ = std::min(16, Nx1/4); // TODO for uniform grid?
  ngh_grf_ = 16;
  noffset1_ = lNx1-Nx1-ng_;
  noffset2_ = ngh_grf_;
  if (fabs(remainder(2*PI,mesh_size.x2max-mesh_size.x2min)) > 1e-15) {
    std::stringstream msg;
    if (Globals::my_rank==0) msg << "### FATAL ERROR in OBCGravityCyl" << std::endl
         << "The azimuthal domain length should be integer fraction of 2 pi."
         << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  else {
    pfold_ = std::round((2*PI)/(mesh_size.x2max-mesh_size.x2min));
    if (Globals::my_rank==0) std::cout << "pfold = " << pfold_ << std::endl;
  }

  // create plan for remap and FFT
  int np2d1,np2d2;
  bifactor(Globals::nranks, &np2d2, &np2d1);
  int ip1 = Globals::my_rank % np2d1; // ip1 is fast index
  int ip2 = Globals::my_rank / np2d1; // ip2 is slow index

  for (int i=XB;i<=Gkk_BLOCK;++i) {
    dcmps[i].is = -1;
    dcmps[i].ie = -1;
    dcmps[i].js = -1;
    dcmps[i].je = -1;
    dcmps[i].ks = -1;
    dcmps[i].ke = -1;
    dcmps[i].nx1 = -1;
    dcmps[i].nx2 = -1;
    dcmps[i].nx3 = -1;
    dcmps[i].block_size = -1;
  }

  dcmps[XB].is = (pmy_block->loc.lx1)*nx1;
  dcmps[XB].ie = dcmps[XB].is + nx1 - 1;
  dcmps[XB].js = (pmy_block->loc.lx2)*nx2;
  dcmps[XB].je = dcmps[XB].js + nx2 - 1;
  dcmps[XB].ks = (pmy_block->loc.lx3)*nx3;
  dcmps[XB].ke = dcmps[XB].ks + nx3 - 1;

  dcmps[X1P].is = 0;
  dcmps[X1P].ie = Nx1 - 1;
  dcmps[X1P].js = ip2*hNx2/np2d2;
  dcmps[X1P].je = (ip2+1)*hNx2/np2d2 - 1;
  dcmps[X1P].ks = ip1*Nx3/np2d1;
  dcmps[X1P].ke = (ip1+1)*Nx3/np2d1 - 1;
  if (ip2 == np2d2-1) dcmps[X1P].je++;

  dcmps[X2P].is = ip1*Nx1/np2d1;
  dcmps[X2P].ie = (ip1+1)*Nx1/np2d1 - 1;
  dcmps[X2P].js = 0;
  dcmps[X2P].je = hNx2;
  dcmps[X2P].ks = ip2*Nx3/np2d2;
  dcmps[X2P].ke = (ip2+1)*Nx3/np2d2 - 1;

  dcmps[X3P].is = ip1*Nx1/np2d1;
  dcmps[X3P].ie = (ip1+1)*Nx1/np2d1 - 1;
  dcmps[X3P].js = ip2*hNx2/np2d2;
  dcmps[X3P].je = (ip2+1)*hNx2/np2d2 - 1;
  dcmps[X3P].ks = 0;
  dcmps[X3P].ke = Nx3 - 1;
  if (ip2 == np2d2-1) dcmps[X3P].je++;

  dcmps[X2P0].is = ip1*Nx1/np2d1;
  dcmps[X2P0].ie = (ip1+1)*Nx1/np2d1 - 1;
  dcmps[X2P0].js = 0;
  dcmps[X2P0].je = Nx2 - 1;
  dcmps[X2P0].ks = ip2*Nx3/np2d2;
  dcmps[X2P0].ke = (ip2+1)*Nx3/np2d2 - 1;

  dcmps[EB].is = (pmy_block->loc.lx1)*lNx1/np1;
  dcmps[EB].ie = dcmps[EB].is + lNx1/np1 - 1;
  dcmps[EB].js = (pmy_block->loc.lx2)*nx2;
  dcmps[EB].je = dcmps[EB].js + nx2 - 1;
  dcmps[EB].ks = (pmy_block->loc.lx3)*nx3 + ngh_grf_;
  dcmps[EB].ke = dcmps[EB].ks + nx3 - 1;
  if (pmy_block->loc.lx3==0) dcmps[EB].ks -= ngh_grf_;
  if (pmy_block->loc.lx3==np3-1) dcmps[EB].ke += ngh_grf_;

  dcmps[E1P].is = 0;
  dcmps[E1P].ie = lNx1 - 1;
  dcmps[E1P].js = ip2*hNx2/np2d2;
  dcmps[E1P].je = (ip2+1)*hNx2/np2d2 - 1;
  dcmps[E1P].ks = ip1*Nx3/np2d1 + ngh_grf_;
  dcmps[E1P].ke = (ip1+1)*Nx3/np2d1 - 1 + ngh_grf_;
  if (ip2 == np2d2-1) dcmps[E1P].je++;
  if (ip1==0) dcmps[E1P].ks -= ngh_grf_;
  if (ip1==np2d1-1) dcmps[E1P].ke += ngh_grf_;

  dcmps[E2P].is = ip1*(lNx1)/np2d1;
  dcmps[E2P].ie = (ip1+1)*(lNx1)/np2d1 - 1;
  dcmps[E2P].js = 0;
  dcmps[E2P].je = hNx2;
  dcmps[E2P].ks = ip2*Nx3/np2d2 + ngh_grf_;
  dcmps[E2P].ke = (ip2+1)*Nx3/np2d2 - 1 + ngh_grf_;
  if (ip2==0) dcmps[E2P].ks -= ngh_grf_;
  if (ip2==np2d2-1) dcmps[E2P].ke += ngh_grf_;

  dcmps[E3P].is = ip1*(lNx1)/np2d1;
  dcmps[E3P].ie = (ip1+1)*(lNx1)/np2d1 - 1;
  dcmps[E3P].js = ip2*hNx2/np2d2;
  dcmps[E3P].je = (ip2+1)*hNx2/np2d2 - 1;
  dcmps[E3P].ks = 0;
  dcmps[E3P].ke = Nx3 - 1 + 2*ngh_grf_;
  if (ip2 == np2d2-1) dcmps[E3P].je++;

  dcmps[E2P0].is = ip1*(lNx1)/np2d1;
  dcmps[E2P0].ie = (ip1+1)*(lNx1)/np2d1 - 1;
  dcmps[E2P0].js = 0;
  dcmps[E2P0].je = Nx2 - 1;
  dcmps[E2P0].ks = ip2*Nx3/np2d2 + ngh_grf_;
  dcmps[E2P0].ke = (ip2+1)*Nx3/np2d2 - 1 + ngh_grf_;
  if (ip2==0) dcmps[E2P0].ks -= ngh_grf_;
  if (ip2==np2d2-1) dcmps[E2P0].ke += ngh_grf_;

  // third dimension corresponds to the primed index for the Green's functions
  dcmps[Gii].is = dcmps[XB].is;
  dcmps[Gii].ie = dcmps[XB].ie;
  dcmps[Gii].js = (pmy_block->loc.lx2)*hnx2;
  dcmps[Gii].je = dcmps[Gii].js + hnx2 - 1;
  dcmps[Gii].ks = (pmy_block->loc.lx3)*(Nx1/np3);
  dcmps[Gii].ke = dcmps[Gii].ks + (Nx1/np3) - 1;
  if (dcmps[Gii].je == hNx2-1) dcmps[Gii].je++;

  dcmps[Gik].is = dcmps[XB].is;
  dcmps[Gik].ie = dcmps[XB].ie;
  dcmps[Gik].js = (pmy_block->loc.lx2)*hnx2;
  dcmps[Gik].je = dcmps[Gik].js + hnx2 - 1;
  dcmps[Gik].ks = dcmps[XB].ks;
  dcmps[Gik].ke = dcmps[XB].ke;
  if (dcmps[Gik].je == hNx2-1) dcmps[Gik].je++;

  dcmps[Gki].is = dcmps[XB].ks;
  dcmps[Gki].ie = dcmps[XB].ke;
  dcmps[Gki].js = (pmy_block->loc.lx2)*hnx2;
  dcmps[Gki].je = dcmps[Gki].js + hnx2 - 1;
  dcmps[Gki].ks = dcmps[XB].is;
  dcmps[Gki].ke = dcmps[XB].ie;
  if (dcmps[Gki].je == hNx2-1) dcmps[Gki].je++;

  dcmps[Gkk].is = dcmps[XB].ks;
  dcmps[Gkk].ie = dcmps[XB].ke;
  dcmps[Gkk].js = (pmy_block->loc.lx2)*hnx2;
  dcmps[Gkk].je = dcmps[Gkk].js + hnx2 - 1;
  dcmps[Gkk].ks = (pmy_block->loc.lx1)*(Nx3/np1);
  dcmps[Gkk].ke = dcmps[Gkk].ks + (Nx3/np1) - 1;
  if (dcmps[Gkk].je == hNx2-1) dcmps[Gkk].je++;

  dcmps[Gii_BLOCK].is = (pmy_block->loc.lx1)*nx1;
  dcmps[Gii_BLOCK].ie = dcmps[Gii_BLOCK].is + nx1 - 1;
  dcmps[Gii_BLOCK].js = (pmy_block->loc.lx2)*nx2;
  dcmps[Gii_BLOCK].je = dcmps[Gii_BLOCK].js + nx2 - 1;
  dcmps[Gii_BLOCK].ks = (pmy_block->loc.lx3)*Nx1/np3;
  dcmps[Gii_BLOCK].ke = dcmps[Gii_BLOCK].ks + Nx1/np3 - 1;

  dcmps[Gik_BLOCK].is = (pmy_block->loc.lx1)*nx1;
  dcmps[Gik_BLOCK].ie = dcmps[Gik_BLOCK].is + nx1 - 1;
  dcmps[Gik_BLOCK].js = (pmy_block->loc.lx2)*nx2;
  dcmps[Gik_BLOCK].je = dcmps[Gik_BLOCK].js + nx2 - 1;
  dcmps[Gik_BLOCK].ks = (pmy_block->loc.lx3)*nx3;
  dcmps[Gik_BLOCK].ke = dcmps[Gik_BLOCK].ks + nx3 - 1;

  dcmps[Gki_BLOCK].is = (pmy_block->loc.lx1)*Nx3/np1;
  dcmps[Gki_BLOCK].ie = dcmps[Gki_BLOCK].is + Nx3/np1 - 1;
  dcmps[Gki_BLOCK].js = (pmy_block->loc.lx2)*nx2;
  dcmps[Gki_BLOCK].je = dcmps[Gki_BLOCK].js + nx2 - 1;
  dcmps[Gki_BLOCK].ks = (pmy_block->loc.lx3)*Nx1/np3;
  dcmps[Gki_BLOCK].ke = dcmps[Gki_BLOCK].ks + Nx1/np3 - 1;

  dcmps[Gkk_BLOCK].is = (pmy_block->loc.lx1)*Nx3/np1;
  dcmps[Gkk_BLOCK].ie = dcmps[Gkk_BLOCK].is + Nx3/np1 - 1;
  dcmps[Gkk_BLOCK].js = (pmy_block->loc.lx2)*nx2;
  dcmps[Gkk_BLOCK].je = dcmps[Gkk_BLOCK].js + nx2 - 1;
  dcmps[Gkk_BLOCK].ks = (pmy_block->loc.lx3)*nx3;
  dcmps[Gkk_BLOCK].ke = dcmps[Gkk_BLOCK].ks + nx3 - 1;

  dcmps[Gii2P0].is = ip1*Nx1/np2d1;;
  dcmps[Gii2P0].ie = (ip1+1)*(Nx1)/np2d1 - 1;
  dcmps[Gii2P0].js = 0;
  dcmps[Gii2P0].je = Nx2 - 1;
  dcmps[Gii2P0].ks = ip2*Nx1/np2d2;
  dcmps[Gii2P0].ke = (ip2+1)*Nx1/np2d2 - 1;

  dcmps[Gik2P0].is = ip1*Nx1/np2d1;;
  dcmps[Gik2P0].ie = (ip1+1)*(Nx1)/np2d1 - 1;
  dcmps[Gik2P0].js = 0;
  dcmps[Gik2P0].je = Nx2 - 1;
  dcmps[Gik2P0].ks = ip2*Nx3/np2d2;
  dcmps[Gik2P0].ke = (ip2+1)*Nx3/np2d2 - 1;

  dcmps[Gki2P0].is = ip1*Nx3/np2d1;;
  dcmps[Gki2P0].ie = (ip1+1)*(Nx3)/np2d1 - 1;
  dcmps[Gki2P0].js = 0;
  dcmps[Gki2P0].je = Nx2 - 1;
  dcmps[Gki2P0].ks = ip2*Nx1/np2d2;
  dcmps[Gki2P0].ke = (ip2+1)*Nx1/np2d2 - 1;

  dcmps[Gkk2P0].is = ip1*Nx3/np2d1;;
  dcmps[Gkk2P0].ie = (ip1+1)*(Nx3)/np2d1 - 1;
  dcmps[Gkk2P0].js = 0;
  dcmps[Gkk2P0].je = Nx2 - 1;
  dcmps[Gkk2P0].ks = ip2*Nx3/np2d2;
  dcmps[Gkk2P0].ke = (ip2+1)*Nx3/np2d2 - 1;

  dcmps[Gii2P].is = ip1*Nx1/np2d1;;
  dcmps[Gii2P].ie = (ip1+1)*(Nx1)/np2d1 - 1;
  dcmps[Gii2P].js = 0;
  dcmps[Gii2P].je = hNx2;
  dcmps[Gii2P].ks = ip2*Nx1/np2d2;
  dcmps[Gii2P].ke = (ip2+1)*Nx1/np2d2 - 1;

  dcmps[Gik2P].is = ip1*Nx1/np2d1;;
  dcmps[Gik2P].ie = (ip1+1)*(Nx1)/np2d1 - 1;
  dcmps[Gik2P].js = 0;
  dcmps[Gik2P].je = hNx2;
  dcmps[Gik2P].ks = ip2*Nx3/np2d2;
  dcmps[Gik2P].ke = (ip2+1)*Nx3/np2d2 - 1;

  dcmps[Gki2P].is = ip1*Nx3/np2d1;;
  dcmps[Gki2P].ie = (ip1+1)*(Nx3)/np2d1 - 1;
  dcmps[Gki2P].js = 0;
  dcmps[Gki2P].je = hNx2;
  dcmps[Gki2P].ks = ip2*Nx1/np2d2;
  dcmps[Gki2P].ke = (ip2+1)*Nx1/np2d2 - 1;

  dcmps[Gkk2P].is = ip1*Nx3/np2d1;;
  dcmps[Gkk2P].ie = (ip1+1)*(Nx3)/np2d1 - 1;
  dcmps[Gkk2P].js = 0;
  dcmps[Gkk2P].je = hNx2;
  dcmps[Gkk2P].ks = ip2*Nx3/np2d2;
  dcmps[Gkk2P].ke = (ip2+1)*Nx3/np2d2 - 1;

  for (int i=XB;i<=Gkk_BLOCK;++i) {
    dcmps[i].nx1 = dcmps[i].ie - dcmps[i].is + 1;
    dcmps[i].nx2 = dcmps[i].je - dcmps[i].js + 1;
    dcmps[i].nx3 = dcmps[i].ke - dcmps[i].ks + 1;
    dcmps[i].block_size = dcmps[i].nx1*dcmps[i].nx2*dcmps[i].nx3;
  }

  int color;
  color = (dcmps[XB].ke == Nx3-1) ? dcmps[XB].ke : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[TOP]);
  color = (dcmps[XB].ks == 0) ? dcmps[XB].ks : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[BOT]);
  color = (dcmps[XB].is == 0) ? dcmps[XB].is : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[INN]);
  color = (dcmps[XB].ie == Nx1-1) ? dcmps[XB].ie : MPI_UNDEFINED;
  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &bndcomm[OUT]);
  MPI_Comm_split(MPI_COMM_WORLD, dcmps[XB].js+Nx2*dcmps[XB].ks, dcmps[XB].is, &x1comm);
  MPI_Comm_split(MPI_COMM_WORLD, dcmps[XB].js+Nx2*dcmps[XB].is, dcmps[XB].ks, &x3comm);
  MPI_Comm_size(x1comm, &x1comm_size);
  MPI_Comm_size(x3comm, &x3comm_size);
  MPI_Comm_rank(x1comm, &x1rank);
  MPI_Comm_rank(x3comm, &x3rank);
  for (int i=TOP;i<=OUT;++i ) {
    for (int j=BLOCK;j<=SIGr;++j) {
      bndry_dcmps[i][j].is = 0;
      bndry_dcmps[i][j].ie = -1;
      bndry_dcmps[i][j].js = 0;
      bndry_dcmps[i][j].je = -1;
      bndry_dcmps[i][j].ks = 0;
      bndry_dcmps[i][j].ke = -1;
    }
  }
  int bndrank, nx;
  if (dcmps[XB].ke==Nx3-1) { // top boundary
    bndry_dcmps[TOP][BLOCK].is = dcmps[XB].is;
    bndry_dcmps[TOP][BLOCK].ie = dcmps[XB].ie;
    bndry_dcmps[TOP][BLOCK].js = dcmps[XB].js;
    bndry_dcmps[TOP][BLOCK].je = dcmps[XB].je;
    bndry_dcmps[TOP][BLOCK].ks = Nx3;
    bndry_dcmps[TOP][BLOCK].ke = Nx3;
    MPI_Comm_rank(bndcomm[TOP], &bndrank);
    nx = Nx1/np1/np2;
    bndry_dcmps[TOP][FFT_LONG].is = bndrank * nx;
    bndry_dcmps[TOP][FFT_LONG].ie = (bndrank+1) * nx - 1;
    bndry_dcmps[TOP][FFT_LONG].js = 0;
    bndry_dcmps[TOP][FFT_LONG].je = Nx2-1;
    bndry_dcmps[TOP][FFT_LONG].ks = Nx3;
    bndry_dcmps[TOP][FFT_LONG].ke = Nx3;
    bndry_dcmps[TOP][FFT_SHORT].is = bndrank * nx;
    bndry_dcmps[TOP][FFT_SHORT].ie = (bndrank+1) * nx - 1;
    bndry_dcmps[TOP][FFT_SHORT].js = 0;
    bndry_dcmps[TOP][FFT_SHORT].je = hNx2;
    bndry_dcmps[TOP][FFT_SHORT].ks = Nx3;
    bndry_dcmps[TOP][FFT_SHORT].ke = Nx3;
    bndry_dcmps[TOP][PSI].is = dcmps[Gii].is;
    bndry_dcmps[TOP][PSI].ie = dcmps[Gii].ie;
    bndry_dcmps[TOP][PSI].js = dcmps[Gii].js;
    bndry_dcmps[TOP][PSI].je = dcmps[Gii].je;
    bndry_dcmps[TOP][PSI].ks = Nx3;
    bndry_dcmps[TOP][PSI].ke = Nx3;
  }
  if (dcmps[XB].ks==0) { // bottom boundary
    bndry_dcmps[BOT][BLOCK].is = dcmps[XB].is;
    bndry_dcmps[BOT][BLOCK].ie = dcmps[XB].ie;
    bndry_dcmps[BOT][BLOCK].js = dcmps[XB].js;
    bndry_dcmps[BOT][BLOCK].je = dcmps[XB].je;
    bndry_dcmps[BOT][BLOCK].ks = -1;
    bndry_dcmps[BOT][BLOCK].ke = -1;
    MPI_Comm_rank(bndcomm[BOT], &bndrank);
    nx = Nx1/np1/np2;
    bndry_dcmps[BOT][FFT_LONG].is = bndrank * nx;
    bndry_dcmps[BOT][FFT_LONG].ie = (bndrank+1) * nx - 1;
    bndry_dcmps[BOT][FFT_LONG].js = 0;
    bndry_dcmps[BOT][FFT_LONG].je = Nx2-1;
    bndry_dcmps[BOT][FFT_LONG].ks = -1;
    bndry_dcmps[BOT][FFT_LONG].ke = -1;
    bndry_dcmps[BOT][FFT_SHORT].is = bndrank * nx;
    bndry_dcmps[BOT][FFT_SHORT].ie = (bndrank+1) * nx - 1;
    bndry_dcmps[BOT][FFT_SHORT].js = 0;
    bndry_dcmps[BOT][FFT_SHORT].je = hNx2;
    bndry_dcmps[BOT][FFT_SHORT].ks = -1;
    bndry_dcmps[BOT][FFT_SHORT].ke = -1;
    bndry_dcmps[BOT][PSI].is = dcmps[Gii].is;
    bndry_dcmps[BOT][PSI].ie = dcmps[Gii].ie;
    bndry_dcmps[BOT][PSI].js = dcmps[Gii].js;
    bndry_dcmps[BOT][PSI].je = dcmps[Gii].je;
    bndry_dcmps[BOT][PSI].ks = -1;
    bndry_dcmps[BOT][PSI].ke = -1;
  }
  if (dcmps[XB].is==0) { // inner boundary
    bndry_dcmps[INN][BLOCK].is = -1;
    bndry_dcmps[INN][BLOCK].ie = -1;
    bndry_dcmps[INN][BLOCK].js = dcmps[XB].js;
    bndry_dcmps[INN][BLOCK].je = dcmps[XB].je;
    bndry_dcmps[INN][BLOCK].ks = dcmps[XB].ks;
    bndry_dcmps[INN][BLOCK].ke = dcmps[XB].ke;
    MPI_Comm_rank(bndcomm[INN], &bndrank);
    nx = Nx3/np2/np3;
    bndry_dcmps[INN][FFT_LONG].is = -1;
    bndry_dcmps[INN][FFT_LONG].ie = -1;
    bndry_dcmps[INN][FFT_LONG].js = 0;
    bndry_dcmps[INN][FFT_LONG].je = Nx2-1;
    bndry_dcmps[INN][FFT_LONG].ks = bndrank * nx;
    bndry_dcmps[INN][FFT_LONG].ke = (bndrank+1) * nx - 1;
    bndry_dcmps[INN][FFT_SHORT].is = -1;
    bndry_dcmps[INN][FFT_SHORT].ie = -1;
    bndry_dcmps[INN][FFT_SHORT].js = 0;
    bndry_dcmps[INN][FFT_SHORT].je = hNx2;
    bndry_dcmps[INN][FFT_SHORT].ks = bndrank * nx;
    bndry_dcmps[INN][FFT_SHORT].ke = (bndrank+1) * nx - 1;
    bndry_dcmps[INN][PSI].is = -1;
    bndry_dcmps[INN][PSI].ie = -1;
    bndry_dcmps[INN][PSI].js = dcmps[Gki].js;
    bndry_dcmps[INN][PSI].je = dcmps[Gki].je;
    bndry_dcmps[INN][PSI].ks = dcmps[Gki].is;
    bndry_dcmps[INN][PSI].ke = dcmps[Gki].ie;
  }
  if (dcmps[XB].ie==Nx1-1) { // outer boundary
    bndry_dcmps[OUT][BLOCK].is = Nx1;
    bndry_dcmps[OUT][BLOCK].ie = Nx1;
    bndry_dcmps[OUT][BLOCK].js = dcmps[XB].js;
    bndry_dcmps[OUT][BLOCK].je = dcmps[XB].je;
    bndry_dcmps[OUT][BLOCK].ks = dcmps[XB].ks;
    bndry_dcmps[OUT][BLOCK].ke = dcmps[XB].ke;
    MPI_Comm_rank(bndcomm[OUT], &bndrank);
    nx = Nx3/np2/np3;
    bndry_dcmps[OUT][FFT_LONG].is = Nx1;
    bndry_dcmps[OUT][FFT_LONG].ie = Nx1;
    bndry_dcmps[OUT][FFT_LONG].js = 0;
    bndry_dcmps[OUT][FFT_LONG].je = Nx2-1;
    bndry_dcmps[OUT][FFT_LONG].ks = bndrank * nx;
    bndry_dcmps[OUT][FFT_LONG].ke = (bndrank+1) * nx - 1;
    bndry_dcmps[OUT][FFT_SHORT].is = Nx1;
    bndry_dcmps[OUT][FFT_SHORT].ie = Nx1;
    bndry_dcmps[OUT][FFT_SHORT].js = 0;
    bndry_dcmps[OUT][FFT_SHORT].je = hNx2;
    bndry_dcmps[OUT][FFT_SHORT].ks = bndrank * nx;
    bndry_dcmps[OUT][FFT_SHORT].ke = (bndrank+1) * nx - 1;
    bndry_dcmps[OUT][PSI].is = Nx1;
    bndry_dcmps[OUT][PSI].ie = Nx1;
    bndry_dcmps[OUT][PSI].js = dcmps[Gki].js;
    bndry_dcmps[OUT][PSI].je = dcmps[Gki].je;
    bndry_dcmps[OUT][PSI].ks = dcmps[Gki].is;
    bndry_dcmps[OUT][PSI].ke = dcmps[Gki].ie;
  }
  if (dcmps[XB].is==0) {
    bndry_dcmps[TOP][SIGv].is = dcmps[Gii].ks;
    bndry_dcmps[TOP][SIGv].ie = dcmps[Gii].ke;
    bndry_dcmps[TOP][SIGv].js = dcmps[Gii].js;
    bndry_dcmps[TOP][SIGv].je = dcmps[Gii].je;
    bndry_dcmps[TOP][SIGv].ks = Nx3;
    bndry_dcmps[TOP][SIGv].ke = Nx3;
    bndry_dcmps[BOT][SIGv].is = dcmps[Gii].ks;
    bndry_dcmps[BOT][SIGv].ie = dcmps[Gii].ke;
    bndry_dcmps[BOT][SIGv].js = dcmps[Gii].js;
    bndry_dcmps[BOT][SIGv].je = dcmps[Gii].je;
    bndry_dcmps[BOT][SIGv].ks = -1;
    bndry_dcmps[BOT][SIGv].ke = -1;
    bndry_dcmps[INN][SIGv].is = -1;
    bndry_dcmps[INN][SIGv].ie = -1;
    bndry_dcmps[INN][SIGv].js = dcmps[Gik].js;
    bndry_dcmps[INN][SIGv].je = dcmps[Gik].je;
    bndry_dcmps[INN][SIGv].ks = dcmps[Gik].ks;
    bndry_dcmps[INN][SIGv].ke = dcmps[Gik].ke;
  }
  if (dcmps[XB].ie==Nx1-1) {
    bndry_dcmps[OUT][SIGv].is = Nx1;
    bndry_dcmps[OUT][SIGv].ie = Nx1;
    bndry_dcmps[OUT][SIGv].js = dcmps[Gik].js;
    bndry_dcmps[OUT][SIGv].je = dcmps[Gik].je;
    bndry_dcmps[OUT][SIGv].ks = dcmps[Gik].ks;
    bndry_dcmps[OUT][SIGv].ke = dcmps[Gik].ke;
  }
  if (dcmps[XB].ks==0) {
    bndry_dcmps[BOT][SIGr].is = dcmps[Gki].ks;
    bndry_dcmps[BOT][SIGr].ie = dcmps[Gki].ke;
    bndry_dcmps[BOT][SIGr].js = dcmps[Gki].js;
    bndry_dcmps[BOT][SIGr].je = dcmps[Gki].je;
    bndry_dcmps[BOT][SIGr].ks = -1;
    bndry_dcmps[BOT][SIGr].ke = -1;
  }
  if (dcmps[XB].ke==Nx3-1) {
    bndry_dcmps[INN][SIGr].is = -1;
    bndry_dcmps[INN][SIGr].ie = -1;
    bndry_dcmps[INN][SIGr].js = dcmps[Gkk].js;
    bndry_dcmps[INN][SIGr].je = dcmps[Gkk].je;
    bndry_dcmps[INN][SIGr].ks = dcmps[Gkk].ks;
    bndry_dcmps[INN][SIGr].ke = dcmps[Gkk].ke;
    bndry_dcmps[OUT][SIGr].is = Nx1;
    bndry_dcmps[OUT][SIGr].ie = Nx1;
    bndry_dcmps[OUT][SIGr].js = dcmps[Gkk].js;
    bndry_dcmps[OUT][SIGr].je = dcmps[Gkk].je;
    bndry_dcmps[OUT][SIGr].ks = dcmps[Gkk].ks;
    bndry_dcmps[OUT][SIGr].ke = dcmps[Gkk].ke;
    bndry_dcmps[TOP][SIGr].is = dcmps[Gki].ks;
    bndry_dcmps[TOP][SIGr].ie = dcmps[Gki].ke;
    bndry_dcmps[TOP][SIGr].js = dcmps[Gki].js;
    bndry_dcmps[TOP][SIGr].je = dcmps[Gki].je;
    bndry_dcmps[TOP][SIGr].ks = Nx3;
    bndry_dcmps[TOP][SIGr].ke = Nx3;
  }

  for (int i=TOP;i<=OUT;++i) {
    for (int j=BLOCK;j<=SIGr;++j) {
      bndry_dcmps[i][j].nx1 = bndry_dcmps[i][j].ie - bndry_dcmps[i][j].is + 1;
      bndry_dcmps[i][j].nx2 = bndry_dcmps[i][j].je - bndry_dcmps[i][j].js + 1;
      bndry_dcmps[i][j].nx3 = bndry_dcmps[i][j].ke - bndry_dcmps[i][j].ks + 1;
      bndry_dcmps[i][j].block_size = 
        bndry_dcmps[i][j].nx1*bndry_dcmps[i][j].nx2*bndry_dcmps[i][j].nx3;
    }
  }

  if (dcmps[X3P].nx3 % 2 != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CYLGRAV" << std::endl
         << "number of vertical grid cell must be even" << std::endl; 
    throw std::runtime_error(msg.str().c_str());
  }
  if ((hNx2 % np2d2 != 0)||(Nx3 % np2d2 != 0)||(Nx1 % np2d2 != 0)
      ||(Nx1 % np2d1 != 0)||(Nx3 % np2d1 != 0)||(Nx3 % np1 != 0)
      ||(Nx1 % np3 != 0)||(Nx2 % 2 != 0)||(nx2 % 2 != 0)||(Nx1 % 2 != 0)
      ||(nx1 % np2 != 0)||(nx3 % np2 != 0)) { 
    // nx1 % np2 is for the bndry_dcmps, which requires Nx1/np1/np2
    // nx3 % np2 is for the bndry_dcmps, which requires Nx3/np2/np3
    std::stringstream msg;
    msg << "### FATAL ERROR in OBCGravityCyl" << std::endl
         << "domain decomposition failed in obcgrav" << std::endl
         << "np2d1 = " << np2d1 << " np2d2 = " << np2d2 << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }


  // allocate arrays
  int ncells = dcmps[X1P].nx1 + 2*NGHOST;
  int bufsize = dcmps[E2P0].block_size;
  bufsize = MAX(bufsize, dcmps[Gii_BLOCK].block_size);
  bufsize = MAX(bufsize, dcmps[Gik_BLOCK].block_size);
  bufsize = MAX(bufsize, dcmps[Gki_BLOCK].block_size);
  bufsize = MAX(bufsize, dcmps[Gkk_BLOCK].block_size);
  a_.NewAthenaArray(dcmps[X1P].nx1);
  b_.NewAthenaArray(dcmps[X1P].nx1);
  c_.NewAthenaArray(dcmps[X1P].nx1);
  x_.NewAthenaArray(dcmps[X1P].nx1);
  r_.NewAthenaArray(dcmps[X1P].nx1);
  lambda2_.NewAthenaArray(dcmps[X1P].nx2, dcmps[X1P].nx1);
  lambda3_.NewAthenaArray(dcmps[X1P].nx3);
  dx1f_.NewAthenaArray(ncells);
  x1f_.NewAthenaArray(ncells+1);
  dx1v_.NewAthenaArray(ncells-1);
  x1v_.NewAthenaArray(ncells);
  int ncells2 = dcmps[E1P].nx1 + 2*NGHOST;
  aa_.NewAthenaArray(dcmps[E1P].nx1);
  bb_.NewAthenaArray(dcmps[E1P].nx1);
  cc_.NewAthenaArray(dcmps[E1P].nx1);
  xx_.NewAthenaArray(dcmps[E1P].nx1);
  rr_.NewAthenaArray(dcmps[E1P].nx1);
  lambda22_.NewAthenaArray(dcmps[E1P].nx2, dcmps[E1P].nx1);
  lambda33_.NewAthenaArray(dcmps[E1P].nx3);
  dx1f2_.NewAthenaArray(ncells2);
  x1f2_.NewAthenaArray(ncells2+1);
  dx1v2_.NewAthenaArray(ncells2-1);
  x1v2_.NewAthenaArray(ncells2);
  arrsize = MAX(dcmps[X2P].block_size, dcmps[X3P].block_size);
  arrsize = MAX(arrsize, dcmps[X1P].block_size);
  in_  = fftw_alloc_complex(arrsize);
  out_  = fftw_alloc_complex(arrsize);
  arrsize = MAX(dcmps[E2P].block_size, dcmps[E3P].block_size);
  arrsize = MAX(arrsize, dcmps[E1P].block_size);
  arrsize = MAX(arrsize, dcmps[E2P0].block_size);
  in2_  = fftw_alloc_complex(arrsize);
  out2_  = fftw_alloc_complex(arrsize);
  bufsize = MAX(bufsize, arrsize);
  buf_ = fftw_alloc_complex(bufsize);
  grf[TOP][TOP] = fftw_alloc_complex( MAX(dcmps[Gii].block_size, dcmps[Gii2P].block_size) );
  grf[BOT][TOP] = fftw_alloc_complex( MAX(dcmps[Gii].block_size, dcmps[Gii2P].block_size) );
  grf[INN][TOP] = fftw_alloc_complex( MAX(dcmps[Gik].block_size, dcmps[Gik2P].block_size) );
  grf[OUT][TOP] = fftw_alloc_complex( MAX(dcmps[Gik].block_size, dcmps[Gik2P].block_size) );
  grf[TOP][BOT] = fftw_alloc_complex( MAX(dcmps[Gii].block_size, dcmps[Gii2P].block_size) );
  grf[BOT][BOT] = fftw_alloc_complex( MAX(dcmps[Gii].block_size, dcmps[Gii2P].block_size) );
  grf[INN][BOT] = fftw_alloc_complex( MAX(dcmps[Gik].block_size, dcmps[Gik2P].block_size) );
  grf[OUT][BOT] = fftw_alloc_complex( MAX(dcmps[Gik].block_size, dcmps[Gik2P].block_size) );
  grf[TOP][INN] = fftw_alloc_complex( MAX(dcmps[Gki].block_size, dcmps[Gki2P].block_size) );
  grf[BOT][INN] = fftw_alloc_complex( MAX(dcmps[Gki].block_size, dcmps[Gki2P].block_size) );
  grf[INN][INN] = fftw_alloc_complex( MAX(dcmps[Gkk].block_size, dcmps[Gkk2P].block_size) );
  grf[OUT][INN] = fftw_alloc_complex( MAX(dcmps[Gkk].block_size, dcmps[Gkk2P].block_size) );
  grf[TOP][OUT] = fftw_alloc_complex( MAX(dcmps[Gki].block_size, dcmps[Gki2P].block_size) );
  grf[BOT][OUT] = fftw_alloc_complex( MAX(dcmps[Gki].block_size, dcmps[Gki2P].block_size) );
  grf[INN][OUT] = fftw_alloc_complex( MAX(dcmps[Gkk].block_size, dcmps[Gkk2P].block_size) );
  grf[OUT][OUT] = fftw_alloc_complex( MAX(dcmps[Gkk].block_size, dcmps[Gkk2P].block_size) );
  for (int i=TOP;i<=BOT;++i) {
    sigma[i] = fftw_alloc_real( dcmps[XB].nx2*dcmps[XB].nx1 );
    sigma_fft[i] = fftw_alloc_complex( (hNx2+1)*Nx1/np1/np2 );
    sigma_fft_v[i] = fftw_alloc_complex( dcmps[Gii].nx2*dcmps[Gii].nx3 );
    sigma_fft_r[i] = fftw_alloc_complex( dcmps[Gki].nx2*dcmps[Gki].nx3 );
    psi[i] = fftw_alloc_real( dcmps[XB].nx2*dcmps[XB].nx1 );
    psi2[i] = fftw_alloc_real( dcmps[EB].nx2*lNx1 );
    arrsize = MAX(dcmps[Gii].nx2*dcmps[Gii].nx1, (hNx2+1)*Nx1/np1/np2);
    psi_fft[i] = fftw_alloc_complex( arrsize );
  }
  for (int i=INN;i<=OUT;++i) {
    sigma[i] = fftw_alloc_real( dcmps[XB].nx2*dcmps[XB].nx3 );
    sigma_fft[i] = fftw_alloc_complex( (hNx2+1)*Nx3/np2/np3 );
    sigma_fft_v[i] = fftw_alloc_complex( dcmps[Gik].nx2*dcmps[Gik].nx3 );
    sigma_fft_r[i] = fftw_alloc_complex( dcmps[Gkk].nx2*dcmps[Gkk].nx3 );
    psi[i] = fftw_alloc_real( dcmps[XB].nx2*dcmps[XB].nx3 );
    psi2[i] = fftw_alloc_real( dcmps[EB].nx2*(Nx3+2*ngh_grf_) );
    arrsize = MAX(dcmps[Gki].nx2*dcmps[Gki].nx1, (hNx2+1)*Nx3/np2/np3);
    psi_fft[i] = fftw_alloc_complex( arrsize );
  }

  // fill arrays
  dx2_ = pcrd->dx2v(NGHOST);
  dx3_ = pcrd->dx3v(NGHOST);
  int noffset;

  if (rat == 1.0) { // uniform spacing
    Real dx=pcrd->dx1f(NGHOST);
    for (int i=0; i<=ncells; ++i) {
      noffset = i-NGHOST;
      x1f_(i) = mesh_size.x1min + noffset*dx;
    }
    for (int i=0; i<ncells; ++i) {
      dx1f_(i)=dx;
      x1v_(i) = 0.5*(x1f_(i+1) + x1f_(i));
    }
    for (int i=0; i<=ncells2; ++i) {
      noffset = i-(NGHOST+noffset1_);
      x1f2_(i) = mesh_size.x1min + noffset*dx;
    }
    for (int i=0; i<ncells2; ++i) {
      dx1f2_(i)=dx;
      x1v2_(i) = 0.5*(x1f2_(i+1) + x1f2_(i));
    }
  }
  else { // logarithmic spacing
    for (int i=0; i<=ncells; ++i) {
      noffset = i-NGHOST;
      x1f_(i) = mesh_size.x1min * pow(rat, noffset);
    }
    for (int i=0; i<ncells; ++i) {
      dx1f_(i)=x1f_(i+1)-x1f_(i);
      x1v_(i) = (TWO_3RD)*(pow(x1f_(i+1),3)-pow(x1f_(i),3))/(pow(x1f_(i+1),2) - pow(x1f_(i),2));
    }
    for (int i=0; i<=ncells2; ++i) {
      noffset = i-(NGHOST+noffset1_);
      x1f2_(i) = mesh_size.x1min * pow(rat, noffset);
    }
    for(int i=0; i<ncells2; ++i) {
      dx1f2_(i)=x1f2_(i+1)-x1f2_(i);
      x1v2_(i) = (TWO_3RD)*(pow(x1f2_(i+1),3)-pow(x1f2_(i),3))/(pow(x1f2_(i+1),2) - pow(x1f2_(i),2));
    }
  }
  for (int i=0; i<ncells-1; ++i) {
    dx1v_(i) = x1v_(i+1) - x1v_(i);
  }
  for (int i=0; i<ncells2-1; ++i) {
    dx1v2_(i) = x1v2_(i+1) - x1v2_(i);
  }
  if ((rat != 1.0) && (rat !=  pow(mesh_size.x1max/mesh_size.x1min, 1./mesh_size.nx1))) {
    std::stringstream msg;
    msg << "### FATAL ERROR in CYLGRAV" << std::endl
         << "cell ratio must be *logarithmic*, such that "
         << "x1rat = (R_max/R_min)^(1/N_R)." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  for (int i=0;i<dcmps[X1P].nx1;++i) {
    if (mesh_size.x1rat == 1.0) {
      a_(i) = 1.0 / SQR(dx1v_(i+NGHOST)) - 0.5/dx1v_(i+NGHOST)/x1v_(i+NGHOST);
      c_(i) = 1.0 / SQR(dx1v_(i+NGHOST)) + 0.5/dx1v_(i+NGHOST)/x1v_(i+NGHOST);
    }
    else {
      a_(i) = SQR( 1.0 / std::log(rat) / x1v_(i+NGHOST) );
      c_(i) = SQR( 1.0 / std::log(rat) / x1v_(i+NGHOST) );
    }
  }
  for (int k=0;k<dcmps[X1P].nx3;++k) {
    lambda3_(k) = -4*SQR(sin(0.5*PI*(Real)(dcmps[X1P].ks+k+1)/((Real)((Nx3)+1))))/SQR(dx3_);
  }
  for (int j=0;j<dcmps[X1P].nx2;++j) {
    for (int i=0;i<dcmps[X1P].nx1;++i) {
      lambda2_(j,i) = -4*SQR(sin(PI*(Real)(dcmps[X1P].js+j) / ((Real)Nx2)))
                     / SQR(dx2_) / SQR(x1v_(i+NGHOST));
    }
  }
  for (int i=0;i<dcmps[E1P].nx1;++i) {
    if (mesh_size.x1rat == 1.0) {
      aa_(i) = 1.0 / SQR(dx1v2_(i+NGHOST)) - 0.5/dx1v2_(i+NGHOST)/x1v2_(i+NGHOST);
      cc_(i) = 1.0 / SQR(dx1v2_(i+NGHOST)) + 0.5/dx1v2_(i+NGHOST)/x1v2_(i+NGHOST);
    }
    else {
      aa_(i) = SQR( 1.0 / std::log(rat) / x1v2_(i+NGHOST) );
      cc_(i) = SQR( 1.0 / std::log(rat) / x1v2_(i+NGHOST) );
    }
  }
  for (int k=0;k<dcmps[E1P].nx3;++k) {
    lambda33_(k) = -4*SQR(sin(0.5*PI*(Real)(dcmps[E1P].ks+k+1)/((Real)((Nx3+2*ngh_grf_)+1))))/SQR(dx3_);
  }
  for (int j=0;j<dcmps[E1P].nx2;++j) {
    for (int i=0;i<dcmps[E1P].nx1;++i) {
      lambda22_(j,i) = -4*SQR(sin(PI*(Real)(dcmps[E1P].js+j) / ((Real)Nx2)))
                     / SQR(dx2_) / SQR(x1v2_(i+NGHOST));
    }
  }

  // Initialize FFT and remap plans
  fftw_r2r_kind kind[] = {FFTW_RODFT00};
  fft_x2_forward_[0] = fftw_plan_many_dft_r2c(1, &(dcmps[X2P0].nx2),
    dcmps[X2P0].block_size/dcmps[X2P0].nx2, ((Real*)in_), NULL, 1, dcmps[X2P0].nx2,
    in_, NULL, 1, dcmps[X2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[1] = fftw_plan_many_dft_r2c(1, &(dcmps[E2P0].nx2),
    dcmps[E2P0].block_size/dcmps[E2P0].nx2, ((Real*)in2_), NULL, 1, dcmps[E2P0].nx2,
    in2_, NULL, 1, dcmps[E2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[2] = fftw_plan_many_dft_r2c(1, &(dcmps[Gii2P0].nx2),
    dcmps[Gii2P0].block_size/dcmps[Gii2P0].nx2, ((Real*)(grf[TOP][TOP])), NULL, 1, dcmps[Gii2P0].nx2,
    grf[TOP][TOP], NULL, 1, dcmps[Gii2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[3] = fftw_plan_many_dft_r2c(1, &(dcmps[Gii2P0].nx2),
    dcmps[Gii2P0].block_size/dcmps[Gii2P0].nx2, ((Real*)(grf[TOP][BOT])), NULL, 1, dcmps[Gii2P0].nx2,
    grf[TOP][BOT], NULL, 1, dcmps[Gii2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[4] = fftw_plan_many_dft_r2c(1, &(dcmps[Gii2P0].nx2),
    dcmps[Gii2P0].block_size/dcmps[Gii2P0].nx2, ((Real*)(grf[BOT][TOP])), NULL, 1, dcmps[Gii2P0].nx2,
    grf[BOT][TOP], NULL, 1, dcmps[Gii2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[5] = fftw_plan_many_dft_r2c(1, &(dcmps[Gii2P0].nx2),
    dcmps[Gii2P0].block_size/dcmps[Gii2P0].nx2, ((Real*)(grf[BOT][BOT])), NULL, 1, dcmps[Gii2P0].nx2,
    grf[BOT][BOT], NULL, 1, dcmps[Gii2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[6] = fftw_plan_many_dft_r2c(1, &(dcmps[Gik2P0].nx2),
    dcmps[Gik2P0].block_size/dcmps[Gik2P0].nx2, ((Real*)(grf[INN][TOP])), NULL, 1, dcmps[Gik2P0].nx2,
    grf[INN][TOP], NULL, 1, dcmps[Gik2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[7] = fftw_plan_many_dft_r2c(1, &(dcmps[Gik2P0].nx2),
    dcmps[Gik2P0].block_size/dcmps[Gik2P0].nx2, ((Real*)(grf[INN][BOT])), NULL, 1, dcmps[Gik2P0].nx2,
    grf[INN][BOT], NULL, 1, dcmps[Gik2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[8] = fftw_plan_many_dft_r2c(1, &(dcmps[Gik2P0].nx2),
    dcmps[Gik2P0].block_size/dcmps[Gik2P0].nx2, ((Real*)(grf[OUT][TOP])), NULL, 1, dcmps[Gik2P0].nx2,
    grf[OUT][TOP], NULL, 1, dcmps[Gik2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[9] = fftw_plan_many_dft_r2c(1, &(dcmps[Gik2P0].nx2),
    dcmps[Gik2P0].block_size/dcmps[Gik2P0].nx2, ((Real*)(grf[OUT][BOT])), NULL, 1, dcmps[Gik2P0].nx2,
    grf[OUT][BOT], NULL, 1, dcmps[Gik2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[10] = fftw_plan_many_dft_r2c(1, &(dcmps[Gki2P0].nx2),
    dcmps[Gki2P0].block_size/dcmps[Gki2P0].nx2, ((Real*)(grf[TOP][INN])), NULL, 1, dcmps[Gki2P0].nx2,
    grf[TOP][INN], NULL, 1, dcmps[Gki2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[11] = fftw_plan_many_dft_r2c(1, &(dcmps[Gki2P0].nx2),
    dcmps[Gki2P0].block_size/dcmps[Gki2P0].nx2, ((Real*)(grf[TOP][OUT])), NULL, 1, dcmps[Gki2P0].nx2,
    grf[TOP][OUT], NULL, 1, dcmps[Gki2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[12] = fftw_plan_many_dft_r2c(1, &(dcmps[Gki2P0].nx2),
    dcmps[Gki2P0].block_size/dcmps[Gki2P0].nx2, ((Real*)(grf[BOT][INN])), NULL, 1, dcmps[Gki2P0].nx2,
    grf[BOT][INN], NULL, 1, dcmps[Gki2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[13] = fftw_plan_many_dft_r2c(1, &(dcmps[Gki2P0].nx2),
    dcmps[Gki2P0].block_size/dcmps[Gki2P0].nx2, ((Real*)(grf[BOT][OUT])), NULL, 1, dcmps[Gki2P0].nx2,
    grf[BOT][OUT], NULL, 1, dcmps[Gki2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[14] = fftw_plan_many_dft_r2c(1, &(dcmps[Gkk2P0].nx2),
    dcmps[Gkk2P0].block_size/dcmps[Gkk2P0].nx2, ((Real*)(grf[INN][INN])), NULL, 1, dcmps[Gkk2P0].nx2,
    grf[INN][INN], NULL, 1, dcmps[Gkk2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[15] = fftw_plan_many_dft_r2c(1, &(dcmps[Gkk2P0].nx2),
    dcmps[Gkk2P0].block_size/dcmps[Gkk2P0].nx2, ((Real*)(grf[INN][OUT])), NULL, 1, dcmps[Gkk2P0].nx2,
    grf[INN][OUT], NULL, 1, dcmps[Gkk2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[16] = fftw_plan_many_dft_r2c(1, &(dcmps[Gkk2P0].nx2),
    dcmps[Gkk2P0].block_size/dcmps[Gkk2P0].nx2, ((Real*)(grf[OUT][INN])), NULL, 1, dcmps[Gkk2P0].nx2,
    grf[OUT][INN], NULL, 1, dcmps[Gkk2P].nx2, FFTW_MEASURE);
  fft_x2_forward_[17] = fftw_plan_many_dft_r2c(1, &(dcmps[Gkk2P0].nx2),
    dcmps[Gkk2P0].block_size/dcmps[Gkk2P0].nx2, ((Real*)(grf[OUT][OUT])), NULL, 1, dcmps[Gkk2P0].nx2,
    grf[OUT][OUT], NULL, 1, dcmps[Gkk2P].nx2, FFTW_MEASURE);

  fft_x2_backward_[0] = fftw_plan_many_dft_c2r(1, &(dcmps[X2P0].nx2),
    dcmps[X2P0].block_size/dcmps[X2P0].nx2, out_, NULL, 1, dcmps[X2P].nx2,
    ((Real*)out_), NULL, 1, dcmps[X2P0].nx2, FFTW_MEASURE);
  fft_x2_backward_[1] = fftw_plan_many_dft_c2r(1, &(dcmps[E2P0].nx2),
    dcmps[E2P0].block_size/dcmps[E2P0].nx2, out2_, NULL, 1, dcmps[E2P].nx2,
    ((Real*)out2_), NULL, 1, dcmps[E2P0].nx2, FFTW_MEASURE);
  fft_x3_r2r_[0] = fftw_plan_many_r2r(1, &(dcmps[X3P].nx3),
    dcmps[X3P].block_size/dcmps[X3P].nx3, (Real*)in_, NULL, 2,
    2*dcmps[X3P].nx3, (Real*)in_, NULL, 2, 2*dcmps[X3P].nx3, kind, FFTW_MEASURE);
  fft_x3_r2r_[1] = fftw_plan_many_r2r(1, &(dcmps[X3P].nx3),
    dcmps[X3P].block_size/dcmps[X3P].nx3, (Real*)in_+1, NULL, 2,
    2*dcmps[X3P].nx3, (Real*)in_+1, NULL, 2, 2*dcmps[X3P].nx3, kind, FFTW_MEASURE);
  fft_x3_r2r_[2] = fftw_plan_many_r2r(1, &(dcmps[X3P].nx3),
    dcmps[X3P].block_size/dcmps[X3P].nx3, (Real*)out_, NULL, 2,
    2*dcmps[X3P].nx3, (Real*)out_, NULL, 2, 2*dcmps[X3P].nx3, kind, FFTW_MEASURE);
  fft_x3_r2r_[3] = fftw_plan_many_r2r(1, &(dcmps[X3P].nx3),
    dcmps[X3P].block_size/dcmps[X3P].nx3, (Real*)out_+1, NULL, 2,
    2*dcmps[X3P].nx3, (Real*)out_+1, NULL, 2, 2*dcmps[X3P].nx3, kind, FFTW_MEASURE);
  fft_x3_r2r_[4] = fftw_plan_many_r2r(1, &(dcmps[E3P].nx3),
    dcmps[E3P].block_size/dcmps[E3P].nx3, (Real*)in2_, NULL, 2,
    2*dcmps[E3P].nx3, (Real*)in2_, NULL, 2, 2*dcmps[E3P].nx3, kind, FFTW_MEASURE);
  fft_x3_r2r_[5] = fftw_plan_many_r2r(1, &(dcmps[E3P].nx3),
    dcmps[E3P].block_size/dcmps[E3P].nx3, (Real*)in2_+1, NULL, 2,
    2*dcmps[E3P].nx3, (Real*)in2_+1, NULL, 2, 2*dcmps[E3P].nx3, kind, FFTW_MEASURE);
  fft_x3_r2r_[6] = fftw_plan_many_r2r(1, &(dcmps[E3P].nx3),
    dcmps[E3P].block_size/dcmps[E3P].nx3, (Real*)out2_, NULL, 2,
    2*dcmps[E3P].nx3, (Real*)out2_, NULL, 2, 2*dcmps[E3P].nx3, kind, FFTW_MEASURE);
  fft_x3_r2r_[7] = fftw_plan_many_r2r(1, &(dcmps[E3P].nx3),
    dcmps[E3P].block_size/dcmps[E3P].nx3, (Real*)out2_+1, NULL, 2,
    2*dcmps[E3P].nx3, (Real*)out2_+1, NULL, 2, 2*dcmps[E3P].nx3, kind, FFTW_MEASURE);

  RmpPlan[XB][X2P0] = remap_3d_create_plan(MPI_COMM_WORLD,dcmps[XB].is,
    dcmps[XB].ie, dcmps[XB].js, dcmps[XB].je, dcmps[XB].ks, dcmps[XB].ke,
    dcmps[X2P0].is, dcmps[X2P0].ie, dcmps[X2P0].js, dcmps[X2P0].je,
    dcmps[X2P0].ks, dcmps[X2P0].ke, 1, 1, 0, 2);
  RmpPlan[X2P][X3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[X2P].js,
    dcmps[X2P].je, dcmps[X2P].ks, dcmps[X2P].ke, dcmps[X2P].is, dcmps[X2P].ie,
    dcmps[X3P].js, dcmps[X3P].je, dcmps[X3P].ks, dcmps[X3P].ke, dcmps[X3P].is,
    dcmps[X3P].ie, 2, 1, 0, 2);
  RmpPlan[X3P][X1P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[X3P].ks,
    dcmps[X3P].ke, dcmps[X3P].is, dcmps[X3P].ie, dcmps[X3P].js, dcmps[X3P].je,
    dcmps[X1P].ks, dcmps[X1P].ke, dcmps[X1P].is, dcmps[X1P].ie, dcmps[X1P].js,
    dcmps[X1P].je, 2, 1, 0, 2);
  RmpPlan[X1P][X3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[X1P].is,
    dcmps[X1P].ie, dcmps[X1P].js, dcmps[X1P].je, dcmps[X1P].ks, dcmps[X1P].ke,
    dcmps[X3P].is, dcmps[X3P].ie, dcmps[X3P].js, dcmps[X3P].je, dcmps[X3P].ks,
    dcmps[X3P].ke, 2, 2, 0, 2);
  RmpPlan[X3P][X2P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[X3P].ks,
    dcmps[X3P].ke, dcmps[X3P].is, dcmps[X3P].ie, dcmps[X3P].js, dcmps[X3P].je,
    dcmps[X2P].ks, dcmps[X2P].ke, dcmps[X2P].is, dcmps[X2P].ie, dcmps[X2P].js,
    dcmps[X2P].je, 2, 2, 0, 2);
  RmpPlan[X2P0][XB] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[X2P0].js,
    dcmps[X2P0].je, dcmps[X2P0].ks, dcmps[X2P0].ke, dcmps[X2P0].is,
    dcmps[X2P0].ie, dcmps[XB].js, dcmps[XB].je, dcmps[XB].ks, dcmps[XB].ke,
    dcmps[XB].is, dcmps[XB].ie, 1, 2, 0, 2);
  RmpPlan[E2P][E3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[E2P].js,
    dcmps[E2P].je, dcmps[E2P].ks, dcmps[E2P].ke, dcmps[E2P].is, dcmps[E2P].ie,
    dcmps[E3P].js, dcmps[E3P].je, dcmps[E3P].ks, dcmps[E3P].ke, dcmps[E3P].is,
    dcmps[E3P].ie, 2, 1, 0, 2);
  RmpPlan[E3P][E1P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[E3P].ks,
    dcmps[E3P].ke, dcmps[E3P].is, dcmps[E3P].ie, dcmps[E3P].js, dcmps[E3P].je,
    dcmps[E1P].ks, dcmps[E1P].ke, dcmps[E1P].is, dcmps[E1P].ie, dcmps[E1P].js,
    dcmps[E1P].je, 2, 1, 0, 2);
  RmpPlan[E1P][E3P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[E1P].is,
    dcmps[E1P].ie, dcmps[E1P].js, dcmps[E1P].je, dcmps[E1P].ks, dcmps[E1P].ke,
    dcmps[E3P].is, dcmps[E3P].ie, dcmps[E3P].js, dcmps[E3P].je, dcmps[E3P].ks,
    dcmps[E3P].ke, 2, 2, 0, 2);
  RmpPlan[E3P][E2P] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[E3P].ks,
    dcmps[E3P].ke, dcmps[E3P].is, dcmps[E3P].ie, dcmps[E3P].js, dcmps[E3P].je,
    dcmps[E2P].ks, dcmps[E2P].ke, dcmps[E2P].is, dcmps[E2P].ie, dcmps[E2P].js,
    dcmps[E2P].je, 2, 2, 0, 2);
  RmpPlan[E2P0][EB] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[E2P0].js,
    dcmps[E2P0].je, dcmps[E2P0].ks, dcmps[E2P0].ke, dcmps[E2P0].is,
    dcmps[E2P0].ie, dcmps[EB].js, dcmps[EB].je, dcmps[EB].ks, dcmps[EB].ke,
    dcmps[EB].is, dcmps[EB].ie, 1, 2, 0, 2);
  RmpPlan[Gii2P][Gii] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gii2P].js,
    dcmps[Gii2P].je, dcmps[Gii2P].ks, dcmps[Gii2P].ke, dcmps[Gii2P].is,
    dcmps[Gii2P].ie, dcmps[Gii].js, dcmps[Gii].je, dcmps[Gii].ks, dcmps[Gii].ke,
    dcmps[Gii].is, dcmps[Gii].ie, 2, 1, 0, 2);
  RmpPlan[Gik2P][Gik] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gik2P].js,
    dcmps[Gik2P].je, dcmps[Gik2P].ks, dcmps[Gik2P].ke, dcmps[Gik2P].is,
    dcmps[Gik2P].ie, dcmps[Gik].js, dcmps[Gik].je, dcmps[Gik].ks, dcmps[Gik].ke,
    dcmps[Gik].is, dcmps[Gik].ie, 2, 1, 0, 2);
  RmpPlan[Gki2P][Gki] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gki2P].js,
    dcmps[Gki2P].je, dcmps[Gki2P].is, dcmps[Gki2P].ie, dcmps[Gki2P].ks,
    dcmps[Gki2P].ke, dcmps[Gki].js, dcmps[Gki].je, dcmps[Gki].is, dcmps[Gki].ie,
    dcmps[Gki].ks, dcmps[Gki].ke, 2, 2, 0, 2);
  RmpPlan[Gkk2P][Gkk] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gkk2P].js,
    dcmps[Gkk2P].je, dcmps[Gkk2P].is, dcmps[Gkk2P].ie, dcmps[Gkk2P].ks,
    dcmps[Gkk2P].ke, dcmps[Gkk].js, dcmps[Gkk].je, dcmps[Gkk].is, dcmps[Gkk].ie,
    dcmps[Gkk].ks, dcmps[Gkk].ke, 2, 2, 0, 2);

  RmpPlan[Gii_BLOCK][Gii2P0] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gii_BLOCK].js,
    dcmps[Gii_BLOCK].je, dcmps[Gii_BLOCK].ks, dcmps[Gii_BLOCK].ke, dcmps[Gii_BLOCK].is,
    dcmps[Gii_BLOCK].ie, dcmps[Gii2P0].js, dcmps[Gii2P0].je, dcmps[Gii2P0].ks, dcmps[Gii2P0].ke,
    dcmps[Gii2P0].is, dcmps[Gii2P0].ie, 1, 0, 0, 2);
  RmpPlan[Gik_BLOCK][Gik2P0] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gik_BLOCK].js,
    dcmps[Gik_BLOCK].je, dcmps[Gik_BLOCK].ks, dcmps[Gik_BLOCK].ke, dcmps[Gik_BLOCK].is,
    dcmps[Gik_BLOCK].ie, dcmps[Gik2P0].js, dcmps[Gik2P0].je, dcmps[Gik2P0].ks, dcmps[Gik2P0].ke,
    dcmps[Gik2P0].is, dcmps[Gik2P0].ie, 1, 0, 0, 2);
  RmpPlan[Gki_BLOCK][Gki2P0] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gki_BLOCK].js,
    dcmps[Gki_BLOCK].je, dcmps[Gki_BLOCK].is, dcmps[Gki_BLOCK].ie, dcmps[Gki_BLOCK].ks,
    dcmps[Gki_BLOCK].ke, dcmps[Gki2P0].js, dcmps[Gki2P0].je, dcmps[Gki2P0].is, dcmps[Gki2P0].ie,
    dcmps[Gki2P0].ks, dcmps[Gki2P0].ke, 1, 0, 0, 2);
  RmpPlan[Gkk_BLOCK][Gkk2P0] = remap_3d_create_plan(MPI_COMM_WORLD, dcmps[Gkk_BLOCK].js,
    dcmps[Gkk_BLOCK].je, dcmps[Gkk_BLOCK].is, dcmps[Gkk_BLOCK].ie, dcmps[Gkk_BLOCK].ks,
    dcmps[Gkk_BLOCK].ke, dcmps[Gkk2P0].js, dcmps[Gkk2P0].je, dcmps[Gkk2P0].is, dcmps[Gkk2P0].ie,
    dcmps[Gkk2P0].ks, dcmps[Gkk2P0].ke, 1, 0, 0, 2);

  for (int i=TOP;i<=BOT;++i) {
    BndryRmpPlan[i][BLOCK][FFT_LONG] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][BLOCK].is, bndry_dcmps[i][BLOCK].ie,
      bndry_dcmps[i][BLOCK].js, bndry_dcmps[i][BLOCK].je,
      bndry_dcmps[i][FFT_LONG].is, bndry_dcmps[i][FFT_LONG].ie,
      bndry_dcmps[i][FFT_LONG].js, bndry_dcmps[i][FFT_LONG].je, 1, 1, 0, 2);
    BndryRmpPlan[i][FFT_SHORT][SIGv] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_SHORT].js, bndry_dcmps[i][FFT_SHORT].je,
      bndry_dcmps[i][FFT_SHORT].is, bndry_dcmps[i][FFT_SHORT].ie,
      bndry_dcmps[i][SIGv].js, bndry_dcmps[i][SIGv].je,
      bndry_dcmps[i][SIGv].is, bndry_dcmps[i][SIGv].ie, 2, 1, 0, 2);
    BndryRmpPlan[i][FFT_SHORT][SIGr] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_SHORT].js, bndry_dcmps[i][FFT_SHORT].je,
      bndry_dcmps[i][FFT_SHORT].is, bndry_dcmps[i][FFT_SHORT].ie,
      bndry_dcmps[i][SIGr].js, bndry_dcmps[i][SIGr].je,
      bndry_dcmps[i][SIGr].is, bndry_dcmps[i][SIGr].ie, 2, 1, 0, 2);
    BndryRmpPlan[i][PSI][FFT_SHORT] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][PSI].is, bndry_dcmps[i][PSI].ie,
      bndry_dcmps[i][PSI].js, bndry_dcmps[i][PSI].je,
      bndry_dcmps[i][FFT_SHORT].is, bndry_dcmps[i][FFT_SHORT].ie,
      bndry_dcmps[i][FFT_SHORT].js, bndry_dcmps[i][FFT_SHORT].je, 2, 1, 0, 2);
    BndryRmpPlan[i][FFT_LONG][BLOCK] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_LONG].js, bndry_dcmps[i][FFT_LONG].je,
      bndry_dcmps[i][FFT_LONG].is, bndry_dcmps[i][FFT_LONG].ie,
      bndry_dcmps[i][BLOCK].js, bndry_dcmps[i][BLOCK].je,
      bndry_dcmps[i][BLOCK].is, bndry_dcmps[i][BLOCK].ie, 1, 1, 0, 2);
    fft_2d_plan[i][0] = fftw_plan_many_dft_r2c(1, &(Nx2),
      bndry_dcmps[i][FFT_LONG].block_size/Nx2, sigma[i], NULL, 1, Nx2,
      sigma_fft[i], NULL, 1, hNx2+1, FFTW_MEASURE);
    fft_2d_plan[i][1] = fftw_plan_many_dft_c2r(1, &(Nx2),
      bndry_dcmps[i][FFT_LONG].block_size/Nx2, psi_fft[i], NULL, 1, hNx2+1,
      psi[i], NULL, 1, Nx2, FFTW_MEASURE);
  }
  for (int i=INN;i<=OUT;++i) {
    BndryRmpPlan[i][BLOCK][FFT_LONG] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][BLOCK].js, bndry_dcmps[i][BLOCK].je,
      bndry_dcmps[i][BLOCK].ks, bndry_dcmps[i][BLOCK].ke,
      bndry_dcmps[i][FFT_LONG].js, bndry_dcmps[i][FFT_LONG].je,
      bndry_dcmps[i][FFT_LONG].ks, bndry_dcmps[i][FFT_LONG].ke, 1, 0, 0, 2);
    BndryRmpPlan[i][FFT_SHORT][SIGv] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_SHORT].js, bndry_dcmps[i][FFT_SHORT].je,
      bndry_dcmps[i][FFT_SHORT].ks, bndry_dcmps[i][FFT_SHORT].ke,
      bndry_dcmps[i][SIGv].js, bndry_dcmps[i][SIGv].je,
      bndry_dcmps[i][SIGv].ks, bndry_dcmps[i][SIGv].ke, 2, 1, 0, 2);
    BndryRmpPlan[i][FFT_SHORT][SIGr] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_SHORT].js, bndry_dcmps[i][FFT_SHORT].je,
      bndry_dcmps[i][FFT_SHORT].ks, bndry_dcmps[i][FFT_SHORT].ke,
      bndry_dcmps[i][SIGr].js, bndry_dcmps[i][SIGr].je,
      bndry_dcmps[i][SIGr].ks, bndry_dcmps[i][SIGr].ke, 2, 1, 0, 2);
    BndryRmpPlan[i][PSI][FFT_SHORT] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][PSI].js, bndry_dcmps[i][PSI].je,
      bndry_dcmps[i][PSI].ks, bndry_dcmps[i][PSI].ke,
      bndry_dcmps[i][FFT_SHORT].js, bndry_dcmps[i][FFT_SHORT].je,
      bndry_dcmps[i][FFT_SHORT].ks, bndry_dcmps[i][FFT_SHORT].ke, 2, 0, 0, 2);
    BndryRmpPlan[i][FFT_LONG][BLOCK] = remap_2d_create_plan(MPI_COMM_WORLD,
      bndry_dcmps[i][FFT_LONG].js, bndry_dcmps[i][FFT_LONG].je,
      bndry_dcmps[i][FFT_LONG].ks, bndry_dcmps[i][FFT_LONG].ke,
      bndry_dcmps[i][BLOCK].js, bndry_dcmps[i][BLOCK].je,
      bndry_dcmps[i][BLOCK].ks, bndry_dcmps[i][BLOCK].ke, 1, 0, 0, 2);
    fft_2d_plan[i][0] = fftw_plan_many_dft_r2c(1, &(Nx2),
      bndry_dcmps[i][FFT_LONG].block_size/Nx2, sigma[i], NULL, 1, Nx2,
      sigma_fft[i], NULL, 1, hNx2+1, FFTW_MEASURE);
    fft_2d_plan[i][1] = fftw_plan_many_dft_c2r(1, &(Nx2),
      bndry_dcmps[i][FFT_LONG].block_size/Nx2, psi_fft[i], NULL, 1, hNx2+1,
      psi[i], NULL, 1, Nx2, FFTW_MEASURE);
  }

  FillDscGrf();
//  FillCntGrf();
  fftw_free(in2_);
  fftw_free(out2_);
  lambda22_.DeleteAthenaArray();
  lambda33_.DeleteAthenaArray();
  x1f2_.DeleteAthenaArray();
  dx1f2_.DeleteAthenaArray();
  x1v2_.DeleteAthenaArray();
  dx1v2_.DeleteAthenaArray();
  for (int i=1;i<18;++i) fftw_destroy_plan(fft_x2_forward_[i]);
  fftw_destroy_plan(fft_x2_backward_[1]);
}

// OBCGravityCyl destructor
OBCGravityCyl::~OBCGravityCyl()
{

  for (int i=0;i<4;++i) {
    fftw_free(psi[i]);
    fftw_free(psi2[i]);
    fftw_free(sigma[i]);
    fftw_free(psi_fft[i]);
    fftw_free(sigma_fft[i]);
    fftw_free(sigma_fft_v[i]);
    fftw_free(sigma_fft_r[i]);
    fftw_destroy_plan(fft_2d_plan[i][0]);
    fftw_destroy_plan(fft_2d_plan[i][1]);
    remap_2d_destroy_plan(BndryRmpPlan[i][BLOCK][FFT_LONG]);
    remap_2d_destroy_plan(BndryRmpPlan[i][FFT_SHORT][SIGv]);
    remap_2d_destroy_plan(BndryRmpPlan[i][FFT_SHORT][SIGr]);
    remap_2d_destroy_plan(BndryRmpPlan[i][PSI][FFT_SHORT]);
    remap_2d_destroy_plan(BndryRmpPlan[i][FFT_LONG][BLOCK]);
  }
  for (int i=TOP;i<=OUT;++i) {
    for (int j=TOP;j<=OUT;++j) {
      fftw_free(grf[i][j]);
    }
  }
  for (int i=0;i<8;++i) fftw_destroy_plan(fft_x3_r2r_[i]);
  fftw_free(in_);
  fftw_free(out_);
  fftw_free(buf_);
  a_.DeleteAthenaArray();
  b_.DeleteAthenaArray();
  c_.DeleteAthenaArray();
  x_.DeleteAthenaArray();
  r_.DeleteAthenaArray();
  lambda2_.DeleteAthenaArray();
  lambda3_.DeleteAthenaArray();
  x1f_.DeleteAthenaArray();
  dx1f_.DeleteAthenaArray();
  x1v_.DeleteAthenaArray();
  dx1v_.DeleteAthenaArray();
  aa_.DeleteAthenaArray();
  bb_.DeleteAthenaArray();
  cc_.DeleteAthenaArray();
  xx_.DeleteAthenaArray();
  rr_.DeleteAthenaArray();
  fftw_destroy_plan(fft_x2_forward_[0]);
  fftw_destroy_plan(fft_x2_backward_[0]);
  remap_3d_destroy_plan(RmpPlan[XB][X2P0]);
  remap_3d_destroy_plan(RmpPlan[X2P][X3P]);
  remap_3d_destroy_plan(RmpPlan[X3P][X1P]);
  remap_3d_destroy_plan(RmpPlan[X1P][X3P]);
  remap_3d_destroy_plan(RmpPlan[X3P][X2P]);
  remap_3d_destroy_plan(RmpPlan[X2P0][XB]);
  remap_3d_destroy_plan(RmpPlan[E2P][E3P]);
  remap_3d_destroy_plan(RmpPlan[E3P][E1P]);
  remap_3d_destroy_plan(RmpPlan[E1P][E3P]);
  remap_3d_destroy_plan(RmpPlan[E3P][E2P]);
  remap_3d_destroy_plan(RmpPlan[E2P0][EB]);
  remap_3d_destroy_plan(RmpPlan[Gii2P][Gii]);
  remap_3d_destroy_plan(RmpPlan[Gik2P][Gik]);
  remap_3d_destroy_plan(RmpPlan[Gki2P][Gki]);
  remap_3d_destroy_plan(RmpPlan[Gkk2P][Gkk]);
  remap_3d_destroy_plan(RmpPlan[Gii_BLOCK][Gii2P0]);
  remap_3d_destroy_plan(RmpPlan[Gik_BLOCK][Gik2P0]);
  remap_3d_destroy_plan(RmpPlan[Gki_BLOCK][Gki2P0]);
  remap_3d_destroy_plan(RmpPlan[Gkk_BLOCK][Gkk2P0]);
  for (int i=0;i<4;++i) {
    if (bndcomm[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&(bndcomm[i]));
    }
  }
  MPI_Comm_free(&x1comm);
  MPI_Comm_free(&x3comm);

}

void OBCGravityCyl::FillCntGrf()
{
  int idx, idx2, gip, gjp, gkp;
  int gi,gj,gk;
  Real R, Rp, rds, psi, dV;
  if (Globals::my_rank == 0)
    std::cout << "Calculating discrete Green's functions..." << std::endl;
  gjp = 0;

  for (int i=0;i<dcmps[Gii2P0].nx1;++i) {
    for (int ip=0;ip<dcmps[Gii2P0].nx3;++ip) {
      for (int j=0;j<dcmps[Gii2P0].nx2;++j) {
        idx = j + dcmps[Gii2P0].nx2*(ip + dcmps[Gii2P0].nx3*i);
        gj = dcmps[Gii2P0].js+j;
        gip = dcmps[Gii2P0].ks + ip; 
        gi = dcmps[Gii2P0].is+i;
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        Rp = x1v_(NGHOST+gip);
        R  = x1v_(NGHOST+gi);
        gkp = Nx3;
        gk = Nx3;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) (((Real*)(grf[TOP][TOP])))[idx] = 0;
        else ((Real*)(grf[TOP][TOP]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gkp = Nx3;
        gk = -1;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) ((Real*)(grf[TOP][BOT]))[idx] = 0;
        else ((Real*)(grf[TOP][BOT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gkp = -1;
        gk = Nx3;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) ((Real*)(grf[BOT][TOP]))[idx] = 0;
        else ((Real*)(grf[BOT][TOP]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gkp = -1;
        gk = -1;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) ((Real*)(grf[BOT][BOT]))[idx] = 0;
        else ((Real*)(grf[BOT][BOT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
      }
    }
  }
  for (int ip=0;ip<dcmps[Gki2P0].nx3;++ip) {
    for (int k=0;k<dcmps[Gki2P0].nx1;++k) {
      for (int j=0;j<dcmps[Gki2P0].nx2;++j) {
        idx = j + dcmps[Gki2P0].nx2*(k + dcmps[Gki2P0].nx1*ip);
        gj = dcmps[Gki2P0].js + j;
        gk = dcmps[Gki2P0].is + k; 
        gip = dcmps[Gki2P0].ks + ip;
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        Rp = x1v_(NGHOST+gip);
        gkp = Nx3;
        gi = -1;
        R  = x1v_(NGHOST+gi);
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) ((Real*)(grf[TOP][INN]))[idx] = 0;
        else ((Real*)(grf[TOP][INN]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gkp = Nx3;
        gi = Nx1;
        R  = x1v_(NGHOST+gi);
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) ((Real*)(grf[TOP][OUT]))[idx] = 0;
        else ((Real*)(grf[TOP][OUT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gkp = -1;
        gi = -1;
        R  = x1v_(NGHOST+gi);
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) ((Real*)(grf[BOT][INN]))[idx] = 0;
        else ((Real*)(grf[BOT][INN]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gkp = -1;
        gi = Nx1;
        R  = x1v_(NGHOST+gi);
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds == 0) ((Real*)(grf[BOT][OUT]))[idx] = 0;
        else ((Real*)(grf[BOT][OUT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
      }
    }
  }
  for (int i=0;i<dcmps[Gik2P0].nx1;++i) {
    for (int kp=0;kp<dcmps[Gik2P0].nx3;++kp) {
      for (int j=0;j<dcmps[Gik2P0].nx2;++j) {
        idx = j + dcmps[Gik2P0].nx2*(kp + dcmps[Gik2P0].nx3*i);
        gj = dcmps[Gik2P0].js + j;
        gkp = dcmps[Gik2P0].ks + kp;
        gi = dcmps[Gik2P0].is + i;
        R  = x1v_(NGHOST+gi);
        gip = -1;
        gk = Nx3;
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        Rp = x1v_(NGHOST+gip);
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds==0) ((Real*)(grf[INN][TOP]))[idx] = 0;
        else ((Real*)(grf[INN][TOP]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gk = -1;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds==0) ((Real*)(grf[INN][BOT]))[idx] = 0;
        else ((Real*)(grf[INN][BOT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gip = Nx1;
        gk = Nx3;
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        Rp = x1v_(NGHOST+gip);
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds==0) ((Real*)(grf[OUT][TOP]))[idx] = 0;
        else ((Real*)(grf[OUT][TOP]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
        gk = -1;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds==0) ((Real*)(grf[OUT][BOT]))[idx] = 0;
        else ((Real*)(grf[OUT][BOT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
      }
    }
  }
  for (int kp=0;kp<dcmps[Gkk2P0].nx3;++kp) {
    for (int k=0;k<dcmps[Gkk2P0].nx1;++k) {
      for (int j=0;j<dcmps[Gkk2P0].nx2;++j) {
        idx = j + dcmps[Gkk2P0].nx2*(k + dcmps[Gkk2P0].nx1*kp);
        gj = dcmps[Gkk2P0].js + j;
        gk = dcmps[Gkk2P0].is + k;
        gkp = dcmps[Gkk2P0].ks + kp;
        
        gi = -1;
        gip = -1;
        R  = x1v_(NGHOST+gi);
        Rp = x1v_(NGHOST+gip);
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if(rds==0) ((Real*)(grf[INN][INN]))[idx] = 0;
        else ((Real*)(grf[INN][INN]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;

        gi = Nx1;
        gip = -1;
        R  = x1v_(NGHOST+gi);
        Rp = x1v_(NGHOST+gip);
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds==0) ((Real*)(grf[INN][OUT]))[idx] = 0;
        else ((Real*)(grf[INN][OUT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;

        gi = -1;
        gip = Nx1;
        R  = x1v_(NGHOST+gi);
        Rp = x1v_(NGHOST+gip);
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds==0) ((Real*)(grf[OUT][INN]))[idx] = 0;
        else ((Real*)(grf[OUT][INN]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;

        gi = Nx1;
        gip = Nx1;
        R  = x1v_(NGHOST+gi);
        Rp = x1v_(NGHOST+gip);
        dV = 0.5*(SQR(x1f_(NGHOST+gip+1)) - SQR(x1f_(NGHOST+gip)))*dx2_*dx3_;
        rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)) + SQR(dx3_*(gk-gkp)));
        if (rds==0) ((Real*)(grf[OUT][OUT]))[idx] = 0;
        else ((Real*)(grf[OUT][OUT]))[idx] = -(four_pi_G/4.0/PI)*dV/rds;
      }
    }
  }

  for (int i=2;i<=17;++i)
    fftw_execute(fft_x2_forward_[i]);
  remap_3d((Real*)grf[TOP][TOP], (Real*)grf[TOP][TOP], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[TOP][BOT], (Real*)grf[TOP][BOT], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[BOT][TOP], (Real*)grf[BOT][TOP], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[BOT][BOT], (Real*)grf[BOT][BOT], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[INN][TOP], (Real*)grf[INN][TOP], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[INN][BOT], (Real*)grf[INN][BOT], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[OUT][TOP], (Real*)grf[OUT][TOP], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[OUT][BOT], (Real*)grf[OUT][BOT], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[TOP][INN], (Real*)grf[TOP][INN], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[TOP][OUT], (Real*)grf[TOP][OUT], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[BOT][INN], (Real*)grf[BOT][INN], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[BOT][OUT], (Real*)grf[BOT][OUT], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[INN][INN], (Real*)grf[INN][INN], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
  remap_3d((Real*)grf[INN][OUT], (Real*)grf[INN][OUT], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
  remap_3d((Real*)grf[OUT][INN], (Real*)grf[OUT][INN], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
  remap_3d((Real*)grf[OUT][OUT], (Real*)grf[OUT][OUT], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
}

void OBCGravityCyl::FillDscGrf()
{
  int idx, idx2, ip, kp, gip, gkp;
  if (Globals::my_rank == 0)
    std::cout << "Calculating discrete Green's functions..." << std::endl;

  // surface charge at top boundary 
  gkp = Nx3;
  for (gip=0;gip<Nx1;++gip) {
    CalcGrf(gip, gkp);
    // If this proc is responsible for the ip index, store the Green's function
    if ((gip >= dcmps[Gii_BLOCK].ks)&&(gip <= dcmps[Gii_BLOCK].ke)) {
      ip = gip - dcmps[Gii_BLOCK].ks;
      for (int i=0;i<dcmps[Gii_BLOCK].nx1;++i) {
        for (int j=0;j<dcmps[Gii_BLOCK].nx2;++j) {
          idx = j + dcmps[Gii_BLOCK].nx2*(ip + dcmps[Gii_BLOCK].nx3*i);
          idx2 = j + dcmps[Gii_BLOCK].nx2*(dcmps[Gii_BLOCK].is+i+noffset1_);
          ((Real*)(grf[TOP][TOP]))[idx] = psi2[TOP][idx2];
          ((Real*)(grf[TOP][BOT]))[idx] = psi2[BOT][idx2];
        }
      }
    }
    if ((gip >= dcmps[Gki_BLOCK].ks)&&(gip <= dcmps[Gki_BLOCK].ke)) {
      ip = gip - dcmps[Gki_BLOCK].ks;
      for (int k=0;k<dcmps[Gki_BLOCK].nx1;++k) {
        for (int j=0;j<dcmps[Gki_BLOCK].nx2;++j) {
          idx = j + dcmps[Gki_BLOCK].nx2*(k + dcmps[Gki_BLOCK].nx1*ip);
          idx2 = j + dcmps[Gki_BLOCK].nx2*(dcmps[Gki_BLOCK].is+k+noffset2_);
          ((Real*)(grf[TOP][INN]))[idx] = psi2[INN][idx2];
          ((Real*)(grf[TOP][OUT]))[idx] = psi2[OUT][idx2];
        }
      }
    }
    if (Globals::my_rank == 0) std::cout << "top boundary gip = " << gip << std::endl;
  }
  // surface charge at bottom boundary 
  gkp = -1;
  for (gip=0;gip<Nx1;++gip) {
    CalcGrf(gip, gkp);
    // If this proc is responsible for the ip index, store the Green's function
    if ((gip >= dcmps[Gii_BLOCK].ks)&&(gip <= dcmps[Gii_BLOCK].ke)) {
      ip = gip - dcmps[Gii_BLOCK].ks;
      for (int i=0;i<dcmps[Gii_BLOCK].nx1;++i) {
        for (int j=0;j<dcmps[Gii_BLOCK].nx2;++j) {
          idx = j + dcmps[Gii_BLOCK].nx2*(ip + dcmps[Gii_BLOCK].nx3*i);
          idx2 = j + dcmps[Gii_BLOCK].nx2*(dcmps[Gii_BLOCK].is+i+noffset1_);
          ((Real*)(grf[BOT][TOP]))[idx] = psi2[TOP][idx2];
          ((Real*)(grf[BOT][BOT]))[idx] = psi2[BOT][idx2];
        }
      }
    }
    if ((gip >= dcmps[Gki_BLOCK].ks)&&(gip <= dcmps[Gki_BLOCK].ke)) {
      ip = gip - dcmps[Gki_BLOCK].ks;
      for (int k=0;k<dcmps[Gki_BLOCK].nx1;++k) {
        for (int j=0;j<dcmps[Gki_BLOCK].nx2;++j) {
          idx = j + dcmps[Gki_BLOCK].nx2*(k + dcmps[Gki_BLOCK].nx1*ip);
          idx2 = j + dcmps[Gki_BLOCK].nx2*(dcmps[Gki_BLOCK].is+k+noffset2_);
          ((Real*)(grf[BOT][INN]))[idx] = psi2[INN][idx2];
          ((Real*)(grf[BOT][OUT]))[idx] = psi2[OUT][idx2];
        }
      }
    }
    if (Globals::my_rank == 0) std::cout << "bot boundary gip = " << gip << std::endl;
  }
  // surface charge at inner boundary 
  gip = -1;
  for (gkp=0;gkp<Nx3;++gkp) {
    CalcGrf(gip, gkp);
    // If this proc is responsible for the ip index, store the Green's function
    if ((gkp >= dcmps[Gik_BLOCK].ks)&&(gkp <= dcmps[Gik_BLOCK].ke)) {
      kp = gkp - dcmps[Gik_BLOCK].ks;
      for (int i=0;i<dcmps[Gik_BLOCK].nx1;++i) {
        for (int j=0;j<dcmps[Gik_BLOCK].nx2;++j) {
          idx = j + dcmps[Gik_BLOCK].nx2*(kp + dcmps[Gik_BLOCK].nx3*i);
          idx2 = j + dcmps[Gii_BLOCK].nx2*(dcmps[Gii_BLOCK].is+i+noffset1_);
          ((Real*)(grf[INN][TOP]))[idx] = psi2[TOP][idx2];
          ((Real*)(grf[INN][BOT]))[idx] = psi2[BOT][idx2];
        }
      }
    }
    if ((gkp >= dcmps[Gkk_BLOCK].ks)&&(gkp <= dcmps[Gkk_BLOCK].ke)) {
      kp = gkp - dcmps[Gkk_BLOCK].ks;
      for (int k=0;k<dcmps[Gkk_BLOCK].nx1;++k) {
        for (int j=0;j<dcmps[Gkk_BLOCK].nx2;++j) {
          idx = j + dcmps[Gkk_BLOCK].nx2*(k + dcmps[Gkk_BLOCK].nx1*kp);
          idx2 = j + dcmps[Gki_BLOCK].nx2*(dcmps[Gki_BLOCK].is+k+noffset2_);
          ((Real*)(grf[INN][INN]))[idx] = psi2[INN][idx2];
          ((Real*)(grf[INN][OUT]))[idx] = psi2[OUT][idx2];
        }
      }
    }
    if (Globals::my_rank == 0) std::cout << "inn boundary gkp = " << gkp << std::endl;
  }
  // surface charge at outer boundary 
  gip = Nx1;
  for (gkp=0;gkp<Nx3;++gkp) {
    CalcGrf(gip, gkp);
    // If this proc is responsible for the ip index, store the Green's function
    if ((gkp >= dcmps[Gik_BLOCK].ks)&&(gkp <= dcmps[Gik_BLOCK].ke)) {
      kp = gkp - dcmps[Gik_BLOCK].ks;
      for (int i=0;i<dcmps[Gik_BLOCK].nx1;++i) {
        for (int j=0;j<dcmps[Gik_BLOCK].nx2;++j) {
          idx = j + dcmps[Gik_BLOCK].nx2*(kp + dcmps[Gik_BLOCK].nx3*i);
          idx2 = j + dcmps[Gii_BLOCK].nx2*(dcmps[Gii_BLOCK].is+i+noffset1_);
          ((Real*)(grf[OUT][TOP]))[idx] = psi2[TOP][idx2];
          ((Real*)(grf[OUT][BOT]))[idx] = psi2[BOT][idx2];
        }
      }
    }
    if ((gkp >= dcmps[Gkk_BLOCK].ks)&&(gkp <= dcmps[Gkk_BLOCK].ke)) {
      kp = gkp - dcmps[Gkk_BLOCK].ks;
      for (int k=0;k<dcmps[Gkk_BLOCK].nx1;++k) {
        for (int j=0;j<dcmps[Gkk_BLOCK].nx2;++j) {
          idx = j + dcmps[Gkk_BLOCK].nx2*(k + dcmps[Gkk_BLOCK].nx1*kp);
          idx2 = j + dcmps[Gki_BLOCK].nx2*(dcmps[Gki_BLOCK].is+k+noffset2_);
          ((Real*)(grf[OUT][INN]))[idx] = psi2[INN][idx2];
          ((Real*)(grf[OUT][OUT]))[idx] = psi2[OUT][idx2];
        }
      }
    }
    if (Globals::my_rank == 0) std::cout << "out boundary gkp = " << gkp << std::endl;
  }
  remap_3d(((Real*)(grf[TOP][TOP])), ((Real*)(grf[TOP][TOP])), (Real*)buf_, RmpPlan[Gii_BLOCK][Gii2P0]);
  remap_3d(((Real*)(grf[TOP][BOT])), ((Real*)(grf[TOP][BOT])), (Real*)buf_, RmpPlan[Gii_BLOCK][Gii2P0]);
  remap_3d(((Real*)(grf[BOT][TOP])), ((Real*)(grf[BOT][TOP])), (Real*)buf_, RmpPlan[Gii_BLOCK][Gii2P0]);
  remap_3d(((Real*)(grf[BOT][BOT])), ((Real*)(grf[BOT][BOT])), (Real*)buf_, RmpPlan[Gii_BLOCK][Gii2P0]);
  remap_3d(((Real*)(grf[INN][TOP])), ((Real*)(grf[INN][TOP])), (Real*)buf_, RmpPlan[Gik_BLOCK][Gik2P0]);
  remap_3d(((Real*)(grf[INN][BOT])), ((Real*)(grf[INN][BOT])), (Real*)buf_, RmpPlan[Gik_BLOCK][Gik2P0]);
  remap_3d(((Real*)(grf[OUT][TOP])), ((Real*)(grf[OUT][TOP])), (Real*)buf_, RmpPlan[Gik_BLOCK][Gik2P0]);
  remap_3d(((Real*)(grf[OUT][BOT])), ((Real*)(grf[OUT][BOT])), (Real*)buf_, RmpPlan[Gik_BLOCK][Gik2P0]);
  remap_3d(((Real*)(grf[TOP][INN])), ((Real*)(grf[TOP][INN])), (Real*)buf_, RmpPlan[Gki_BLOCK][Gki2P0]);
  remap_3d(((Real*)(grf[TOP][OUT])), ((Real*)(grf[TOP][OUT])), (Real*)buf_, RmpPlan[Gki_BLOCK][Gki2P0]);
  remap_3d(((Real*)(grf[BOT][INN])), ((Real*)(grf[BOT][INN])), (Real*)buf_, RmpPlan[Gki_BLOCK][Gki2P0]);
  remap_3d(((Real*)(grf[BOT][OUT])), ((Real*)(grf[BOT][OUT])), (Real*)buf_, RmpPlan[Gki_BLOCK][Gki2P0]);
  remap_3d(((Real*)(grf[INN][INN])), ((Real*)(grf[INN][INN])), (Real*)buf_, RmpPlan[Gkk_BLOCK][Gkk2P0]);
  remap_3d(((Real*)(grf[INN][OUT])), ((Real*)(grf[INN][OUT])), (Real*)buf_, RmpPlan[Gkk_BLOCK][Gkk2P0]);
  remap_3d(((Real*)(grf[OUT][INN])), ((Real*)(grf[OUT][INN])), (Real*)buf_, RmpPlan[Gkk_BLOCK][Gkk2P0]);
  remap_3d(((Real*)(grf[OUT][OUT])), ((Real*)(grf[OUT][OUT])), (Real*)buf_, RmpPlan[Gkk_BLOCK][Gkk2P0]);
  for (int i=2;i<=17;++i)
    fftw_execute(fft_x2_forward_[i]);
  remap_3d((Real*)grf[TOP][TOP], (Real*)grf[TOP][TOP], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[TOP][BOT], (Real*)grf[TOP][BOT], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[BOT][TOP], (Real*)grf[BOT][TOP], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[BOT][BOT], (Real*)grf[BOT][BOT], (Real*)buf_, RmpPlan[Gii2P][Gii]);
  remap_3d((Real*)grf[INN][TOP], (Real*)grf[INN][TOP], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[INN][BOT], (Real*)grf[INN][BOT], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[OUT][TOP], (Real*)grf[OUT][TOP], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[OUT][BOT], (Real*)grf[OUT][BOT], (Real*)buf_, RmpPlan[Gik2P][Gik]);
  remap_3d((Real*)grf[TOP][INN], (Real*)grf[TOP][INN], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[TOP][OUT], (Real*)grf[TOP][OUT], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[BOT][INN], (Real*)grf[BOT][INN], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[BOT][OUT], (Real*)grf[BOT][OUT], (Real*)buf_, RmpPlan[Gki2P][Gki]);
  remap_3d((Real*)grf[INN][INN], (Real*)grf[INN][INN], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
  remap_3d((Real*)grf[INN][OUT], (Real*)grf[INN][OUT], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
  remap_3d((Real*)grf[OUT][INN], (Real*)grf[OUT][INN], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
  remap_3d((Real*)grf[OUT][OUT], (Real*)grf[OUT][OUT], (Real*)buf_, RmpPlan[Gkk2P][Gkk]);
}

void OBCGravityCyl::CalcGrf(int gip, int gkp)
// place charge at (i,j,k) = (gip, 0, gkp) and solve the Poisson eq.
// with open boundary condition in an enlarged domain.
// return in psi2[...]
{
  gip += noffset1_;
  gkp += noffset2_;
  int gjp=0; // position of the point source
  int gi,gj,gk,i,k,idx;
  Real R, Rp, rds, psi, dV;
  
  dV = 0.5*(SQR(x1f2_(NGHOST+gip+1)) - SQR(x1f2_(NGHOST+gip)))*dx2_*dx3_;
  Rp = x1v2_(NGHOST+gip);
  // Add point charge
  for (int i=0;i<dcmps[E2P0].nx1;++i) {
    for (int k=0;k<dcmps[E2P0].nx3;++k) {
      for (int j=0;j<dcmps[E2P0].nx2;++j) {
        idx = j + dcmps[E2P0].nx2*(k + dcmps[E2P0].nx3*i);
        if (((dcmps[E2P0].is+i)==gip)&&((dcmps[E2P0].js+j)==gjp)&&((dcmps[E2P0].ks+k)==gkp))
          ((Real*)in2_)[idx] = 1.0;
        else
          ((Real*)in2_)[idx] = 0.0;
      }
    }
  }
  // Add boundary charge
  for (int k=0;k<dcmps[E2P0].nx3;++k) {
    for (int j=0;j<dcmps[E2P0].nx2;++j) {
      gj = dcmps[E2P0].js+j;
      gk = dcmps[E2P0].ks+k;
      if (dcmps[E2P0].is == 0) {
        i = 0;
        idx = j + dcmps[E2P0].nx2*(k + dcmps[E2P0].nx3*i);
        Real psir = 0;
        Real psil = 0;
        for (int p=0;p<pfold_;++p) {
          R  = x1v2_(NGHOST+i);
          rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)+2*PI*p/pfold_) + SQR(dx3_*(gk-gkp)));
          psir -= (four_pi_G/4.0/PI)*dV/rds;
          R  = x1v2_(NGHOST+i-1);
          rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)+2*PI*p/pfold_) + SQR(dx3_*(gk-gkp)));
          psil -= (four_pi_G/4.0/PI)*dV/rds;
        }
        ((Real*)in2_)[idx] += aa_(i)*(psir-psil)/four_pi_G;  // INN
      }
      if (dcmps[E2P0].ie == lNx1-1) {
        i = dcmps[E2P0].nx1-1;
        idx = j + dcmps[E2P0].nx2*(k + dcmps[E2P0].nx3*i);
        gi = lNx1-1;
        R  = x1v2_(NGHOST+gi+1);
        for (int p=0;p<pfold_;++p) {
          rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)+2*PI*p/pfold_) + SQR(dx3_*(gk-gkp)));
          psi = -(four_pi_G/4.0/PI)*dV/rds;
          ((Real*)in2_)[idx] -= cc_(gi)*psi/four_pi_G;  // OUT
        }
      }
    }
  }
  for (int i=0;i<dcmps[E2P0].nx1;++i) {
    for (int j=0;j<dcmps[E2P0].nx2;++j) {
      gj = dcmps[E2P0].js+j;
      gi = dcmps[E2P0].is+i;
      R  = x1v2_(NGHOST+gi);
      if (dcmps[E2P0].ke == Nx3-1+2*ngh_grf_) {
        k = dcmps[E2P0].nx3-1;
        idx = j + dcmps[E2P0].nx2*(k + dcmps[E2P0].nx3*i); // TOP
        gk = Nx3-1+2*ngh_grf_;
        for (int p=0;p<pfold_;++p) {
          rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)+2*PI*p/pfold_) + SQR(dx3_*(gk+1-gkp)));
          psi = -(four_pi_G/4.0/PI)*dV/rds;
          ((Real*)in2_)[idx] -= psi/four_pi_G/SQR(dx3_);
        }
      }
      if (dcmps[E2P0].ks == 0) {
        k = 0;
        idx = j + dcmps[E2P0].nx2*(k + dcmps[E2P0].nx3*i); // BOT
        gk = 0;
        for (int p=0;p<pfold_;++p) {
          rds = sqrt(R*R + Rp*Rp - 2*R*Rp*cos(dx2_*(gj-gjp)+2*PI*p/pfold_) + SQR(dx3_*(gk-1-gkp)));
          psi = -(four_pi_G/4.0/PI)*dV/rds;
          ((Real*)in2_)[idx] -= psi/four_pi_G/SQR(dx3_);
        }
      }
    }
  }
  fftw_execute(fft_x2_forward_[1]); // ((Real*)in2_) -> in2_
  remap_3d((Real*)in2_, (Real*)in2_, (Real*)buf_, RmpPlan[E2P][E3P]);
  fftw_execute(fft_x3_r2r_[4]);
  fftw_execute(fft_x3_r2r_[5]);
  remap_3d((Real*)in2_, (Real*)in2_, (Real*)buf_, RmpPlan[E3P][E1P]);
  for (int k=0;k<dcmps[E1P].nx3;++k) {
    for (int j=0;j<dcmps[E1P].nx2;++j) {
      for (int i=0;i<dcmps[E1P].nx1;++i) {
        int idx = i + dcmps[E1P].nx1*(j + dcmps[E1P].nx2*k);
        rr_(i) = four_pi_G * in2_[idx][0];
        if (rat==1.0) {
          bb_(i) = lambda22_(j,i) + lambda33_(k) - 2./SQR( dx1v2_(i+NGHOST) );
        }
        else {
          bb_(i) = lambda22_(j,i) + lambda33_(k) - 2.*SQR( 1.0 / std::log(rat) / x1v2_(i+NGHOST) );
        }
      }
      bb_(0) += aa_(0);
      tridag(aa_,bb_,cc_,xx_,rr_);
      for (int i=0;i<dcmps[E1P].nx1;++i) {
        int idx = i + dcmps[E1P].nx1*(j + dcmps[E1P].nx2*k);
        out2_[idx][0] = xx_(i);
        rr_(i) = four_pi_G * in2_[idx][1];
      }
      tridag(aa_,bb_,cc_,xx_,rr_);
      for (int i=0;i<dcmps[E1P].nx1;++i) {
        int idx = i + dcmps[E1P].nx1*(j + dcmps[E1P].nx2*k);
        out2_[idx][1] = xx_(i);
      }
    }
  }
  remap_3d((Real*)out2_, (Real*)out2_, (Real*)buf_, RmpPlan[E1P][E3P]);
  fftw_execute(fft_x3_r2r_[6]);
  fftw_execute(fft_x3_r2r_[7]);
  remap_3d((Real*)out2_, (Real*)out2_, (Real*)buf_, RmpPlan[E3P][E2P]);
  fftw_execute(fft_x2_backward_[1]);
  remap_3d(((Real*)out2_), ((Real*)out2_), (Real*)buf_, RmpPlan[E2P0][EB]);
  Real normfac = 1.0 / (Nx2*2*(Nx3+1+2*ngh_grf_));

  for (int j=0;j<dcmps[EB].nx2;++j) {
    for (int i=0;i<lNx1;++i) {
      int idx2 = j + dcmps[EB].nx2*i;
      psi2[TOP][idx2] = 0.0;
      psi2[BOT][idx2] = 0.0;
    }
    for (int k=0;k<Nx3+2*ngh_grf_;++k) {
      int idx2 = j + dcmps[EB].nx2*gk;
      psi2[INN][idx2] = 0.0;
      psi2[OUT][idx2] = 0.0;
    }
  }
  for (int k=0;k<dcmps[EB].nx3;++k) {
    for (int j=0;j<dcmps[EB].nx2;++j) {
      for (int i=0;i<dcmps[EB].nx1;++i) {
        int idx = i + dcmps[EB].nx1*(j + dcmps[EB].nx2*k);
        gk = dcmps[EB].ks + k;
        gi = dcmps[EB].is + i;
        ((Real*)out2_)[idx] *= normfac;
        if (gk == Nx3+noffset2_) { // top boundary
          int idx2 = j + dcmps[EB].nx2*gi;
          psi2[TOP][idx2] = ((Real*)out2_)[idx];
        }
        if (gk == -1 + noffset2_) { // bottom boundary
          int idx2 = j + dcmps[EB].nx2*gi;
          psi2[BOT][idx2] = ((Real*)out2_)[idx];
        }
        if (gi == -1 + noffset1_) { // inner boundary
          int idx2 = j + dcmps[EB].nx2*gk;
          psi2[INN][idx2] = ((Real*)out2_)[idx];
        }
        if (gi == Nx1 + noffset1_) { // outer boundary
          int idx2 = j + dcmps[EB].nx2*gk;
          psi2[OUT][idx2] = ((Real*)out2_)[idx];
        }
      }
    }
  }
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, psi2[TOP],
    dcmps[EB].nx2*dcmps[EB].nx1, MPI_DOUBLE, x1comm);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, psi2[BOT],
    dcmps[EB].nx2*dcmps[EB].nx1, MPI_DOUBLE, x1comm);

  int recvcount[x3comm_size], displs[x3comm_size];
  for (int i=0;i<x3comm_size;++i) {
    recvcount[i] = dcmps[EB].nx2*nx3;
    displs[i] = 0;
  }
  recvcount[0] += dcmps[EB].nx2*ngh_grf_;
  recvcount[x3comm_size-1] += dcmps[EB].nx2*ngh_grf_;
  for (int i=1;i<x3comm_size;++i) {
    displs[i] = displs[i-1] + recvcount[i-1];
  }
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, psi2[INN],
    recvcount, displs, MPI_DOUBLE, x3comm);
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, psi2[OUT],
    recvcount, displs, MPI_DOUBLE, x3comm);

  // WARNING : Note that in the vertical direction, the processor responsible for the top boundary of the extended domain should be also responsible for the top boundary for the original domain.
  // However, in the radial direction, the processor responsible for the inner boundary of the extended domain would in general be different from the processor responsible for the inner boundary of the original domain.
  MPI_Bcast(psi2[TOP], dcmps[EB].nx2*lNx1, MPI_DOUBLE, x3comm_size-1, x3comm); 
  MPI_Bcast(psi2[BOT], dcmps[EB].nx2*lNx1, MPI_DOUBLE, 0, x3comm);
  MPI_Bcast(psi2[INN], dcmps[EB].nx2*(Nx3+2*ngh_grf_), MPI_DOUBLE, (-1+noffset1_)/(lNx1/np1), x1comm);
  MPI_Bcast(psi2[OUT], dcmps[EB].nx2*(Nx3+2*ngh_grf_), MPI_DOUBLE, (Nx1+noffset1_)/(lNx1/np1), x1comm);
}


void OBCGravityCyl::LoadSource(const AthenaArray<Real> &src)
{
  int is, ie, js, je, ks, ke;
  int idx;
  is = pmy_block->is; js = pmy_block->js; ks = pmy_block->ks;
  ie = pmy_block->ie; je = pmy_block->je; ke = pmy_block->ke;
  for (int k=ks;k<=ke;++k) {
    for (int j=js;j<=je;++j) {
      for (int i=is;i<=ie;++i) {
        idx = (i-is) + dcmps[XB].nx1*((j-js) + dcmps[XB].nx2*(k-ks));
        ((Real*)in_)[idx] = src(IDN,k,j,i);
      }
    }
  }
}

void OBCGravityCyl::SolveZeroBC()
{
  remap_3d(((Real*)in_), ((Real*)in_), (Real*)buf_, RmpPlan[XB][X2P0]);
  fftw_execute(fft_x2_forward_[0]); // ((Real*)in_) -> in_
  remap_3d((Real*)in_, (Real*)in_, (Real*)buf_, RmpPlan[X2P][X3P]);
  fftw_execute(fft_x3_r2r_[0]);
  fftw_execute(fft_x3_r2r_[1]);
  remap_3d((Real*)in_, (Real*)in_, (Real*)buf_, RmpPlan[X3P][X1P]);
  int idx;
  for (int k=0;k<dcmps[X1P].nx3;++k) {
    for (int j=0;j<dcmps[X1P].nx2;++j) {
      for (int i=0;i<dcmps[X1P].nx1;++i) {
        idx = i + dcmps[X1P].nx1*(j + dcmps[X1P].nx2*k);
        r_(i) = four_pi_G * in_[idx][0];
        if (rat==1.0) {
          b_(i) = lambda2_(j,i) + lambda3_(k) - 2./SQR( dx1v_(i+NGHOST) );
        }
        else {
          b_(i) = lambda2_(j,i) + lambda3_(k) - 2.*SQR( 1.0 / std::log(rat) / x1v_(i+NGHOST) );
        }
      }
      tridag(a_,b_,c_,x_,r_);
      for (int i=0;i<dcmps[X1P].nx1;++i) {
        idx = i + dcmps[X1P].nx1*(j + dcmps[X1P].nx2*k);
        out_[idx][0] = x_(i);
        r_(i) = four_pi_G * in_[idx][1];
      }
      tridag(a_,b_,c_,x_,r_);
      for (int i=0;i<dcmps[X1P].nx1;++i) {
        idx = i + dcmps[X1P].nx1*(j + dcmps[X1P].nx2*k);
        out_[idx][1] = x_(i);
      }
    }
  }
  remap_3d((Real*)out_, (Real*)out_, (Real*)buf_, RmpPlan[X1P][X3P]);
  fftw_execute(fft_x3_r2r_[2]);
  fftw_execute(fft_x3_r2r_[3]);
  remap_3d((Real*)out_, (Real*)out_, (Real*)buf_, RmpPlan[X3P][X2P]);
  fftw_execute(fft_x2_backward_[0]);
  remap_3d(((Real*)out_), ((Real*)out_), (Real*)buf_, RmpPlan[X2P0][XB]);
  int is,ie,js,je,ks,ke;
  is = pmy_block->is; js = pmy_block->js; ks = pmy_block->ks;
  ie = pmy_block->ie; je = pmy_block->je; ke = pmy_block->ke;
  Real normfac = 1.0 / (Nx2*2*((Nx3)+1));
  for (int k=ks;k<=ke;++k) {
    for (int j=js;j<=je;++j) {
      for (int i=is;i<=ie;++i) {
        idx = (i-is) + dcmps[XB].nx1*((j-js) + dcmps[XB].nx2*(k-ks));
        ((Real*)out_)[idx] *= normfac;
      }
    }
  }
}

void OBCGravityCyl::CalcBndCharge()
{
  Coordinates *pcrd = pmy_block->pcoord;
  RegionSize& mesh_size = pmy_block->pmy_mesh->mesh_size;
  RegionSize& block_size = pmy_block->block_size;
  Real normfac1 = 1.0 / (four_pi_G*dx3_*dx3_);
  Real normfac2, normfac3;
  if (mesh_size.x1rat == 1.0) {
    normfac2 = (1.0 / SQR(dx1v_(-1+NGHOST)) + 0.5/dx1v_(-1+NGHOST)/x1v_(-1+NGHOST)) / four_pi_G;
    normfac3 = (1.0 / SQR(dx1v_(Nx1+NGHOST)) - 0.5/dx1v_(Nx1+NGHOST)/x1v_(Nx1+NGHOST)) / four_pi_G;
  }
  else {
    normfac2 = SQR( 1.0 / std::log(rat) / x1v_(-1+NGHOST)  ) / four_pi_G;
    normfac3 = SQR( 1.0 / std::log(rat) / x1v_(Nx1+NGHOST) ) / four_pi_G;
  }
  if (dcmps[XB].ke == Nx3-1) {
    for (int j=0;j<dcmps[XB].nx2;++j) {
      for (int i=0;i<dcmps[XB].nx1;++i) {
        int idx2 = i + dcmps[XB].nx1*j;
        int idx = i + dcmps[XB].nx1*(j + dcmps[XB].nx2*(dcmps[XB].nx3-1));
        sigma[TOP][idx2] = ((Real*)out_)[idx]*normfac1;
      }
    }
  }
  if (dcmps[XB].ks == 0) {
    for (int j=0;j<dcmps[XB].nx2;++j) {
      for (int i=0;i<dcmps[XB].nx1;++i) {
        int idx2 = i + dcmps[XB].nx1*j;
        int idx = i + dcmps[XB].nx1*(j + dcmps[XB].nx2*(0));
        sigma[BOT][idx2] = ((Real*)out_)[idx]*normfac1;
      }
    }
  }
  if (dcmps[XB].is == 0) {
    for (int k=0;k<dcmps[XB].nx3;++k) {
      for (int j=0;j<dcmps[XB].nx2;++j) {
        int idx2 = j + dcmps[XB].nx2*k;
        int idx = 0 + dcmps[XB].nx1*(j + dcmps[XB].nx2*k);
        sigma[INN][idx2] = ((Real*)out_)[idx]*normfac2;
      }
    }
  }
  if (dcmps[XB].ie == Nx1-1) {
    for (int k=0;k<dcmps[XB].nx3;++k) {
      for (int j=0;j<dcmps[XB].nx2;++j) {
        int idx2 = j + dcmps[XB].nx2*k;
        int idx = dcmps[XB].nx1-1 + dcmps[XB].nx1*(j + dcmps[XB].nx2*k);
        sigma[OUT][idx2] = ((Real*)out_)[idx]*normfac3;
      }
    }
  }
}

void OBCGravityCyl::CalcBndPot()
{
  MPI_Request reqv[4], reqr[4];
  int rootv[] = {0, 0, 0, x1comm_size-1};
  int countv[] = {2*dcmps[Gii].nx2*dcmps[Gii].nx3,
                  2*dcmps[Gii].nx2*dcmps[Gii].nx3,
                  2*dcmps[Gik].nx2*dcmps[Gik].nx3,
                  2*dcmps[Gik].nx2*dcmps[Gik].nx3};
  int rootr[] = {x3comm_size-1, 0, x3comm_size-1, x3comm_size-1};
  int countr[] = {2*dcmps[Gki].nx2*dcmps[Gki].nx3,
                  2*dcmps[Gki].nx2*dcmps[Gki].nx3,
                  2*dcmps[Gkk].nx2*dcmps[Gkk].nx3,
                  2*dcmps[Gkk].nx2*dcmps[Gkk].nx3};
  for (int i=TOP;i<=OUT;++i) {
    remap_2d(sigma[i], sigma[i], (Real*)buf_, BndryRmpPlan[i][BLOCK][FFT_LONG]);
    fftw_execute(fft_2d_plan[i][0]);
    remap_2d((Real*)sigma_fft[i], (Real*)sigma_fft_v[i], (Real*)buf_, BndryRmpPlan[i][FFT_SHORT][SIGv]);
    MPI_Ibcast((Real*)sigma_fft_v[i], countv[i], MPI_DOUBLE, rootv[i], x1comm, &reqv[i]);
    remap_2d((Real*)sigma_fft[i], (Real*)sigma_fft_r[i], (Real*)buf_, BndryRmpPlan[i][FFT_SHORT][SIGr]);
    MPI_Ibcast((Real*)sigma_fft_r[i], countr[i], MPI_DOUBLE, rootr[i], x3comm, &reqr[i]);
  }
  
  // Initialize
  for (int dst=TOP;dst<=BOT;++dst) {
    for (int j=0;j<dcmps[Gii].nx2;++j) {
      for (int i=0;i<dcmps[Gii].nx1;++i) {
        int idx = i + dcmps[Gii].nx1*j;
        psi_fft[dst][idx][0] = 0;
        psi_fft[dst][idx][1] = 0;
      }
    }
  }
  for (int dst=INN;dst<=OUT;++dst) {
    for (int k=0;k<dcmps[Gki].nx1;++k) {
      for (int j=0;j<dcmps[Gki].nx2;++j) {
        int idx = j + dcmps[Gki].nx2*k;
        psi_fft[dst][idx][0] = 0;
        psi_fft[dst][idx][1] = 0;
      }
    }
  }
  // Calculate top/bottom boundary potential
  MPI_Waitall(4, reqv, MPI_STATUSES_IGNORE);
  for (int dst=TOP;dst<=BOT;++dst) {
    for (int src=TOP;src<=BOT;++src) {
      for (int j=0;j<dcmps[Gii].nx2;++j) {
        for (int i=0;i<dcmps[Gii].nx1;++i) {
          int idx3 = i + dcmps[Gii].nx1*j;
          for (int ip=0;ip<dcmps[Gii].nx3;++ip) {
            int idx = ip + dcmps[Gii].nx3*(i + dcmps[Gii].nx1*j);
            int idx2 = ip + dcmps[Gii].nx3*j;
            psi_fft[dst][idx3][0] += grf[src][dst][idx][0]*sigma_fft_v[src][idx2][0]
                                   - grf[src][dst][idx][1]*sigma_fft_v[src][idx2][1];
            psi_fft[dst][idx3][1] += grf[src][dst][idx][0]*sigma_fft_v[src][idx2][1]
                                   + grf[src][dst][idx][1]*sigma_fft_v[src][idx2][0];
          }
        }
      }
    }
    for (int src=INN;src<=OUT;++src) {
      for (int j=0;j<dcmps[Gik].nx2;++j) {
        for (int i=0;i<dcmps[Gik].nx1;++i) {
          int idx3 = i + dcmps[Gik].nx1*j;
          for (int kp=0;kp<dcmps[Gik].nx3;++kp) {
            int idx = kp + dcmps[Gik].nx3*(i + dcmps[Gik].nx1*j);
            int idx2 = kp + dcmps[Gik].nx3*j;
            psi_fft[dst][idx3][0] += grf[src][dst][idx][0]*sigma_fft_v[src][idx2][0]
                                   - grf[src][dst][idx][1]*sigma_fft_v[src][idx2][1];
            psi_fft[dst][idx3][1] += grf[src][dst][idx][0]*sigma_fft_v[src][idx2][1]
                                   + grf[src][dst][idx][1]*sigma_fft_v[src][idx2][0];
          }
        }
      }
    }
  }
  if (x3rank == x3comm_size-1) 
    MPI_Ireduce(MPI_IN_PLACE, (Real*)psi_fft[TOP],
     2*dcmps[Gii].nx2*dcmps[Gii].nx1, MPI_DOUBLE, MPI_SUM, x3comm_size-1,
     x3comm, &reqv[TOP]);
  else
    MPI_Ireduce((Real*)psi_fft[TOP], (Real*)psi_fft[TOP],
     2*dcmps[Gii].nx2*dcmps[Gii].nx1, MPI_DOUBLE, MPI_SUM, x3comm_size-1,
     x3comm, &reqv[TOP]);
  if (x3rank == 0) 
    MPI_Ireduce(MPI_IN_PLACE, (Real*)psi_fft[BOT],
     2*dcmps[Gii].nx2*dcmps[Gii].nx1, MPI_DOUBLE, MPI_SUM, 0,
     x3comm, &reqv[BOT]);
  else
    MPI_Ireduce((Real*)psi_fft[BOT], (Real*)psi_fft[BOT],
     2*dcmps[Gii].nx2*dcmps[Gii].nx1, MPI_DOUBLE, MPI_SUM, 0,
     x3comm, &reqv[BOT]);

  // Calculate inner/outer boundary potential
  MPI_Waitall(4, reqr, MPI_STATUSES_IGNORE);
  for (int dst=INN;dst<=OUT;++dst) {
    for (int src=TOP;src<=BOT;++src) {
      for (int k=0;k<dcmps[Gki].nx1;++k) {
        for (int j=0;j<dcmps[Gki].nx2;++j) {
          int idx3 = j + dcmps[Gki].nx2*k;
          for (int ip=0;ip<dcmps[Gki].nx3;++ip) {
            int idx = ip + dcmps[Gki].nx3*(j + dcmps[Gki].nx2*k);
            int idx2 = ip + dcmps[Gki].nx3*j;
            psi_fft[dst][idx3][0] += grf[src][dst][idx][0]*sigma_fft_r[src][idx2][0]
                                   - grf[src][dst][idx][1]*sigma_fft_r[src][idx2][1];
            psi_fft[dst][idx3][1] += grf[src][dst][idx][0]*sigma_fft_r[src][idx2][1]
                                   + grf[src][dst][idx][1]*sigma_fft_r[src][idx2][0];
          }
        }
      }
    }
    for (int src=INN;src<=OUT;++src) {
      for (int k=0;k<dcmps[Gkk].nx1;++k) {
        for (int j=0;j<dcmps[Gkk].nx2;++j) {
          int idx3 = j + dcmps[Gkk].nx2*k;
          for (int kp=0;kp<dcmps[Gkk].nx3;++kp) {
            int idx = kp + dcmps[Gkk].nx3*(j + dcmps[Gkk].nx2*k);
            int idx2 = kp + dcmps[Gkk].nx3*j;
            psi_fft[dst][idx3][0] += grf[src][dst][idx][0]*sigma_fft_r[src][idx2][0]
                                   - grf[src][dst][idx][1]*sigma_fft_r[src][idx2][1];
            psi_fft[dst][idx3][1] += grf[src][dst][idx][0]*sigma_fft_r[src][idx2][1]
                                   + grf[src][dst][idx][1]*sigma_fft_r[src][idx2][0];
          }
        }
      }
    }
  }
  if (x1rank == 0) 
    MPI_Ireduce(MPI_IN_PLACE, (Real*)psi_fft[INN],
    2*dcmps[Gki].nx2*dcmps[Gki].nx1, MPI_DOUBLE, MPI_SUM, 0,
    x1comm, &reqv[INN]);
  else 
    MPI_Ireduce((Real*)psi_fft[INN], (Real*)psi_fft[INN],
    2*dcmps[Gki].nx2*dcmps[Gki].nx1, MPI_DOUBLE, MPI_SUM, 0,
    x1comm, &reqv[INN]);
  if (x1rank == x1comm_size-1) 
    MPI_Ireduce(MPI_IN_PLACE, (Real*)psi_fft[OUT],
    2*dcmps[Gki].nx2*dcmps[Gki].nx1, MPI_DOUBLE, MPI_SUM, x1comm_size-1,
    x1comm, &reqv[OUT]);
  else 
    MPI_Ireduce((Real*)psi_fft[OUT], (Real*)psi_fft[OUT],
    2*dcmps[Gki].nx2*dcmps[Gki].nx1, MPI_DOUBLE, MPI_SUM, x1comm_size-1,
    x1comm, &reqv[OUT]);

  for (int i=TOP;i<=OUT;++i) {
    MPI_Wait(&reqv[i], MPI_STATUSES_IGNORE);
    remap_2d((Real*)psi_fft[i], (Real*)psi_fft[i], (Real*)buf_, BndryRmpPlan[i][PSI][FFT_SHORT]);
    fftw_execute(fft_2d_plan[i][1]);
    remap_2d(psi[i], psi[i], (Real*)buf_, BndryRmpPlan[i][FFT_LONG][BLOCK]);
  }
  Real normfac1 = 1.0 / (four_pi_G*dx3_*dx3_);
  Real normfac2, normfac3;
  if (rat == 1.0) {
    normfac2 = (1.0 / SQR(dx1v_(0+NGHOST)) - 0.5/dx1v_(0+NGHOST)/x1v_(0+NGHOST)) / four_pi_G;
    normfac3 = (1.0 / SQR(dx1v_(Nx1-1+NGHOST)) + 0.5/dx1v_(Nx1-1+NGHOST)/x1v_(Nx1-1+NGHOST)) / four_pi_G;
  }
  else {
    normfac2 = SQR( 1.0 / std::log(rat) / x1v_(0+NGHOST)     ) / four_pi_G;
    normfac3 = SQR( 1.0 / std::log(rat) / x1v_(Nx1-1+NGHOST) ) / four_pi_G;
  }
  if (dcmps[XB].ke == Nx3-1) {
    for (int j=0;j<dcmps[XB].nx2;++j) {
      for (int i=0;i<dcmps[XB].nx1;++i) {
        int idx2 = i + dcmps[XB].nx1*j;
        int idx = i + dcmps[XB].nx1*(j + dcmps[XB].nx2*(dcmps[XB].nx3-1));
        psi[TOP][idx2] /= Nx2;
        ((Real*)in_)[idx] += psi[TOP][idx2]*normfac1;
      }
    }
  }
  if (dcmps[XB].ks == 0) {
    for (int j=0;j<dcmps[XB].nx2;++j) {
      for (int i=0;i<dcmps[XB].nx1;++i) {
        int idx2 = i + dcmps[XB].nx1*j;
        int idx = i + dcmps[XB].nx1*(j + dcmps[XB].nx2*(0));
        psi[BOT][idx2] /= Nx2;
        ((Real*)in_)[idx] += psi[BOT][idx2]*normfac1;
      }
    }
  }
  if (dcmps[XB].is == 0) {
    for (int k=0;k<dcmps[XB].nx3;++k) {
      for (int j=0;j<dcmps[XB].nx2;++j) {
        int idx2 = j + dcmps[XB].nx2*k;
        int idx = 0 + dcmps[XB].nx1*(j + dcmps[XB].nx2*k);
        psi[INN][idx2] /= Nx2;
        ((Real*)in_)[idx] += psi[INN][idx2]*normfac2;
      }
    }
  }
  if (dcmps[XB].ie == Nx1-1) {
    for (int k=0;k<dcmps[XB].nx3;++k) {
      for (int j=0;j<dcmps[XB].nx2;++j) {
        int idx2 = j + dcmps[XB].nx2*k;
        int idx = dcmps[XB].nx1-1 + dcmps[XB].nx1*(j + dcmps[XB].nx2*k);
        psi[OUT][idx2] /= Nx2;
        ((Real*)in_)[idx] += psi[OUT][idx2]*normfac3;
      }
    }
  }
}

void OBCGravityCyl::RetrieveResult(AthenaArray<Real> &dst)
{
  int is, ie, js, je, ks, ke;
  int idx;
  is = pmy_block->is; js = pmy_block->js; ks = pmy_block->ks;
  ie = pmy_block->ie; je = pmy_block->je; ke = pmy_block->ke;

  for (int k=ks;k<=ke;++k) {
    for (int j=js;j<=je;++j) {
      for (int i=is;i<=ie;++i) {
        idx = (i-is) + dcmps[XB].nx1*((j-js) + dcmps[XB].nx2*(k-ks));
        dst(k,j,i) = ((Real*)out_)[idx];
      }
    }
  }
  if (dcmps[XB].ke == Nx3-1) {
    for (int j=0;j<dcmps[XB].nx2;++j) {
      for (int i=0;i<dcmps[XB].nx1;++i) {
        int idx2 = i + dcmps[XB].nx1*j;
        dst(ke+1,j+js,i+is) = -psi[TOP][idx2];
      }
    }
  }
  if (dcmps[XB].ks == 0) {
    for (int j=0;j<dcmps[XB].nx2;++j) {
      for (int i=0;i<dcmps[XB].nx1;++i) {
        int idx2 = i + dcmps[XB].nx1*j;
        dst(ks-1,j+js,i+is) = -psi[BOT][idx2];
      }
    }
  }
  if (dcmps[XB].is == 0) {
    for (int k=0;k<dcmps[XB].nx3;++k) {
      for (int j=0;j<dcmps[XB].nx2;++j) {
        int idx2 = j + dcmps[XB].nx2*k;
        dst(k+ks,j+js,is-1) = -psi[INN][idx2];
      }
    }
  }
  if (dcmps[XB].ie == Nx1-1) {
    for (int k=0;k<dcmps[XB].nx3;++k) {
      for (int j=0;j<dcmps[XB].nx2;++j) {
        int idx2 = j + dcmps[XB].nx2*k;
        dst(k+ks,j+js,ie+1) = -psi[OUT][idx2];
      }
    }
  }
}
