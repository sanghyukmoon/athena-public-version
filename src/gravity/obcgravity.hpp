#ifndef GRAVITY_OBCGRAVITY_HPP_
#define GRAVITY_OBCGRAVITY_HPP_

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_fft.hpp
//  \brief defines FFT class which implements parallel FFT using MPI/OpenMP

// Athena++ classes headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../mesh/meshblock_tree.hpp"
#include "gravity.hpp"

#include <iostream>

#ifdef FFT
#include "fftw3.h"

#ifdef MPI_PARALLEL
#include "mpi.h"
#include "../fft/plimpton/fft_3d.h"
#include "../fft/plimpton/fft_2d.h"
#endif // MPI_PARALLEL
#endif // FFT

class Mesh;
class MeshBlock;
class ParameterInput;
class OBCGravityCar;
class OBCGravityCyl;
class OBCGravityDriver;

enum CylBoundaryFace {TOP=0, BOT=1, INN=2, OUT=3};
enum CylDecompNames {XB=0, X1P=1, X2P=2, X3P=3, X2P0=4, E1P=5, E2P=6, E3P=7,
                   E2P0=8, Gii=9, Gik=10, Gki=11, Gkk=12, EB=13, Gii2P=14,
                   Gik2P=15, Gki2P=16, Gkk2P=17, Gii2P0=18, Gik2P0=19,
                   Gki2P0=20, Gkk2P0=21, Gii_BLOCK=22, Gik_BLOCK=23,
                   Gki_BLOCK=24, Gkk_BLOCK=25};
enum CylBndryDcmp {BLOCK=0, FFT_LONG=1, FFT_SHORT=2, PSI=3, SIGv=4, SIGr=5};

enum CarBoundaryFace {STH=0, NTH=1, WST=2, EST=3, CBOT=4, CTOP=5};
enum CarDecompNames {CXB=0, CX1P=1, CX2P=2, CX3P=3, PB=4, P1P=5, P2P=6, P3P=7,
                   CEB=8, CE1P=9, CE2P=10, CE3P=11};
enum CarBndryDcmp {CBLOCK=0, FFT_FIRST=1, FFT_SECOND=2};
enum James {C=0, S=1};

typedef struct DomainDecomp {
  int is,ie,js,je,ks,ke,nx1,nx2,nx3,block_size;
} DomainDecomp;

//! \class OBCGravityDriver
//  \brief OBC driver

class OBCGravityDriver {
public:
  OBCGravityDriver(Mesh *pm, ParameterInput *pin);
  ~OBCGravityDriver();

  OBCGravityCar *pmy_og_car;
  OBCGravityCyl *pmy_og_cyl;
  void Solve(int stage);

protected:
  Mesh *pmy_mesh_;

private:
  GravitySolverTaskList *gtlist_;
};


//! \class OBCGravityCar
//  \brief 

class OBCGravityCar : public Gravity {
public:
  OBCGravityCar(OBCGravityDriver *pcd, MeshBlock *pmb, ParameterInput *pin);
  ~OBCGravityCar();

  void BndFFTForward(int first_nslow, int first_nfast, int second_nslow, int second_nfast, int B);
  void BndFFTBackward(int first_nslow, int first_nfast, int second_nslow, int second_nfast, int B);
  void FillDscGrf();
  void FillCntGrf();
  void LoadSource(const AthenaArray<Real> &src);
  void SolveZeroBC();
  void CalcBndCharge();
  void CalcBndPot();
  void RetrieveResult(AthenaArray<Real> &dst);

protected:
  int Nx1,Nx2,Nx3,nx1,nx2,nx3;
  int x1rank,x2rank,x3rank,x1comm_size,x2comm_size,x3comm_size;
  int np1,np2,np3;
  int ngh_, ngh_grf_;
  DomainDecomp dcmps[12];
  DomainDecomp bndry_dcmps[6][3];

  OBCGravityDriver *pmy_driver_;
  Real *sigma[6], *sigma_mid[6][2], *sigma_fft[6][2][2], *sigfft[2][2][2], *grf;
  Real *in_, *out_, *buf_, *in2_;
  AthenaArray<Real> lambda1_, lambda2_, lambda3_, lambda11_, lambda22_, lambda33_;
  Real dx1_, dx2_, dx3_;

  fftw_plan fft_plan_r2r_[15];
  fftw_plan fft_2d_plan[6][14];

  struct remap_plan_2d *BndryRmpPlan[6][3][3];
  struct remap_plan_3d *RmpPlan[12][12];
  MPI_Comm bndcomm[6],x1comm,x2comm,x3comm;
private:
};

//! \class OBCGravityCyl
//  \brief 

class OBCGravityCyl : public Gravity {
public:
  OBCGravityCyl(OBCGravityDriver *pcd, MeshBlock *pmb, ParameterInput *pin);
  ~OBCGravityCyl();

  void FillDscGrf();
  void FillCntGrf();
  void CalcGrf(int gip, int gkp);
  void LoadSource(const AthenaArray<Real> &src);
  void SolveZeroBC();
  void CalcBndCharge();
  void CalcBndPot();
  void RetrieveResult(AthenaArray<Real> &dst);

protected:
  int Nx1,Nx2,Nx3,nx1,nx2,nx3,lNx1,lNx3,hNx2,hnx2;
  int x1rank, x3rank, x1comm_size, x3comm_size;
  int np1,np2,np3;
  DomainDecomp dcmps[26];
  DomainDecomp bndry_dcmps[4][6];

  OBCGravityDriver *pmy_driver_;
  Real *psi[4], *psi2[4], *sigma[4];
  fftw_complex *sigma_fft[4], *psi_fft[4], *sigma_fft_v[4], *sigma_fft_r[4];
  fftw_complex *in_, *in2_, *out_, *out2_, *buf_;
  fftw_complex *grf[4][4];
  AthenaArray<Real> a_,b_,c_,x_,r_,lambda2_,lambda3_;
  AthenaArray<Real> aa_,bb_,cc_,xx_,rr_,lambda22_,lambda33_;
  AthenaArray<Real> x1f_, dx1f_, x1v_, dx1v_;
  AthenaArray<Real> x1f2_, dx1f2_, x1v2_, dx1v2_;
  Real rat, dx2_, dx3_;

  fftw_plan fft_x2_forward_[18];
  fftw_plan fft_x2_backward_[2];
  fftw_plan fft_x3_r2r_[8];
  fftw_plan fft_2d_plan[4][2];

  struct remap_plan_2d *BndryRmpPlan[4][6][6];
  struct remap_plan_3d *RmpPlan[26][26];
  MPI_Comm bndcomm[4],x1comm,x3comm;

private:
  int ng_, noffset1_, noffset2_, ngh_grf_, pfold_;
};


#endif // GRAVITY_OBCGRAVITY_HPP_
