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

Real cube(Real x, Real y, Real z, Real x0, Real y0, Real z0, Real rad, Real gconst) {
  Real Phi = 0;
  Real r, at1, at2, at3, tmp;
  Real xa[2], ya[2], za[2];
  xa[0] = rad + (x-x0);
  xa[1] = rad - (x-x0);
  ya[0] = rad + (y-y0);
  ya[1] = rad - (y-y0);
  za[0] = rad + (z-z0);
  za[1] = rad - (z-z0);
  for (int i=0;i<=1;++i) {
    for (int j=0;j<=1;++j) {
      for (int k=0;k<=1;++k) {
        r = std::sqrt(xa[i]*xa[i]+ya[j]*ya[j]+za[k]*za[k]);
        at1 = 0.5*xa[i]*xa[i]*std::atan(ya[j]*za[k]/xa[i]/r);
        at2 = 0.5*ya[j]*ya[j]*std::atan(za[k]*xa[i]/ya[j]/r);
        at3 = 0.5*za[k]*za[k]*std::atan(xa[i]*ya[j]/za[k]/r);
        tmp = xa[i]*ya[j]*std::atanh(za[k]/r) + ya[j]*za[k]*std::atanh(xa[i]/r)
          + za[k]*xa[i]*std::atanh(ya[j]/r) - at1 - at2 - at3;
        Phi -= tmp;
      }
    }
  }
  Phi *= gconst;
  return Phi;
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  Real four_pi_G = pin->GetReal("problem","four_pi_G");
  Real eps = pin->GetOrAddReal("problem","grav_eps", 0.0);
  SetFourPiG(four_pi_G);
  SetGravityThreshold(eps);
  SetMeanDensity(0.0);
}

void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  Real r2;
  Real rho0 = pin->GetReal("problem","rho0");
  Real rad  = pin->GetReal("problem","rad");
  Real x0   = pin->GetReal("problem","x0");
  Real y0   = pin->GetReal("problem","y0");
  Real z0   = pin->GetReal("problem","z0");
  int iprob = pin->GetInteger("problem","iprob");
  Real four_pi_G = pin->GetReal("problem","four_pi_G");
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (iprob==1) {
          /* uniform sphere */
          r2 = SQR(pcoord->x1v(i) - x0)
               + SQR(pcoord->x2v(j) - y0)
               + SQR(pcoord->x3v(k) - z0);
          if (r2 < rad*rad) {
            phydro->u(IDN,k,j,i) = rho0;
            phydro->u(IM1,k,j,i) = -0.5*(four_pi_G)*rho0*(rad*rad - r2/3.);
          }
          else {
            phydro->u(IDN,k,j,i) = 0.0;
            phydro->u(IM1,k,j,i) = -(four_pi_G)*rad*rad*rad*rho0/3./sqrt(r2);
          }
        }
        else if (iprob==2) {
          /* cube */
          Real x = pcoord->x1v(i);
          Real y = pcoord->x2v(j);
          Real z = pcoord->x3v(k);
          if ((x > x0-rad)&&(x < x0+rad)&&(y > y0-rad)&&(y < y0+rad)&&(z > z0-rad)&&(z < z0+rad)) {
            phydro->u(IDN,k,j,i) = rho0;
          }
          else {
            phydro->u(IDN,k,j,i) = 0.0;
          }
          phydro->u(IM1,k,j,i) = cube(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k),
              x0, y0, z0, rad, four_pi_G/(4*PI));
        }
      }
    }
  }
  return;
}

//void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//{
////  AllocateUserOutputVariables(2);
//  return;
//}

void Mesh::UserWorkAfterLoop(ParameterInput *pin)
{
  MeshBlock *pmb=pblock;
  int is=pblock->is, ie=pblock->ie;
  int js=pblock->js, je=pblock->je;
  int ks=pblock->ks, ke=pblock->ke;
  int cnt = (ke-ks+1)*(je-js+1)*(ie-is+1);

  /* accuracy test */
  std::cout << std::setprecision(15) << std::scientific;
  Real rerr=0, maxerr=0, avgerr=0;
  Hydro *phydro = pmb->phydro;
  Gravity *pgrav = pmb->pgrav;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        rerr = std::abs((pgrav->phi(k,j,i) - phydro->u(IM1,k,j,i))/phydro->u(IM1,k,j,i));
        avgerr += rerr;
        maxerr = std::max(rerr,maxerr);
      }
    }
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&avgerr,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&maxerr,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  avgerr = avgerr/(static_cast<Real>(cnt*nbtotal));

  if (Globals::my_rank == 0) {
    std::cout << "=====================================================" << std::endl;
    std::cout << " mean relative error : " << avgerr << std::endl;
    std::cout << " maximum relative error: " << maxerr << std::endl;
    std::cout << "=====================================================" << std::endl;
  }

  /* timing test */
  int ncycle = pin->GetInteger("problem", "ncycle");
  if (Globals::my_rank == 0) {
    std::cout << "time resolution is " << MPI_Wtick() << " second" << std::endl;
  }
  Real time_per_solve, time_solve, tavg, tmin, tmax;
  OBCGravityCar *pog = pogrd->pmy_og_car;
  if(pmb!=NULL) {
    pog->LoadSource(pmb->phydro->u);
  }

  /* ================ */
  /* time SolveZeroBC */
  /* ================ */
  time_solve = 0;
  MPI_Barrier(MPI_COMM_WORLD); /* synchronize all processes */
  for (int iter=0; iter<ncycle; ++iter) {
    time_per_solve = MPI_Wtime(); /* get time just before work section */
    //pmgrd->Solve(1);
    pog->SolveZeroBC();
    time_per_solve = MPI_Wtime() - time_per_solve; /* get time just after work section */
    time_solve += time_per_solve;
  }
  time_solve /= ncycle;
  MPI_Reduce(&time_solve, &tavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  tavg /= Globals::nranks;
  if (Globals::my_rank == 0) {
    std::cout << "First solve = MIN: " << tmin << " AVG: " << tavg << " MAX: " << tmax << std::endl;
  }

  /* ================ */
  /* time CalcBndPot */
  /* ================ */
  time_solve = 0;
  MPI_Barrier(MPI_COMM_WORLD); /* synchronize all processes */
  for (int iter=0; iter<ncycle; ++iter) {
    time_per_solve = MPI_Wtime(); /* get time just before work section */
    pog->CalcBndPot();
    time_per_solve = MPI_Wtime() - time_per_solve; /* get time just after work section */
    time_solve += time_per_solve;
  }
  time_solve /= ncycle;
  MPI_Reduce(&time_solve, &tavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  tavg /= Globals::nranks;
  if (Globals::my_rank == 0) {
    std::cout << "Boundary correction = MIN: " << tmin << " AVG: " << tavg << " MAX: " << tmax << std::endl;
  }

  /* ================ */
  /* time Total Solve */
  /* ================ */
  time_solve = 0;
  MPI_Barrier(MPI_COMM_WORLD); /* synchronize all processes */
  for (int iter=0; iter<ncycle; ++iter) {
    time_per_solve = MPI_Wtime(); /* get time just before work section */
    pogrd->Solve(1);
    time_per_solve = MPI_Wtime() - time_per_solve; /* get time just after work section */
    time_solve += time_per_solve;
  }
  time_solve /= ncycle;
  MPI_Reduce(&time_solve, &tavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  tavg /= Globals::nranks;
  if (Globals::my_rank == 0) {
    std::cout << "Gravity Total = MIN: " << tmin << " AVG: " << tavg << " MAX: " << tmax << std::endl;
  }

  /* ================ */
  /* time MHD */
  /* ================ */
  TaskList *ptlist;
  try {
    ptlist = new TimeIntegratorTaskList(pin, this);
  }
  catch(std::bad_alloc& ba) {
    std::cout << "### FATAL ERROR in main" << std::endl << "memory allocation failed "
              << "in creating task list " << ba.what() << std::endl;
#ifdef MPI_PARALLEL
    MPI_Finalize();
#endif
    return;
  }
  time_solve = 0;
  MPI_Barrier(MPI_COMM_WORLD); /* synchronize all processes */
  for (int iter=0; iter<ncycle; ++iter) {
    time_per_solve = MPI_Wtime(); /* get time just before work section */
    for (int stage=1; stage<=ptlist->nstages; ++stage) {
      ptlist->DoTaskListOneStage(this, stage);
    }
    time_per_solve = MPI_Wtime() - time_per_solve; /* get time just after work section */
    time_solve += time_per_solve;
  }
  time_solve /= ncycle;
  MPI_Reduce(&time_solve, &tavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&time_solve, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  tavg /= Globals::nranks;
  if (Globals::my_rank == 0) {
    std::cout << "MHD = MIN: " << tmin << " AVG: " << tavg << " MAX: " << tmax << std::endl;
  }
  delete ptlist;

  return;
}
