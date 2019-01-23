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
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
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

class Hure;
// Hure class  for the analytic potential
typedef struct Fparams {
  double R;
  double phi;
  double z;
  double a1;
  double a2;
  double t1;
  double t2;
  double z1;
  double z2;
  Hure *hr;
} Fparams;

class Hure {
public:
  Hure(double a1, double a2, double t1, double t2, double z1, double z2);
  ~Hure();
  double f(double R, double a, double bet, double zet);
  double g(double R, double a, double bet, double zet);
  double M(double R, double a, double bet, double zet, double alpha);
  double N(double R, double a, double bet, double zet, double alpha);
  double Phi(double R, double phi, double z);
private:
  double a1_,a2_,t1_,t2_,z1_,z2_;
  double four_pi_G_, rho0_;
  gsl_integration_romberg_workspace *rmbrg;
};

Hure::Hure(double a1, double a2, double t1, double t2, double z1, double z2)
{
  a1_ = a1;
  a2_ = a2;
  t1_ = t1;
  t2_ = t2;
  z1_ = z1;
  z2_ = z2;
  four_pi_G_ = 4*M_PI;
  rho0_ = 1.0;
  rmbrg = gsl_integration_romberg_alloc(20);
}

Hure::~Hure()
{
  gsl_integration_romberg_free(rmbrg);
}

double Hure::f(double R, double a, double bet, double zet)
{
  if ((fabs(R-a) < 1e-16)&&(fabs(bet-0.5*M_PI) < 1e-16)) {
    return 0;
  }
  else {
    double sinb = sin(bet);
    return -R*sin(2.*bet)*asinh(zet/sqrt((a+R)*(a+R)-4.*a*R*sinb*sinb));
  }
}

double Hure::g(double R, double a, double bet, double zet)
{
  double sin2b = sin(2.*bet);
  return -R*sin2b*asinh((a+R*cos(2.*bet))/sqrt(zet*zet+R*R*sin2b*sin2b));
}

double Hure::M(double R, double a, double bet, double zet, double alpha)
{
  double k = 2.*sqrt(a*R)/sqrt((a+R)*(a+R)+zet*zet);
  if (fabs(k - 1.) < 1e-16) {
    return 0.5*alpha*f(R,a,bet,zet);
  }
  else {
    double ell1 = gsl_sf_ellint_F(bet,k,GSL_PREC_SINGLE);
    return 0.5*zet*sqrt(a/R)*k*ell1 + 0.5*alpha*f(R,a,bet,zet);
  }
}

double Hure::N(double R, double a, double bet, double zet, double alpha)
{
  double k = 2.*sqrt(a*R)/sqrt((a+R)*(a+R)+zet*zet);
  if (fabs(k - 1.) < 1e-16) {
    double nc = floor(bet/M_PI + 0.5);
    double bet_red = bet - nc * M_PI;

    return sqrt(a*R)*(2*nc + sin(bet_red))  +  0.5*(1.-alpha)*g(R,a,bet,zet);
//    double ell2 = gsl_sf_ellint_E(bet,k,GSL_PREC_SINGLE);
//    return 0.5*sqrt(a/R)*k*(2.*R/k/k*ell2) +  0.5*(1.-alpha)*g(R,a,bet,zet);
  }
  else {
    double ell1 = gsl_sf_ellint_F(bet,k,GSL_PREC_SINGLE);
    double ell2 = gsl_sf_ellint_E(bet,k,GSL_PREC_SINGLE);
    return 0.5*sqrt(a/R)*k*((a+R)*ell1 - 2.*R/k/k*(ell1-ell2))
           + 0.5*(1.-alpha)*g(R,a,bet,zet);
  }
}

double dM(double x, void *p) {
  Fparams *params = (Fparams *)p;
  double bet1 = 0.5*(M_PI + params->phi - params->t1);
  double bet2 = 0.5*(M_PI + params->phi - params->t2);
  double zet1 = params->z - params->z1;
  double zet2 = params->z - params->z2;
  return params->hr->M(params->R, x, bet1, zet1, 0) + params->hr->M(params->R, x, bet2, zet2, 0)
       - params->hr->M(params->R, x, bet2, zet1, 0) - params->hr->M(params->R, x, bet1, zet2, 0);
}

double dN(double x, void *p) {
  Fparams *params = (Fparams *)p;
  double bet1 = 0.5*(M_PI + params->phi - params->t1);
  double bet2 = 0.5*(M_PI + params->phi - params->t2);
  double zet = params->z - x;
  return params->hr->N(params->R, params->a2, bet1, zet, 0) + params->hr->N(params->R, params->a1, bet2, zet, 0)
       - params->hr->N(params->R, params->a2, bet2, zet, 0) - params->hr->N(params->R, params->a1, bet1, zet, 0);
}

double Hure::Phi(double R, double phi, double z) {
  double res=0, result;
  size_t neval;
  Fparams params;
  params.R = R;
  params.phi = phi;
  params.z = z;
  params.a1 = a1_;
  params.a2 = a2_;
  params.t1 = t1_;
  params.t2 = t2_;
  params.z1 = z1_;
  params.z2 = z2_;
  params.hr = this;
  gsl_function F;
  F.params = &params;
  F.function = &dM;
  if (gsl_integration_romberg(&F, a1_, a2_, 0, 1e-10, &result, &neval, rmbrg)
      != GSL_SUCCESS ) std::cout << "Error in romberg integration" << std::endl;
//  cout << "number of evaluation = " << neval << endl;
  res += result;
  F.function = &dN;
  if (gsl_integration_romberg(&F, z1_, z2_, 0, 1e-10, &result, &neval, rmbrg)
      != GSL_SUCCESS ) std::cout << "Error in romberg integration" << std::endl;
//  cout << "number of evaluation = " << neval << endl;
  res += result;
  return -four_pi_G_ * rho0_ * res / 4.0 / M_PI;
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
  Real R0   = pin->GetReal("problem","R0");
  Real phi0 = pin->GetReal("problem","phi0");
  Real z0   = pin->GetReal("problem","z0");
  Real a1   = pin->GetReal("problem","a1");
  Real a2   = pin->GetReal("problem","a2");
  Real t1   = pin->GetReal("problem","t1");
  Real t2   = pin->GetReal("problem","t2");
  Real z1   = pin->GetReal("problem","z1");
  Real z2   = pin->GetReal("problem","z2");
  int iprob = pin->GetInteger("problem","iprob");
  int pfold = pin->GetInteger("problem","pfold");
 
  Real four_pi_G = pin->GetReal("problem","four_pi_G");

  Hure hr(a1, a2, t1, t2, z1, z2);
  
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (iprob==1) {
          /* homogeneous torus (Bannikova 2011) */
          r2 = SQR(pcoord->x1v(i)) + SQR(R0)
               - 2*pcoord->x1v(i)*R0
               + SQR(pcoord->x3v(k) - z0);
          if (r2 < rad*rad) {
            phydro->u(IDN,k,j,i) = rho0;
          }
          else {
            phydro->u(IDN,k,j,i) = 0.0;
          }
        }
        else if (iprob==2) {
          phydro->u(IDN,k,j,i) = 0.0;
          phydro->u(IM1,k,j,i) = 0;
          /* uniform sphere */
          for (int p=2;p<pfold;++p) {
            r2 = SQR(pcoord->x1v(i)) + SQR(R0)
                 - 2*pcoord->x1v(i)*R0*cos(pcoord->x2v(j) - phi0 - 2*PI*p/pfold)
                 + SQR(pcoord->x3v(k) - z0);
            if (r2 < rad*rad) {
              phydro->u(IDN,k,j,i) = rho0;
              phydro->u(IM1,k,j,i) -= 0.5*(four_pi_G)*rho0*(rad*rad - r2/3.);
            }
            else {
              phydro->u(IM1,k,j,i) -= (four_pi_G)*rad*rad*rad*rho0/3./sqrt(r2);
            }
          }
        }
        else if (iprob==3) {
          /* cylindrical mesh element (Hure 2014) */
          if ( (a1 < pcoord->x1v(i))&&(pcoord->x1v(i) < a2)&&(t1 < pcoord->x2v(j))&&(pcoord->x2v(j) < t2)&&(z1 < pcoord->x3v(k))&&(pcoord->x3v(k) < z2) ) {
            phydro->u(IDN,k,j,i) = rho0;
          }
          else {
            phydro->u(IDN,k,j,i) = 0.0;
          }
          phydro->u(IM1,k,j,i) = hr.Phi(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k));
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
  OBCGravityCyl *pog = pogrd->pmy_og_cyl;
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
