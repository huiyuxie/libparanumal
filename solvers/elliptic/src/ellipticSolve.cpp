/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <math.h>
#include "elliptic.hpp"
#include "ellipticPrecon.hpp"
#include "timer.hpp"
using libp::parAlmond::multigridLevel;
using libp::parAlmond::parAlmond_t;

// #define PCG_BLOCKSIZE 512
#define p_mask 0xffffffff
#define p_Nwarp 4
#define p_warpSize 32
#define PCG_BLOCKSIZE 128
#define p_Nloads 10

int elliptic_t::Solve(linearSolver_t<dfloat>& linearSolver,
                      deviceMemory<dfloat>&   o_x,
                      deviceMemory<dfloat>&   o_r,
                      const dfloat            tol,
                      const int               MAXIT,
                      const int               verbose) {
  // setup preconditioner
  //  MultiGridPrecon mgPrecon(*this);

  memory<dfloat> diagA(Ndofs);
  memory<dfloat> double_invDiagA(Ndofs);
  memory<pfloat> invDiagA(Ndofs);
  memory<hfloat> half_invDiagA(Ndofs);
  BuildOperatorDiagonal(diagA);
  for(dlong n = 0; n < Ndofs; n++) {
    invDiagA[n]        = 1.0 / diagA[n];
    double_invDiagA[n] = 1.0 / diagA[n];
    half_invDiagA[n]   = __float2half(1.0 / diagA[n]);
  }
  o_double_invDiagA = platform.malloc<dfloat>(double_invDiagA);
  o_invDiagA        = platform.malloc<pfloat>(invDiagA);
  o_half_invDiagA   = platform.malloc<hfloat>(half_invDiagA);

  // If there is a nullspace, remove the constant vector from r
  if(allNeumann) ZeroMean(o_r);

  // Reserve memory and backup original vectors
  deviceMemory<dfloat> o_xx = platform.reserve<dfloat>(Ndofs + Nhalo);
  deviceMemory<dfloat> o_rr = platform.reserve<dfloat>(Ndofs + Nhalo);
  o_xx.copyFrom(o_x);
  o_rr.copyFrom(o_r);

  int Niter;

  int pcgSolveHalf(elliptic_t & elliptic,
                   deviceMemory<dfloat> & o_x,
                   deviceMemory<dfloat> & o_r,
                   const dfloat tol,
                   const int    MAXIT,
                   memory<int>  Rvt,
                   int          version,
                   const int    verbose);

  int pcgSolveDouble(elliptic_t & elliptic,
                     deviceMemory<dfloat> & o_x,
                     deviceMemory<dfloat> & o_r,
                     const dfloat tol,
                     const int    MAXIT,
                     memory<int>  Rvt,
                     int          version,
                     const int    verbose);

  printf("Length of vectors: %d \n", Ndofs + Nhalo);

  int         version;
  memory<int> Rvt(6, 0); // precisions for z, x, Ap, r, p
  Rvt[0] = 0;            // z
  Rvt[1] = 1;            // x
  Rvt[2] = 0;            // Ap
  Rvt[3] = 0;            // r
  Rvt[4] = 1;            // p
  Rvt[5] = 1;            // ggeo

  if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "NONE")) {
    printf("Vanilla PCG                    \n ");
    version = 0;
  } else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "FUSE")) {
    printf("Fusing kernels                  \n  ");
    version = 1;
  } else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION",
                                    "AXTRILINEAR")) {
    printf("Kernel fusion + Trilinear geo in MG             \n  ");
    version = 2;
  } else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "AXINTERP")) {
    printf("Kernel fusion + Interp geo in MG             \n  ");
    version = 3;
  } else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "ALT")) {
    printf("Kernel fusion + alternating directions + ALT + NC \n");
    version = 4;
  } else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION",
                                    "FUSE+ALT+NC")) {
    printf("Kernel fusion + alternating directions + non-cache load/store\n");
    printf("Under construction \n");
  }

#if 0
  memory<dfloat> x(Ndofs + Nhalo);
  deviceMemory<dfloat> o_xExact  = platform.malloc<dfloat>(x);                         
  for (dlong i=0;i<(Ndofs + Nhalo);i++) x[i] = (dfloat) drand48();
  o_xExact.copyFrom(x);             
  Operator(o_xExact, o_r);
#endif

  printf("Warm up: ");
  if(settings.compareSetting("MULTIGRID PRECISION", "FLOAT")) {
    //    Niter = pcgSolveFloat(*this, mgPrecon, o_x, o_r, tol, 2, Rvt, version,
    //    verbose);
  } else if(settings.compareSetting("MULTIGRID PRECISION", "HALF")) {
    Niter = pcgSolveHalf(*this, o_x, o_r, tol, 3, Rvt, version, verbose);
  } else if(settings.compareSetting("MULTIGRID PRECISION", "DOUBLE")) {
    Niter = pcgSolveDouble(*this, o_x, o_r, tol, 5, Rvt, version, verbose);
  }

  //    for (int i : {31, 30, 14, 6, 2, 0}) { // all double, z float, p float, r
  //    float, Ap float, x float
  for(int i :
      {14}) { // all double, z float, p float, r float, Ap float, x float
    for(int j = 0; j < 5; j++) { Rvt[j] = (i >> j) & 1; }
    std::cout << "Case " << i << ": ";
    std::cout << "z = " << (Rvt[0] ? "Double" : "Float") << ", ";
    std::cout << "x = " << (Rvt[1] ? "Double" : "Float") << ", ";
    std::cout << "Ap = " << (Rvt[2] ? "Double" : "Float") << ", ";
    std::cout << "r = " << (Rvt[3] ? "Double" : "Float") << ", ";
    std::cout << "p = " << (Rvt[4] ? "Double" : "Float") << ", ";
    std::cout << "ggeo = " << (Rvt[5] ? "Double" : "Float") << std::endl;

    o_x.copyFrom(o_xx);
    o_r.copyFrom(o_rr);
    printf("Collection: ");
    if(settings.compareSetting("MULTIGRID PRECISION", "FLOAT")) {
      //      Niter = pcgSolveFloat(*this, mgPrecon, o_x, o_r, tol, MAXIT, Rvt,
      //      version, verbose);
    } else if(settings.compareSetting("MULTIGRID PRECISION", "HALF")) {
      Niter = pcgSolveHalf(*this, o_x, o_r, tol, MAXIT, Rvt, version, verbose);
    } else if(settings.compareSetting("MULTIGRID PRECISION", "DOUBLE")) {
      Niter =
          pcgSolveDouble(*this, o_x, o_r, tol, MAXIT, Rvt, version, verbose);
    }

    int iterSwitch;
    settings.getSetting("LINEAR SOLVER PRECISION SWITCH", iterSwitch);
#if 0
    deviceMemory<dfloat> o_err  = platform.malloc<dfloat>(Ndofs + Nhalo);
    platform.linAlg().zaxpy(Ndofs + Nhalo, 1.0, o_xExact, -1.0, o_x, o_err);
    dfloat errl2 = platform.linAlg().norm2(Ndofs + Nhalo, o_err, 0, comm);
    dfloat normx = platform.linAlg().norm2(Ndofs + Nhalo, o_xExact, 0, comm);
    printf("relative errl2: (%d,%g)\n", iterSwitch, errl2/normx);
#endif
    dfloat errH1 = ComputeError(o_x);
    printf("relative errH1: (%d,%g)\n", iterSwitch, errH1);
  }

  return Niter;
}
