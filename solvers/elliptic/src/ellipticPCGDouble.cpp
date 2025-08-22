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
#define p_Nloads 1

int pcgSolveDouble(elliptic_t&           elliptic,
                   deviceMemory<dfloat>& o_x,
                   deviceMemory<dfloat>& o_r,
                   const dfloat          tol,
                   const int             MAXIT,
                   memory<int>           Rvt,
                   int                   version,
                   const int             verbose) {

  int         rank     = elliptic.comm.rank();
  linAlg_t&   linAlg   = elliptic.platform.linAlg();
  platform_t& platform = elliptic.platform;
  comm_t&     comm     = elliptic.comm;
  settings_t& settings = elliptic.settings;
  mesh_t&     mesh     = elliptic.mesh;

  // timing
  timePoint_t startCG;
  timePoint_t endCG;
  dfloat      elapsedTimeCG;
  timePoint_t start;
  timePoint_t end;
  dfloat      elapsedTime;
  int         printTime = 0;

  // register scalars
  dfloat rdotz1 = 0.0;
  dfloat rdotz2 = 0.0;
  dfloat alpha = 0.0, beta = 0.0, pAp = 0.0;
  dfloat rdotr0 = 0.0, rdotr = 0.0;
  dfloat TOL = 0.0;

  /*Pre-reserve memory pool space to avoid some unnecessary re-sizing*/
  dlong N      = elliptic.Ndofs;
  dlong Nhalo  = elliptic.Nhalo;
  dlong Ntotal = N + Nhalo;

  platform.reserve<std::byte>(sizeof(dfloat) * (3 * Ntotal + PCG_BLOCKSIZE) +
                              sizeof(pfloat) * 5 * Ntotal +
                              6 * platform.memPoolAlignment());

  /*aux variables */
  deviceMemory<dfloat> o_p               = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_z               = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_rTrue           = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_b               = platform.reserve<dfloat>(Ntotal);
  deviceMemory<pfloat> o_pfloat_z        = platform.reserve<pfloat>(Ntotal);
  deviceMemory<pfloat> o_pfloat_p        = platform.reserve<pfloat>(Ntotal);
  deviceMemory<pfloat> o_half_z          = platform.reserve<pfloat>(Ntotal);
  deviceMemory<pfloat> o_half_p          = platform.reserve<pfloat>(Ntotal);
  deviceMemory<dfloat> o_Ap              = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_double_invDiagA = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_rdotr           = platform.reserve<dfloat>(1);
  pinnedMemory<dfloat> h_rdotr           = platform.hostReserve<dfloat>(1);

  // for Ax
  deviceMemory<dfloat> o_MM   = mesh.o_MM;
  deviceMemory<dfloat> o_D    = mesh.o_D;
  deviceMemory<dfloat> o_S    = mesh.o_S;
  deviceMemory<dfloat> o_wJ   = mesh.o_wJ;
  deviceMemory<dfloat> o_ggeo = mesh.o_ggeo;

  // Jac
  pfloat               one = 1.0f, zero = 0.0f;
  deviceMemory<pfloat> o_invDiagA = elliptic.o_invDiagA;
  linAlg.p2d(N, o_invDiagA, o_double_invDiagA);

  int Nblocks = (N + PCG_BLOCKSIZE - 1) / PCG_BLOCKSIZE;
  Nblocks = std::min(Nblocks, PCG_BLOCKSIZE); // limit to PCG_BLOCKSIZE entries

  occa::dim G, B; // for vector operations
  G.dims  = 1;
  B.dims  = 1;
  B[0]    = PCG_BLOCKSIZE;
  Nblocks = (N + PCG_BLOCKSIZE - 1) / (PCG_BLOCKSIZE);
  G[0]    = Nblocks;

  elliptic.p2hKernel.setRunDims(G, B);
  elliptic.h2pKernel.setRunDims(G, B);

  // Comput norm of RHS (for stopping tolerance).
  if(settings.compareSetting("LINEAR SOLVER STOPPING CRITERION",
                             "ABS/REL-RHS-2NORM")) {
    dfloat normb = linAlg.norm2(N, o_r, comm);
    TOL          = std::max(tol * tol * normb * normb, tol * tol);
    std::cout << "normb: " << normb << ", TOL: " << TOL << std::endl;
  }

  o_b.copyFrom(o_r);

  // compute A*x
  elliptic.Operator(o_x, o_Ap);

  // subtract r = r - A*x
  linAlg.axpy(
      N, static_cast<dfloat>(-1.0), o_Ap, static_cast<dfloat>(1.0), o_r);

  {
    memory<dfloat> h_r(N);
    o_r.copyTo(h_r);
    dfloat res = 0;
    for(int n = 0; n < N; ++n) { res += h_r[n] * h_r[n]; }
  }

  rdotr0             = linAlg.norm2(N, o_r, comm);
  dfloat rdotr_iter0 = rdotr0;
  rdotr0             = rdotr0 * rdotr0;
  printf("(%d, %12.12le) \n", 0, 1.0);

  if(settings.compareSetting("LINEAR SOLVER STOPPING CRITERION",
                             "ABS/REL-INITRESID")) {
    TOL = std::max(tol * tol * rdotr0, tol * tol);
  }

  if(verbose && (rank == 0))
    printf("PCG: initial res norm %12.12f \n", sqrt(rdotr0));

  int iter;
  int cnt = 0;

  dfloat relresnorm         = 1.0;
  dfloat invNormr           = 1.0;
  int    precisionIndicator = 0; // 0 is double, 1 is float

  dfloat singlez = 1e-18;
  dfloat halfz   = 1e-18;

  startCG = GlobalPlatformTime(platform);

  for(iter = 0; iter < MAXIT; iter++) {

    // Exit if tolerance is reached, taking at least one step.
    if(((iter == 0) && (relresnorm == 0.0)) ||
       ((iter > 0) && (relresnorm <= tol))) {
      break;
    }

    linAlg.amxpy(
        N, invNormr, o_double_invDiagA, o_r, static_cast<dfloat>(0.0), o_z);

    rdotz2 = rdotz1;
    rdotz1 = linAlg.innerProd(N, o_r, o_z, 0, comm);

    beta = (iter == 0) ? 0.0 : rdotz1 / rdotz2;

    // p = z + beta*p,
    linAlg.axpy(N, static_cast<dfloat>(1.0), o_z, beta, o_p);

    // A*p + p'*Ap
    linAlg.set<dfloat>(1, (dfloat)0, o_rdotr);
    linAlg.set<dfloat>(N, (dfloat)0, o_Ap);
    elliptic.AxTrilinearGatherDotKernel(mesh.NlocalGatherElements,
                                        mesh.o_localGatherElementList,
                                        elliptic.o_GlobalToLocal,
                                        mesh.o_EXYZ,
                                        mesh.o_gllzw,
                                        o_D,
                                        o_S,
                                        o_MM,
                                        static_cast<dfloat>(elliptic.lambda),
                                        o_p,
                                        o_Ap,
                                        o_rdotr,
                                        0);
    h_rdotr.copyFrom(o_rdotr);
    pAp = h_rdotr[0];

    alpha = rdotz1 / pAp;

    // printf("pAp = %.12e, rdotz1 = %.12e \n", pAp, rdotz1);
    linAlg.set<dfloat>(1, (dfloat)0, o_rdotr);
    elliptic.updatePCGDoubleKernel(
        N, Nblocks, o_p, o_Ap, alpha, o_x, o_r, o_rdotr);
    h_rdotr.copyFrom(o_rdotr);
    rdotr = sqrt(h_rdotr[0]);

    invNormr   = 1 / rdotr;
    relresnorm = rdotr / rdotr_iter0;

    rdotr0 = rdotr * rdotr;

    if(verbose && (rank == 0)) {
      if(rdotr0 < 0) printf("WARNING CG: rdotr = %17.15lf\n", rdotr0);
      printf("(%d, %12.12le) \n", iter + 1, relresnorm);
    }
  }

  endCG         = GlobalPlatformTime(platform);
  elapsedTimeCG = ElapsedTime(startCG, endCG);
  printf("PCG runtime: %g seconds per iter \n", elapsedTimeCG / iter);

  return iter;
}
