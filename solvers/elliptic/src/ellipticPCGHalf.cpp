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
#include <cuda_fp16.h>

using libp::parAlmond::multigridLevel;
using libp::parAlmond::parAlmond_t;

// #define PCG_BLOCKSIZE 512
#define p_mask 0xffffffff
#define p_Nwarp 4
#define p_warpSize 32
#define PCG_BLOCKSIZE 128
#define p_Nloads 10

int pcgSolveHalf(elliptic_t&           elliptic,
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
  timePoint_t startDouble;
  timePoint_t endDouble;
  timePoint_t startSingle;
  timePoint_t endSingle;
  timePoint_t startHalf;
  timePoint_t endHalf;
  dfloat      elapsedTimeCG;
  dfloat      elapsedTimeDouble;
  dfloat      elapsedTimeSingle;
  dfloat      elapsedTimeHalf;
  timePoint_t start;
  timePoint_t end;
  dfloat      elapsedTime;
  int         printTime = 0;

  // register scalars
  dfloat rdotz1 = 0.0;
  dfloat rdotz2 = 0.0;
  dfloat alpha = 0.0, beta = 0.0, pAp = 0.0;
  pfloat pfloat_beta = 0.0;
  ushort half_beta;
  //  memory<hfloat> beta_half(1);
  //  deviceMemory<hfloat> o_beta_half = platform.malloc<hfloat>(1);
  dfloat rdotr0 = 0.0, rdotr = 0.0;
  dfloat TOL = 0.0;

  /*Pre-reserve memory pool space to avoid some unnecessary re-sizing*/
  dlong N          = elliptic.Ndofs;
  dlong Nhalo      = elliptic.Nhalo;
  dlong Ntotal     = N + Nhalo;
  dlong Ntotaleven = (N + 1) & ~1;
  dlong Neven      = (N + 1) & ~1;

  printf("N = %d, Neven = %d \n", N, Neven);
  int            mrho = 5;
  memory<dfloat> resvec(MAXIT);
  dfloat         eta = 0.99;

  platform.reserve<std::byte>(sizeof(dfloat) * (6 * Ntotal + PCG_BLOCKSIZE) +
                              sizeof(pfloat) * 5 * Ntotal +
                              6 * platform.memPoolAlignment());

  deviceMemory<pfloat> o_pfloat_r  = platform.reserve<pfloat>(Ntotaleven);
  deviceMemory<pfloat> o_pfloat_z  = platform.reserve<pfloat>(Ntotaleven);
  deviceMemory<pfloat> o_pfloat_x  = platform.reserve<pfloat>(Ntotal);
  deviceMemory<pfloat> o_pfloat_p  = platform.reserve<pfloat>(Ntotaleven);
  deviceMemory<pfloat> o_pfloat_Ap = platform.reserve<pfloat>(Ntotal);

  // half
  deviceMemory<hfloat> o_half_r  = platform.reserve<hfloat>(Ntotal);
  deviceMemory<hfloat> o_half_z  = platform.reserve<hfloat>(Ntotaleven);
  deviceMemory<hfloat> o_half_x  = platform.reserve<hfloat>(Ntotal);
  deviceMemory<hfloat> o_half_p  = platform.reserve<hfloat>(Ntotaleven);
  deviceMemory<hfloat> o_half_Ap = platform.reserve<hfloat>(Ntotal);

  /*aux variables */
  deviceMemory<dfloat> o_p  = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_z  = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_Ap = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_ApL =
      platform.reserve<dfloat>(mesh.Np * mesh.Nelements);
  deviceMemory<dfloat> o_b       = platform.reserve<dfloat>(Ntotal);
  deviceMemory<dfloat> o_trueRes = platform.reserve<dfloat>(Ntotal);
  o_b.copyFrom(o_r);

  // for Ax
  deviceMemory<dfloat> o_MM   = mesh.o_MM;
  deviceMemory<dfloat> o_D    = mesh.o_D;
  deviceMemory<dfloat> o_S    = mesh.o_S;
  deviceMemory<dfloat> o_wJ   = mesh.o_wJ;
  deviceMemory<dfloat> o_ggeo = mesh.o_ggeo;

  // buffer for local A*p
  pinnedMemory<dfloat> h_rdotr        = platform.hostReserve<dfloat>(1);
  deviceMemory<dfloat> o_rdotr        = platform.reserve<dfloat>(1);
  pinnedMemory<pfloat> h_pfloat_rdotr = platform.hostReserve<pfloat>(1);
  deviceMemory<pfloat> o_pfloat_rdotr = platform.reserve<pfloat>(1);

  // Jac
  pfloat               one = 1.0f, zero = 0.0f;
  deviceMemory<dfloat> o_double_invDiagA = elliptic.o_double_invDiagA;
  deviceMemory<pfloat> o_invDiagA        = elliptic.o_invDiagA;
  deviceMemory<hfloat> o_half_invDiagA   = elliptic.o_half_invDiagA;

  // size of thread-blocks
  occa::dim GIn, BIn; // any kernel involving inner product
  GIn.dims    = 1;
  BIn.dims    = 2;
  int Nblocks = (N + p_Nloads * PCG_BLOCKSIZE - 1) / (p_Nloads * PCG_BLOCKSIZE);
  GIn[0]      = Nblocks;
  BIn[0]      = p_warpSize;
  BIn[1]      = p_Nwarp;
  occa::dim GInH2, BInH2; // any kernel involving inner product for half2
  GInH2.dims = 1;
  BInH2.dims = 2;
  Nblocks =
      (Neven / 2 + p_Nloads * PCG_BLOCKSIZE - 1) / (p_Nloads * PCG_BLOCKSIZE);
  GInH2[0] = Nblocks;
  BInH2[0] = p_warpSize;
  BInH2[1] = p_Nwarp;
  occa::dim GAx, BAx; // for Ax
  GAx.dims = 1;
  BAx.dims = 2;
  GAx[0]   = mesh.NlocalGatherElements;
  BAx[0]   = mesh.Nq;
  BAx[1]   = mesh.Nq;
  occa::dim G, B; // for vector operations
  G.dims  = 1;
  B.dims  = 1;
  B[0]    = PCG_BLOCKSIZE;
  Nblocks = (N + PCG_BLOCKSIZE - 1) / (PCG_BLOCKSIZE);
  G[0]    = Nblocks;
  occa::dim Ghalf2, Bhalf2; // for vector operations in half2
  Ghalf2.dims = 1;
  Bhalf2.dims = 1;
  Bhalf2[0]   = PCG_BLOCKSIZE;
  Nblocks     = (Neven / 2 + PCG_BLOCKSIZE - 1) / (PCG_BLOCKSIZE);
  Ghalf2[0]   = Nblocks;

  void debug(dlong Ncols,
             deviceMemory<pfloat> & o_res,
             deviceMemory<pfloat> & o_x,
             linAlg_t linAlg,
             mesh_t   meshL);

  void norm2half(
      elliptic_t & elliptic, dlong Ncols, deviceMemory<hfloat> & o_half_x);

  float pack_half2_as_float(__half h1, __half h2);
  // Comput norm of RHS (for stopping tolerance).
  if(settings.compareSetting("LINEAR SOLVER STOPPING CRITERION",
                             "ABS/REL-RHS-2NORM")) {
    TOL = tol;
    // dfloat normb = linAlg.norm2(N, o_r, comm);
    // TOL = std::max(tol * tol * normb * normb, tol * tol);
    // std::cout << "normb: " << normb << ", TOL: " << TOL << std::endl;
  }

  elliptic.innerProd_H_F_Kernel.setRunDims(GIn, BIn);
  elliptic.innerProd_F_F_Kernel.setRunDims(GIn, BIn);
  elliptic.innerProd_D_F_Kernel.setRunDims(GIn, BIn);
  elliptic.innerProd_H_D_Kernel.setRunDims(GIn, BIn);
  elliptic.innerProd_F_D_Kernel.setRunDims(GIn, BIn);
  elliptic.innerProd_D_D_Kernel.setRunDims(GIn, BIn);
  elliptic.innerProd_Half2_D_Kernel.setRunDims(GIn, BIn);
  elliptic.innerProd_Half2_F_Kernel.setRunDims(GInH2, BInH2);

  elliptic.zamx_H_F_Kernel.setRunDims(G, B);
  elliptic.zamx_F_F_Kernel.setRunDims(G, B);
  elliptic.zamx_D_F_Kernel.setRunDims(G, B);
  elliptic.zamx_H_D_Kernel.setRunDims(G, B);
  elliptic.zamx_F_D_Kernel.setRunDims(G, B);
  elliptic.zamx_D_D_Kernel.setRunDims(G, B);
  elliptic.zamx_Half2_D_Kernel.setRunDims(Ghalf2, Bhalf2);
  elliptic.zamx_Half2_F_Kernel.setRunDims(Ghalf2, Bhalf2);

  elliptic.axpy_H_Kernel.setRunDims(G, B);
  elliptic.axpy_Half2_Kernel.setRunDims(Ghalf2, Bhalf2);
  elliptic.axpy_F2_Kernel.setRunDims(Ghalf2, Bhalf2);
  elliptic.axpy_F_Kernel.setRunDims(G, B);
  elliptic.axpy_D_Kernel.setRunDims(G, B);

  elliptic.AxTrilinearGatherDot_H_F_Kernel.setRunDims(GAx, BAx);
  elliptic.AxTrilinearGatherDot_H_D_Kernel.setRunDims(GAx, BAx);
  elliptic.AxTrilinearGatherDot_F_F_Kernel.setRunDims(GAx, BAx);
  elliptic.AxTrilinearGatherDot_F_D_Kernel.setRunDims(GAx, BAx);
  elliptic.AxTrilinearGatherDot_D_F_Kernel.setRunDims(GAx, BAx);
  elliptic.AxTrilinearGatherDot_D_D_Kernel.setRunDims(GAx, BAx);

  elliptic.updatePCGFloat_H_Kernel.setRunDims(GIn, BIn);
  elliptic.updatePCGFloat_F_Kernel.setRunDims(GIn, BIn);
  elliptic.updatePCGFloat_D_Kernel.setRunDims(GIn, BIn);
  elliptic.updatePCGDouble_H_Kernel.setRunDims(GIn, BIn);
  elliptic.updatePCGDouble_F_Kernel.setRunDims(GIn, BIn);
  elliptic.updatePCGDouble_D_Kernel.setRunDims(GIn, BIn);

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
    TOL = tol;
  }

  if(verbose && (rank == 0))
    printf("PCG: initial res norm %12.12f \n", sqrt(rdotr0));

  int iter;
  int cnt = 0;

  linAlg.d2p(N, o_r, o_pfloat_r, 0);

  elliptic.d2hKernel.setRunDims(G, B);
  elliptic.d2hKernel(N, o_r, o_half_r, 0);
  elliptic.d2hKernel(N, o_x, o_half_x, 0);

  dfloat single_z = 1e-4;
  dfloat half_z   = 1e-8;
  dfloat single_r = 1e-6;
  int    trueRes  = 0;

  dfloat relresnorm      = 1.0;
  dfloat invNormr        = 1.0;
  int    switch_single_z = 0;
  int    switch_half_z   = 1;
  int    switch_single_r = 0;
  int    iterSingle      = 0;
  int    iterHalf        = 0;

  linAlg.d2p(N, o_p, o_pfloat_p, 0);
  elliptic.p2hKernel.setRunDims(G, B);
  elliptic.p2hKernel(N, o_pfloat_p, o_half_p, 0);

  startCG     = GlobalPlatformTime(platform);
  startSingle = GlobalPlatformTime(platform);

  for(iter = 0; iter < MAXIT; iter++) {

    if(relresnorm < single_z && switch_single_z == 0 && switch_half_z == 0) {
      // if(iter==2 && switch_single_z==0 && switch_half_z==0){
      startSingle = GlobalPlatformTime(platform);
      linAlg.d2p(N, o_p, o_pfloat_p, 0);
      switch_single_z = 1;
      printf("Switch z and p to float \n");
      iterSingle = iter;
    }
    //    if(iter==4 && switch_single_r==0){
    //    if(relresnorm<single_r && switch_single_r==0){
    //   startHalf = GlobalPlatformTime(platform);
    //   // from double to float
    //   linAlg.d2p(N, o_r, o_pfloat_r, 0);
    //   switch_single_r = 1;
    //   printf("Switch r, Ap, and ggeo to float \n");
    //   iterHalf = iter;
    // }
    if(((relresnorm * 8.3 * 1e-7 / (1 - eta)) < tol) &&
       (switch_single_r == 0)) {
      startHalf = GlobalPlatformTime(platform);
      // from double to float
      linAlg.d2p(N, o_r, o_pfloat_r, 0);
      switch_single_r = 1;
      printf("Switch r, Ap, and ggeo to float \n");
      iterHalf = iter;
    }
    resvec[iter] = sqrt(rdotr0);
    //    printf("eta = %g, relresnorm = %g \n", eta, relresnorm);
    if(iter > mrho) {
      eta = std::pow(resvec[iter] / resvec[iter - mrho], 1.0 / mrho);
      if(eta > 1) { eta = 0.99; }
    } else {
      eta = 0.99;
    }

    // if(iter==4 && switch_half_z==0){
    if(relresnorm < half_z && switch_half_z == 0) {
      elliptic.p2hKernel.setRunDims(G, B);
      elliptic.p2hKernel(N, o_pfloat_p, o_half_p, 0);
      switch_half_z   = 1;
      switch_single_z = 0;
      printf("Switch z and p to half \n");
      iterHalf = iter;
    }

    // Exit if tolerance is reached, taking at least one step.
    if(((iter == 0) && (rdotr0 == 0.0)) ||
       ((iter > 0) && (relresnorm <= TOL))) {
      break;
    }

    if(switch_single_r) {
      if(switch_half_z) {
        //	elliptic.zamx_H_F_Kernel(N, invNormr, o_half_invDiagA, o_invDiagA,
        //o_pfloat_r, o_half_z, 0);
        elliptic.zamx_Half2_F_Kernel(
            Neven / 2, invNormr, o_half_invDiagA, o_pfloat_r, o_half_z, 0);
      } else if(switch_single_z) {
        elliptic.zamx_F_F_Kernel(
            N, invNormr, o_invDiagA, o_pfloat_r, o_pfloat_z, 0);
      } else {
        elliptic.zamx_D_F_Kernel(
            N, static_cast<dfloat>(1.0), o_double_invDiagA, o_pfloat_r, o_z, 0);
      }
    } else {
      if(switch_half_z) {
        //	elliptic.zamx_H_D_Kernel(N, invNormr, o_half_invDiagA, o_invDiagA,
        //o_r, o_half_z, 0);
        elliptic.zamx_Half2_D_Kernel(
            Neven / 2, invNormr, o_half_invDiagA, o_r, o_half_z, 0);
      } else if(switch_single_z) {
        elliptic.zamx_F_D_Kernel(N, invNormr, o_invDiagA, o_r, o_pfloat_z, 0);
      } else {
        elliptic.zamx_D_D_Kernel(
            N, static_cast<dfloat>(1.0), o_double_invDiagA, o_r, o_z, 0);
      }
    }

    rdotz2 = rdotz1;
    rdotz1 = 0;

    // p2d + r.z + set Ap=0
    linAlg.set(1, static_cast<dfloat>(0.0), o_rdotr);
    if(switch_single_r) { // r in single
      if(switch_half_z) {
        //	elliptic.innerProd_H_F_Kernel(N, o_pfloat_r, o_half_z, o_rdotr, 0);
        elliptic.innerProd_Half2_F_Kernel(
            Neven / 2, N, o_pfloat_r, o_half_z, o_rdotr, 0);
      } else if(switch_single_z) {
        elliptic.innerProd_F_F_Kernel(N, o_pfloat_r, o_pfloat_z, o_rdotr, 0);
      } else {
        elliptic.innerProd_D_F_Kernel(N, o_pfloat_r, o_z, o_rdotr, 0);
      }
    } else {
      if(switch_half_z) {
        elliptic.innerProd_Half2_D_Kernel(
            Neven / 2, N, o_r, o_half_z, o_rdotr, 0);
        //	elliptic.innerProd_H_D_Kernel(N, o_r, o_half_z, o_rdotr, 0);
      } else if(switch_single_z) {
        elliptic.innerProd_F_D_Kernel(N, o_r, o_pfloat_z, o_rdotr, 0);
      } else {
        elliptic.innerProd_D_D_Kernel(N, o_r, o_z, o_rdotr, 0);
      }
    }

    h_rdotr.copyFrom(o_rdotr);
    rdotz1 = h_rdotr[0];

    beta = (iter == 0) ? 0.0 : rdotz1 / rdotz2;

    // p = z + beta*p,
    if(switch_half_z) {
      half_beta = __half_as_ushort(__float2half(beta));
      // elliptic.axpy_H_Kernel(N, half_beta, o_half_z, o_half_p, 0);
      elliptic.axpy_Half2_Kernel(Neven / 2, half_beta, o_half_z, o_half_p, 0);
      //      norm2half(elliptic, N, o_half_p);
    } else if(switch_single_z) {
      pfloat_beta = beta;
      elliptic.axpy_F2_Kernel(
          Neven / 2, pfloat_beta, o_pfloat_z, o_pfloat_p, 0);
      //      elliptic.axpy_F_Kernel(N, pfloat_beta, o_pfloat_z, o_pfloat_p, 0);
    } else {
      elliptic.axpy_D_Kernel(N, beta, o_z, o_p, 0);
    }

    // A*p + p'*Ap
    linAlg.set<dfloat>(1, (dfloat)0, o_rdotr);

    if(switch_single_r)
      linAlg.set<pfloat>(N, (pfloat)0, o_pfloat_Ap);
    else
      linAlg.set<dfloat>(N, (dfloat)0, o_Ap);

    if(switch_single_r) { // float Ap
      if(switch_half_z) {
        elliptic.AxTrilinearGatherDot_H_F_Kernel(
            mesh.NlocalGatherElements,
            mesh.o_localGatherElementList,
            0,
            mesh.NlocalGatherElements,
            elliptic.o_GlobalToLocal,
            mesh.o_pfloat_EXYZ,
            mesh.o_pfloat_gllzw,
            mesh.o_pfloat_D,
            mesh.o_pfloat_S,
            mesh.o_pfloat_MM,
            static_cast<dfloat>(elliptic.lambda),
            o_half_p,
            o_pfloat_Ap,
            o_rdotr,
            0);
      } else if(switch_single_z) {
        elliptic.AxTrilinearGatherDot_F_F_Kernel(
            mesh.NlocalGatherElements,
            mesh.o_localGatherElementList,
            0,
            mesh.NlocalGatherElements,
            elliptic.o_GlobalToLocal,
            mesh.o_pfloat_EXYZ,
            mesh.o_pfloat_gllzw,
            mesh.o_pfloat_D,
            mesh.o_pfloat_S,
            mesh.o_pfloat_MM,
            static_cast<dfloat>(elliptic.lambda),
            o_pfloat_p,
            o_pfloat_Ap,
            o_rdotr,
            0);

      } else {
        elliptic.AxTrilinearGatherDot_D_F_Kernel(
            mesh.NlocalGatherElements,
            mesh.o_localGatherElementList,
            0,
            mesh.NlocalGatherElements,
            elliptic.o_GlobalToLocal,
            mesh.o_pfloat_EXYZ,
            mesh.o_pfloat_gllzw,
            mesh.o_pfloat_D,
            mesh.o_pfloat_S,
            mesh.o_pfloat_MM,
            static_cast<dfloat>(elliptic.lambda),
            o_p,
            o_pfloat_Ap,
            o_rdotr,
            0);
      }
    } else { // double Ap
      if(switch_half_z) {
        elliptic.AxTrilinearGatherDot_H_D_Kernel(
            mesh.NlocalGatherElements,
            mesh.o_localGatherElementList,
            0,
            mesh.NlocalGatherElements,
            elliptic.o_GlobalToLocal,
            mesh.o_EXYZ,
            mesh.o_gllzw,
            mesh.o_D,
            mesh.o_S,
            mesh.o_MM,
            static_cast<dfloat>(elliptic.lambda),
            o_half_p,
            o_Ap,
            o_rdotr,
            0);
      } else if(switch_single_z) {
        elliptic.AxTrilinearGatherDot_F_D_Kernel(
            mesh.NlocalGatherElements,
            mesh.o_localGatherElementList,
            0,
            mesh.NlocalGatherElements,
            elliptic.o_GlobalToLocal,
            mesh.o_EXYZ,
            mesh.o_gllzw,
            mesh.o_D,
            mesh.o_S,
            mesh.o_MM,
            static_cast<dfloat>(elliptic.lambda),
            o_pfloat_p,
            o_Ap,
            o_rdotr,
            0);

      } else {
        elliptic.AxTrilinearGatherDot_D_D_Kernel(
            mesh.NlocalGatherElements,
            mesh.o_localGatherElementList,
            0,
            mesh.NlocalGatherElements,
            elliptic.o_GlobalToLocal,
            mesh.o_EXYZ,
            mesh.o_gllzw,
            mesh.o_D,
            mesh.o_S,
            mesh.o_MM,
            static_cast<dfloat>(elliptic.lambda),
            o_p,
            o_Ap,
            o_rdotr,
            0);
      }
    }
    h_rdotr.copyFrom(o_rdotr);
    pAp = h_rdotr[0];

    alpha = rdotz1 / pAp;

    // x <= x + alpha*p, r <= r - alpha*A*p, dot(r,r), d2p(r)
    linAlg.set(1, static_cast<dfloat>(0.0), o_rdotr);

    //    printf("pAp = %.12e, rdotz1 = %.12e \n", pAp, rdotz1);

    if(switch_single_r) { // float Ap, r
      if(switch_half_z) {
        elliptic.updatePCGFloat_H_Kernel(N,
                                         Nblocks,
                                         o_half_p,
                                         o_pfloat_Ap,
                                         alpha,
                                         o_x,
                                         o_pfloat_r,
                                         o_rdotr,
                                         0);
      } else if(switch_single_z) {
        elliptic.updatePCGFloat_F_Kernel(N,
                                         Nblocks,
                                         o_pfloat_p,
                                         o_pfloat_Ap,
                                         alpha,
                                         o_x,
                                         o_pfloat_r,
                                         o_rdotr,
                                         0);

      } else {
        elliptic.updatePCGFloat_D_Kernel(
            N, Nblocks, o_p, o_pfloat_Ap, alpha, o_x, o_pfloat_r, o_rdotr, 0);
      }
    } else {
      if(switch_half_z) {
        elliptic.updatePCGDouble_H_Kernel(N,
                                          Nblocks,
                                          o_half_p,
                                          o_Ap,
                                          alpha,
                                          o_x,
                                          o_r,
                                          o_pfloat_r,
                                          o_rdotr,
                                          0);
      } else if(switch_single_z) {
        elliptic.updatePCGDouble_F_Kernel(N,
                                          Nblocks,
                                          o_pfloat_p,
                                          o_Ap,
                                          alpha,
                                          o_x,
                                          o_r,
                                          o_pfloat_r,
                                          o_rdotr,
                                          0);

      } else {
        elliptic.updatePCGDouble_D_Kernel(
            N, Nblocks, o_p, o_Ap, alpha, o_x, o_r, o_pfloat_r, o_rdotr, 0);
      }
    }
    h_rdotr.copyFrom(o_rdotr);
    rdotr = sqrt(h_rdotr[0]);

    invNormr   = 1 / rdotr;
    relresnorm = rdotr / rdotr_iter0;

    // dfloat normz = linAlg.norm2(N, o_z, 0, comm);
    // dfloat normp = linAlg.norm2(N, o_p, 0, comm);
    // dfloat normAp = linAlg.norm2(N, o_Ap, 0, comm);
    // dfloat normx = linAlg.norm2(N, o_x, 0, comm);
    // dfloat normr = linAlg.norm2(N, o_r, 0, comm);
    // printf("normz = %.9e, normp = %.9e, normAp = %.9e, normx = %.9e, normr =
    // %.9e \n", normz, normp, normAp, normx, normr); printf("beta = %g, alpha =
    // %g \n", beta, alpha);

    rdotr0 = rdotr * rdotr;

    if(verbose && (rank == 0)) {
      if(rdotr0 < 0) printf("WARNING CG: rdotr = %17.15lf\n", rdotr0);

      // printf(" %.12e, %.12e \n", alpha, beta);

      if(trueRes) {
        elliptic.Operator(o_x, o_trueRes);
        linAlg.axpy(N,
                    static_cast<dfloat>(1.0),
                    o_b,
                    static_cast<dfloat>(-1.0),
                    o_trueRes);
        dfloat normTrueRes = linAlg.norm2(N, o_trueRes, comm);
        printf("(%d, %12.12le) \n", iter, normTrueRes / rdotr_iter0);
      } else {
        printf("(%d, %12.12le) \n", iter, relresnorm);
        //	printf("(%d, %12.12le) \n", iter, sqrt(rdotr0)/rdotr_iter0);
      }
      // printf("CG: it %d, r norm %12.12le, alpha = %le \n", iter + 1,
      // sqrt(rdotr0), alpha);
    }
  }

  endCG             = GlobalPlatformTime(platform);
  elapsedTimeCG     = ElapsedTime(startCG, endCG);
  elapsedTimeDouble = ElapsedTime(startCG, startSingle);
  elapsedTimeSingle = ElapsedTime(startSingle, startHalf);
  elapsedTimeHalf   = ElapsedTime(startHalf, endCG);
  //  printf("PCG runtime: %g seconds per iter, double %g (iter %d), single %g
  //  (iter %d), half %g (iter %d) \n",
  printf("%g, %d, %g, %d, %g, %d, %g, %d \n",
         elapsedTimeCG / (iter),
         iter,
         elapsedTimeDouble / iterSingle,
         iterSingle,
         elapsedTimeSingle / (iterHalf - iterSingle),
         iterHalf - iterSingle,
         elapsedTimeHalf / (iter - iterHalf),
         iter - iterHalf);

  // if(Rvt[1]==0)
  // linAlg.p2d(N, o_pfloat_x, o_x, 0);

  return iter;
}

void debug(dlong                 Ncols,
           deviceMemory<pfloat>& o_res,
           deviceMemory<pfloat>& o_x,
           linAlg_t              linAlg,
           mesh_t                meshL) {

  pfloat normrhs = linAlg.norm2(Ncols, o_res, meshL.comm);
  pfloat normx   = linAlg.norm2(Ncols, o_x, meshL.comm);
  printf(
      "Ncols = %d, normrhs = %.12e, normx = %.12e \n", Ncols, normrhs, normx);
}

void norm2half(elliptic_t&           elliptic,
               dlong                 Ncols,
               deviceMemory<hfloat>& o_half_x) {

  pinnedMemory<pfloat> h_pfloat_rdotr =
      elliptic.platform.hostReserve<pfloat>(1);
  deviceMemory<pfloat> o_pfloat_rdotr = elliptic.platform.reserve<pfloat>(1);

  int Nblocks =
      (Ncols + p_Nloads * PCG_BLOCKSIZE - 1) / (p_Nloads * PCG_BLOCKSIZE);
  occa::dim GIn, BIn;
  GIn.dims = 1;
  BIn.dims = 2;
  GIn[0]   = Nblocks;
  BIn[0]   = p_warpSize;
  BIn[1]   = p_Nwarp;

  elliptic.platform.linAlg().set(1, static_cast<pfloat>(0.0), o_pfloat_rdotr);
  elliptic.norm2Kernel.setRunDims(GIn, BIn);
  elliptic.norm2Kernel(Ncols, o_half_x, o_pfloat_rdotr);
  h_pfloat_rdotr.copyFrom(o_pfloat_rdotr);
  pfloat normInvDhalf = sqrt(h_pfloat_rdotr[0]);
  printf("norm(half)  = %.12e \n", normInvDhalf);
}
