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

#include "elliptic.hpp"
#include "ellipticPrecon.hpp"
#include "timer.hpp"

#include <cstdio>
//#include </usr/local/cuda-12.3/include/cuda_runtime.h>

void elliptic_t::Run() {

#if 0
  int device_id = 0;
  cudaDeviceProp prop;

  // Get device properties
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  size_t size = 1.0*prop.persistingL2CacheMaxSize;
  printf("Persisting L2 cache size: %lu MB\n", size/(1024*1024));
  // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
#endif

  // setup linear algebra module
  platform.linAlg().InitKernels(
      {"set", "norm2", "axpy", "d2p", "p2d", "amxpy", "innerProd"});

  // setup linear solver
  hlong NglobalDofs;
  if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    NglobalDofs = ogsMasked.NgatherGlobal * Nfields;
  } else {
    NglobalDofs = mesh.NelementsGlobal * mesh.Np * Nfields;
  }

  linearSolver_t<dfloat> linearSolver;
  if(settings.compareSetting("LINEAR SOLVER", "NBPCG")) {
    linearSolver.Setup<LinearSolver::nbpcg<dfloat>>(
        Ndofs, Nhalo, platform, settings, comm);
  } else if(settings.compareSetting("LINEAR SOLVER", "NBFPCG")) {
    linearSolver.Setup<LinearSolver::nbfpcg<dfloat>>(
        Ndofs, Nhalo, platform, settings, comm);
  } else if(settings.compareSetting("LINEAR SOLVER", "PCG")) {
    linearSolver.Setup<LinearSolver::pcg<dfloat>>(
        Ndofs, Nhalo, platform, settings, comm);
  } else if(settings.compareSetting("LINEAR SOLVER", "PGMRES")) {
    linearSolver.Setup<LinearSolver::pgmres<dfloat>>(
        Ndofs, Nhalo, platform, settings, comm);
  } else if(settings.compareSetting("LINEAR SOLVER", "PMINRES")) {
    linearSolver.Setup<LinearSolver::pminres<dfloat>>(
        Ndofs, Nhalo, platform, settings, comm);
  }

  properties_t kernelInfo = mesh.props; // copy base occa properties

  std::string dataFileName;
  settings.getSetting("DATA FILE", dataFileName);
  kernelInfo["includes"] += dataFileName;

  // add standard boundary functions
  std::string boundaryHeaderFileName;
  if(mesh.dim == 2)
    boundaryHeaderFileName =
        std::string(DELLIPTIC "/data/ellipticBoundary2D.h");
  else if(mesh.dim == 3)
    boundaryHeaderFileName =
        std::string(DELLIPTIC "/data/ellipticBoundary3D.h");
  kernelInfo["includes"] += boundaryHeaderFileName;

  int Nmax             = std::max(mesh.Np, mesh.Nfaces * mesh.Nfp);
  kernelInfo["defines/"
             "p_Nmax"] = Nmax;

  kernelInfo["defines/"
             "p_Nfields"] = Nfields;

  // set kernel name suffix
  std::string suffix = mesh.elementSuffix();

  std::string oklFilePrefix = DELLIPTIC "/okl/";
  std::string oklFileSuffix = ".okl";

  std::string fileName, kernelName;

  fileName   = oklFilePrefix + "ellipticRhs" + suffix + oklFileSuffix;
  kernelName = "ellipticRhs" + suffix;
  kernel_t forcingKernel =
      platform.buildKernel(fileName, kernelName, kernelInfo);

  //  kernel_t rhsBCKernel, addBCKernel;
  if(settings.compareSetting("DISCRETIZATION", "IPDG")) {
    fileName   = oklFilePrefix + "ellipticRhsBCIpdg" + suffix + oklFileSuffix;
    kernelName = "ellipticRhsBCIpdg" + suffix;

    rhsBCKernel = platform.buildKernel(fileName, kernelName, kernelInfo);
  } else if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    fileName   = oklFilePrefix + "ellipticRhsBC" + suffix + oklFileSuffix;
    kernelName = "ellipticRhsBC" + suffix;

    rhsBCKernel = platform.buildKernel(fileName, kernelName, kernelInfo);

    fileName   = oklFilePrefix + "ellipticAddBC" + suffix + oklFileSuffix;
    kernelName = "ellipticAddBC" + suffix;

    addBCKernel = platform.buildKernel(fileName, kernelName, kernelInfo);
  }

  // create occa buffers
  dlong                Nall = mesh.Np * (mesh.Nelements + mesh.totalHaloPairs);
  memory<dfloat>       rL(Nall);
  memory<dfloat>       xL(Nall);
  deviceMemory<dfloat> o_rL = platform.malloc<dfloat>(Nall);
  deviceMemory<dfloat> o_xL = platform.malloc<dfloat>(Nall);

  deviceMemory<dfloat> o_r, o_x;
  if(settings.compareSetting("DISCRETIZATION", "IPDG")) {
    o_r = o_rL;
    o_x = o_xL;
  } else {
    dlong Ng     = ogsMasked.Ngather;
    dlong Nghalo = gHalo.Nhalo;
    dlong Ngall  = Ng + Nghalo;
    o_r          = platform.malloc<dfloat>(Ngall);
    o_x          = platform.malloc<dfloat>(Ngall);
  }

  mesh.MassMatrixKernelSetup(Nfields); // mass matrix operator

  // populate rhs forcing
  forcingKernel(mesh.Nelements,
                mesh.o_wJ,
                mesh.o_MM,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                lambda,
                o_rL);

  // Set x to zero
  platform.linAlg().set(mesh.Nelements * mesh.Np * Nfields, (dfloat)0.0, o_xL);

  // add boundary condition contribution to rhs
  if(settings.compareSetting("DISCRETIZATION", "IPDG")) {
    rhsBCKernel(mesh.Nelements,
                mesh.o_vmapM,
                tau,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                mesh.o_vgeo,
                mesh.o_sgeo,
                o_EToB,
                mesh.o_D,
                mesh.o_LIFT,
                mesh.o_MM,
                o_rL);
  } else if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    rhsBCKernel(mesh.Nelements,
                mesh.o_wJ,
                mesh.o_ggeo,
                mesh.o_sgeo,
                mesh.o_D,
                mesh.o_S,
                mesh.o_MM,
                mesh.o_vmapM,
                mesh.o_sM,
                lambda,
                mesh.o_x,
                mesh.o_y,
                mesh.o_z,
                o_mapB,
                o_rL);
  }

  // gather rhs to globalDofs if c0
  if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    ogsMasked.Gather(o_r, o_rL, 1, ogs::Add, ogs::Trans);
    ogsMasked.Gather(o_x, o_xL, 1, ogs::Add, ogs::NoTrans);
  }

  int maxIter = 1000;
  int verbose = settings.compareSetting("VERBOSE", "TRUE") ? 1 : 0;

  timePoint_t start = GlobalPlatformTime(platform);

  // call the solver
  dfloat tol  = (sizeof(dfloat) == sizeof(double)) ? 1.0e-10 : 1.0e-10;
  int    iter = Solve(linearSolver, o_x, o_r, tol, maxIter, verbose);

  timePoint_t end         = GlobalPlatformTime(platform);
  double      elapsedTime = ElapsedTime(start, end);

  if((mesh.rank == 0) && verbose) {
    printf("%d, " hlongFormat
           ", %g, %d, %g, %g; global: N, dofs, elapsed, iterations, time per "
           "node, nodes*iterations/time %s\n",
           mesh.N,
           NglobalDofs,
           elapsedTime,
           iter,
           elapsedTime / (NglobalDofs),
           NglobalDofs * ((dfloat)iter / elapsedTime),
           (char*)settings.getSetting("PRECONDITIONER").c_str());
  }

  // add the boundary data to the masked nodes
  if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    // scatter x to LocalDofs if c0
    ogsMasked.Scatter(o_xL, o_x, 1, ogs::NoTrans);
    // fill masked nodes with BC data
    addBCKernel(mesh.Nelements, mesh.o_x, mesh.o_y, mesh.o_z, o_mapB, o_xL);
  }

  if(settings.compareSetting("OUTPUT TO FILE", "TRUE")) {

    // copy data back to host
    o_xL.copyTo(xL);

    // output field files
    std::string name;
    settings.getSetting("OUTPUT FILE NAME", name);
    char fname[BUFSIZ];
    sprintf(fname, "%s_%04d.vtu", name.c_str(), mesh.rank);

    PlotFields(xL, fname);
  }

  //  dfloat errH1 = ComputeError(o_x);

  // // output norm of final solution
  // {
  //   // compute q.M*q
  //   dlong Nentries = mesh.Nelements * mesh.Np * Nfields;
  //   deviceMemory<dfloat> o_MxL = platform.reserve<dfloat>(Nentries);
  //   mesh.MassMatrixApply(o_xL, o_MxL);

  //   deviceMemory<dfloat> o_Ax = platform.reserve<dfloat>(Ndofs);
  //   Operator(o_x, o_Ax);

  //   dfloat norm2 = sqrt(platform.linAlg().innerProd(Nentries, o_xL, o_MxL,
  //   mesh.comm)); dfloat normH1 = sqrt(platform.linAlg().innerProd(Ndofs, o_x,
  //   o_Ax, mesh.comm));

  //   // relative L2 and H1 error
  //   dlong p_NblockC = 1; // (mesh.dim==3) ? 1:(512/(mesh.cubNp));
  //   int NblocksC = (mesh.Nelements+p_NblockC-1)/p_NblockC;
  //   deviceMemory<dfloat> o_errH1 = platform.malloc<dfloat>(NblocksC);
  //   deviceMemory<dfloat> o_errL2 = platform.malloc<dfloat>(NblocksC);
  //   platform.linAlg().set(NblocksC, (dfloat)0.0, o_errH1);
  //   platform.linAlg().set(NblocksC, (dfloat)0.0, o_errL2);
  //   cubatureH1L2ErrorKernel(mesh.Nelements,
  // 			    mesh.o_cubx, mesh.o_cuby, mesh.o_cubz, mesh.o_cubwJ,
  // 			    mesh.o_cubvgeo, mesh.o_cubD, mesh.o_cubInterp, lambda,
  // 			    o_xL, o_errH1, o_errL2);
  //   dfloat errNormH1 = platform.linAlg().sum(NblocksC, o_errH1, mesh.comm);
  //   dfloat errNorm2 = platform.linAlg().sum(NblocksC, o_errL2, mesh.comm);
  //   errNormH1 = sqrt(errNormH1)/normH1;
  //   errNorm2 = sqrt(errNorm2)/norm2;
  //   if (mesh.rank == 0){
  //     printf("Solution H1 norm = %17.15lg, L2 norm = %17.15lg\n", normH1,
  //     norm2); printf("Relative error H1 norm = %17.15lg, L2 norm =
  //     %17.15lg\n", errNormH1, errNorm2);
  //   }
  // }
}

dfloat elliptic_t::ComputeError(deviceMemory<dfloat>& o_x) {
  dlong  Nall = mesh.Np * (mesh.Nelements + mesh.totalHaloPairs);
  dfloat errNormH1;

  deviceMemory<dfloat> o_xL = platform.malloc<dfloat>(Nall);
  platform.linAlg().set(mesh.Nelements * mesh.Np * Nfields, (dfloat)0.0, o_xL);

  // add the boundary data to the masked nodes
  if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    // scatter x to LocalDofs if c0
    ogsMasked.Scatter(o_xL, o_x, 1, ogs::NoTrans);
    // fill masked nodes with BC data
    addBCKernel(mesh.Nelements, mesh.o_x, mesh.o_y, mesh.o_z, o_mapB, o_xL);
  }
  // output norm of final solution
  {
    // compute q.M*q
    dlong                Nentries = mesh.Nelements * mesh.Np * Nfields;
    deviceMemory<dfloat> o_MxL    = platform.reserve<dfloat>(Nentries);
    mesh.MassMatrixApply(o_xL, o_MxL);

    deviceMemory<dfloat> o_Ax = platform.reserve<dfloat>(Ndofs);
    Operator(o_x, o_Ax);

    dfloat norm2 =
        sqrt(platform.linAlg().innerProd(Nentries, o_xL, o_MxL, mesh.comm));
    dfloat normH1 =
        sqrt(platform.linAlg().innerProd(Ndofs, o_x, o_Ax, mesh.comm));

    // relative L2 and H1 error
    dlong p_NblockC              = 1; // (mesh.dim==3) ? 1:(512/(mesh.cubNp));
    int   NblocksC               = (mesh.Nelements + p_NblockC - 1) / p_NblockC;
    deviceMemory<dfloat> o_errH1 = platform.malloc<dfloat>(NblocksC);
    deviceMemory<dfloat> o_errL2 = platform.malloc<dfloat>(NblocksC);
    platform.linAlg().set(NblocksC, (dfloat)0.0, o_errH1);
    platform.linAlg().set(NblocksC, (dfloat)0.0, o_errL2);
    cubatureH1L2ErrorKernel(mesh.Nelements,
                            mesh.o_cubx,
                            mesh.o_cuby,
                            mesh.o_cubz,
                            mesh.o_cubwJ,
                            mesh.o_cubvgeo,
                            mesh.o_cubD,
                            mesh.o_cubInterp,
                            lambda,
                            o_xL,
                            o_errH1,
                            o_errL2);
    errNormH1       = platform.linAlg().sum(NblocksC, o_errH1, mesh.comm);
    dfloat errNorm2 = platform.linAlg().sum(NblocksC, o_errL2, mesh.comm);
    errNormH1       = sqrt(errNormH1) / normH1;
    errNorm2        = sqrt(errNorm2) / norm2;
    if(mesh.rank == 0) {
      //  printf("Solution H1 norm = %17.15lg, L2 norm = %17.15lg\n", normH1,
      //  norm2);
      printf(
          "Relative err H1 = %17.15lg, L2 = %17.15lg\n", errNormH1, errNorm2);
    }
  }
  return errNormH1;
}
