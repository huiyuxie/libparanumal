/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse
Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#include "wave.hpp"

template <typename T>
void printMatrixLocal(linAlgMatrix_t<T>& A,
                      const char*        str) {
#if 1
  std::cout << "matrix: " << std::string(str) << "["
            << std::endl;
  for(int r = 1; r <= A.rows(); ++r) {
    for(int c = 1; c <= A.cols(); ++c) {
      //      std::cout << A(r,c) << " ";
      printf("% 5.4e ", A(r, c));
    }
    std::cout << std::endl;
  }
  std::cout << std::endl << "]" << std::endl;
#endif
}

// wave setup2
void wave_t::Setup(platform_t&     _platform,
                   mesh_t&         _mesh,
                   waveSettings_t& _settings) {
  platform = _platform;
  mesh     = _mesh;
  comm     = _mesh.comm;
  settings = _settings;

  // Trigger JIT kernel builds
  ogs::InitializeKernels(platform, ogs::Dfloat, ogs::Add);

  platform.linAlg().InitKernels(
      {"set", "sum", "norm2", "min", "max", "add"});

  // FOR THIS - WE ASSUME IPDG  and LAMBDA=1
  Nfields = 1;

  ellipticSettings = _settings.extractEllipticSettings();

  // Boundary type translation
  // bc = 1 -> wall
  // bc = 2 -> inflow
  // bc = 3 -> outflow
  // bc = 4 -> x-aligned slip
  // bc = 5 -> y-aligned slip
  // bc = 6 -> z-aligned slip
  int         NBCTypes = 7;
  memory<int> BCType(NBCTypes);
  // bc=3 => outflow => Neumann   => vBCType[3] = 2, etc.
  BCType[0] = 0;
  BCType[1] = 1;
  BCType[2] = 1;
  BCType[3] = 2;
  BCType[4] = 1;
  BCType[5] = 2;
  BCType[6] = 2;

  // notes: elliptic solver (L + lambda*M)*x = b
  // L stiffness matrix, M mass matrix
  // find stiffness matirx ??
  lambdaSolve = 1;
  elliptic.Setup(platform,
                 mesh,
                 ellipticSettings,
                 lambdaSolve,
                 NBCTypes,
                 BCType);

  // find out if this is a C0 discretization
  disc_c0 = elliptic.settings.compareSetting(
                "DISCRETIZATION", "CONTINUOUS")
                ? 1
                : 0;

  if(disc_c0 == 1 && mesh.elementType == Mesh::TRIANGLES) {
    std::cout << "TRYING TO USE TRIANGLE MESH WITH C0 NOT "
                 "ALLOWED WITH WAVE"
              << std::endl;
    exit(-1);
  }

  if(disc_c0 == 1 && mesh.elementType == Mesh::TETRAHEDRA) {
    std::cout
        << "TRYING TO USE TETRAHEDRA MESH WITH C0 NOT ALLOWED WITH
        WAVE "
        << std::endl;
    exit(-1);
  }

  // initialize time stepper
  linAlgMatrix_t<dfloat> BTABLE;

  Nstages  = 0;
  embedded = 0;

  std::string timeIntegrator;
  settings.getSetting("TIME INTEGRATOR", timeIntegrator);
  // notes: BTABLE size (Nstage+2, Nstages+1)
  // search ESDIRK4(3)6L{2}SA for details
  libp::TimeStepper::butcherTables(
      timeIntegrator.c_str(), Nstages, embedded, BTABLE);

  // extract alphas, betas
  alpha.reshape(Nstages + 1, Nstages);
  beta.reshape(1, Nstages);
  betahat.reshape(1, Nstages);
  esdirkC.reshape(1, Nstages);

  for(int n = 1; n <= Nstages; ++n) {
    for(int m = 1; m <= Nstages; ++m) {
      alpha(n, m) = BTABLE(n, m + 1);
    }
    beta(1, n) = BTABLE(Nstages + 1, n + 1);

    if(embedded) betahat(1, n) = BTABLE(Nstages + 2, n + 1);
    esdirkC(n) = BTABLE(n, 1);
  }

  // why do this ??
  for(int m = 1; m <= Nstages; ++m) {
    alpha(Nstages + 1, m) = beta(1, m);
  }

  gamma    = alpha(2, 2);
  invGamma = 1. / gamma;

  std::cout << "gamma = " << gamma << std::endl;

  // some changes here
  // notes: alphatilde and gammatilde are used in
  // waveStageFinalize* okl kernels
  alphatilde.reshape(Nstages + 1, Nstages);
  gammatilde.reshape(1, Nstages);
  alphatilde = 0.;
  gammatilde = 0.;

  // notes: original code sets alphatilde(1, 1) = 0.
  // and sets gammatilde(1, 1) = 1./gamma
  // for(int i=0;i<=Nstages-1;++i){
  //   for(int j=1;j<=i;++j){
  //     alphatilde(i+1,j) += alpha(i+1,j)/gamma;
  //   }
  //   for(int j=1;j<=i;++j){
  //     for(int k=1;k<=j;++k){
  //       alphatilde(i+1,k) +=
  //       alpha(i+1,j)*alpha(j,k)/(gamma*gamma);
  //     }
  //   }
  //   gammatilde(1,i+1) = 1./gamma;
  //   for(int j=1;j<=i;++j){
  //     gammatilde(1,i+1) += alpha(i+1,j)/(gamma*gamma);
  //   }
  // }

  // set alphatilde(1, 1) = 0.
  // and gammatilde(1, 1) = 1.
  for(int i = 0; i <= Nstages - 1; ++i) {
    for(int j = 1; j <= i; ++i) {
      alphatilde(i + 1, j) += alpha(i + 1, j);
    }
    for(int j = 1; j <= i; ++j) {
      for(int k = 1; k <= j; ++k) {
        alphatilde(i + 1, k) +=
            alpha(i + 1, j) * alpha(j, k) / gamma;
      }
    }
    gammatilde(1, i + 1) = 1.;
    for(int j = 1; j <= i; ++j) {
      gammatilde(1, i + 1) += alpha(i + 1, j) / gamma;
    }
  }

  // notes: what are these used for?
  //   linAlgMatrix_t<dfloat> betaAlpha(1, Nstages);
  //   linAlgMatrix_t<dfloat> betahatAlpha(1, Nstages);
  //   betaAlpha = 0.;
  //   betahatAlpha = 0.;
  //   for (int i = 1; i <= Nstages; ++i) {
  //     for (int j = 1; j <= i; ++j) {
  //       betaAlpha(1, j) += beta(1, i) * alpha(i, j);
  //       betahatAlpha(1, j) += betahat(1, i) * alpha(i,
  //       j);
  //     }
  //   }

  // notes: seems not used ??
  //   for (int i = 1; i <= Nstages; ++i) {
  //     alphatilde(1 + Nstages, i) = betaAlpha(1, i);
  //   }

  // create occa buffers
  DL.malloc(Nall);
  PL.malloc(Nall);
  DrhsL.malloc(Nall);
  PhatL.malloc(Nall * Nstages);
  DhatL.malloc(Nall * Nstages);

  o_DL        = platform.malloc<dfloat>(Nall);
  o_PL        = platform.malloc<dfloat>(Nall);
  o_DtildeL   = platform.malloc<dfloat>(Nall);
  o_DrhsL     = platform.malloc<dfloat>(Nall);
  o_DhatL     = platform.malloc<dfloat>(Nall * Nstages);
  o_PhatL     = platform.malloc<dfloat>(Nall * Nstages);
  o_scratch1L = platform.malloc<dfloat>(Nall);
  o_scratch2L = platform.malloc<dfloat>(Nall);

  o_FL     = platform.malloc<dfloat>(Nall);
  o_filtPL = platform.malloc<dfloat>(Nall);

  memory<dfloat> invMM, V, MM;

  if(mesh.elementType == Mesh::TRIANGLES) {
    mesh.VandermondeTri2D(mesh.N, mesh.r, mesh.s, V);
    mesh.invMassMatrixTri2D(mesh.Np, V, invMM);
    mesh.MassMatrixTri2D(mesh.Np, V, MM);
  } else if(mesh.elementType == Mesh::QUADRILATERALS) {
    invMM.malloc(mesh.Np * mesh.Np);
    MM.malloc(mesh.Np * mesh.Np);
    int cnt = 0;
    for(int j = 0; j < mesh.Nq; ++j) {
      for(int i = 0; i < mesh.Nq; ++i) {
        MM[cnt]    = mesh.gllw[i] * mesh.gllw[j];
        invMM[cnt] = 1. / MM[cnt];
        ++cnt;
      }
    }
  } else if(mesh.elementType == Mesh::TETRAHEDRA) {
    mesh.VandermondeTet3D(
        mesh.N, mesh.r, mesh.s, mesh.t, V);
    mesh.invMassMatrixTet3D(mesh.Np, V, invMM);
    mesh.MassMatrixTet3D(mesh.Np, V, MM);
  } else {
    invMM.malloc(mesh.Np * mesh.Np);
    MM.malloc(mesh.Np * mesh.Np);

    int cnt = 0;
    for(int k = 0; k < mesh.Nq; ++k) {
      for(int j = 0; j < mesh.Nq; ++j) {
        for(int i = 0; i < mesh.Nq; ++i) {
          MM[cnt] =
              mesh.gllw[i] * mesh.gllw[j] * mesh.gllw[k];
          invMM[cnt] = 1. / MM[cnt];
          printf("MM[%d]=%g\n", cnt, MM[cnt]);
          ++cnt;
        }
      }
    }
  }

  o_invMM =
      platform.malloc<dfloat>(mesh.Np * mesh.Np, invMM);
  o_MM = platform.malloc<dfloat>(mesh.Np * mesh.Np, MM);

  // triangle specific
  if(mesh.elementType == Mesh::TRIANGLES ||
     mesh.elementType == Mesh::TETRAHEDRA) {
    WJ.malloc(mesh.Nelements);
    invWJ.malloc(mesh.Nelements);

    for(int e = 0; e < mesh.Nelements; ++e) {
      invWJ[e] = 1. / mesh.wJ[e];
      WJ[e]    = mesh.wJ[e];
    }

    o_invWJ =
        platform.malloc<dfloat>(mesh.Nelements, invWJ);
    o_WJ = platform.malloc<dfloat>(mesh.Nelements, WJ);
  } else {
    // use ungathered weights in WJ on device
    WJ.malloc(mesh.Np * mesh.Nelements);
    for(int n = 0; n < mesh.Np * mesh.Nelements; ++n) {
      WJ[n] = mesh.wJ[n];
    }
    o_WJ = platform.malloc<dfloat>(mesh.Nelements * mesh.Np,
                                   WJ);

    // gather weights for C0
    if(disc_c0) {
      elliptic.ogsMasked.GatherScatter(
          WJ, 1, ogs::Add, ogs::Sym);
    }

    // use globalized mass for C0
    invWJ.malloc(mesh.Np * mesh.Nelements);
    for(int n = 0; n < mesh.Np * mesh.Nelements; ++n) {
      invWJ[n] = (WJ[n]) ? 1. / WJ[n] : 0;
    }

    o_invWJ = platform.malloc<dfloat>(
        mesh.Nelements * mesh.Np, invWJ);
  }
}