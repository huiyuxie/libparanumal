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

#ifndef ELLIPTIC_HPP
#define ELLIPTIC_HPP 1

#include "core.hpp"
#include "platform.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "linAlg.hpp"
#include "precon.hpp"
#include "linearSolver.hpp"
#include "parAlmond.hpp"
#include "ogs.hpp"
#include "ogs/ogsOperator.hpp"
#include "stoppingCriteria.hpp"
#include <cuda_fp16.h>

#define DELLIPTIC LIBP_DIR"/solvers/elliptic/"

using namespace libp;

// Matrix-free p-Multigrid levels followed by AMG                                                                     

class ellipticSettings_t: public settings_t {
public:
  ellipticSettings_t() = default;
  ellipticSettings_t(const comm_t& _comm);
  void report();
  void parseFromFile(platformSettings_t& platformSettings,
                     meshSettings_t& meshSettings,
                     const std::string filename);
};
void ellipticAddRunSettings(settings_t& settings);
void ellipticAddSettings(settings_t& settings,
                         const std::string prefix="");

class elliptic_t: public solver_t {
public:
  mesh_t mesh;

  dlong Ndofs, Nhalo;
  int Nfields;

  dfloat lambda;
  dfloat tau;

  int disc_ipdg, disc_c0;

  int multigridDriver = 0;
  
  ogs::halo_t traceHalo;

  precon_t precon;
  
  // NOTE pfloat
  memory<pfloat> weight, weightG;
  deviceMemory<pfloat> o_weight, o_weightG;

  //C0-FEM mask data
  ogs::ogs_t ogsMasked;
  ogs::halo_t gHalo;
  ogs::ogsOperator_t ogsOp;
  memory<int> mapB;      // boundary flag of face nodes
  deviceMemory<int> o_mapB;

  dlong Nmasked;
  memory<dlong> maskIds;
  memory<hlong> maskedGlobalIds;
  memory<hlong> maskedGlobalNumbering;
  memory<dlong> GlobalToLocal;

  // for coloring Ax
  int NsetE; 
  memory<hlong> NeleS;
  
  deviceMemory<dlong> o_maskIds;
  deviceMemory<dlong> o_GlobalToLocal;
  deviceMemory<dlong> o_GlobalToLocalColor;
  deviceMemory<dlong> o_localGatherElementList;

  int NBCTypes;
  memory<int> BCType;
  memory<int> EToB;
  deviceMemory<int> o_EToB;

  deviceMemory<dfloat> o_double_invDiagA;
  deviceMemory<pfloat> o_invDiagA;
  deviceMemory<hfloat> o_half_invDiagA;
  
  int allNeumann;
  dfloat allNeumannPenalty;
  dfloat allNeumannScale;

  kernel_t floatUpdateCheby1Kernel;
  kernel_t floatUpdateCheby1ncKernel;
  kernel_t floatUpdateCheby2Kernel;
  kernel_t floatUpdateCheby2ncKernel;
  kernel_t floatUpdateCheby3Kernel;
  kernel_t floatUpdateCheby3ncKernel;
  kernel_t floatUpdateCheby4Kernel;
  kernel_t floatUpdateCheby4ncKernel;
  kernel_t floatUpdateCheby5Kernel;
  kernel_t floatUpdateCheby5ncKernel;
  
  kernel_t maskKernel;
  kernel_t partialAxKernel;
  kernel_t partialGradientKernel;
  kernel_t partialIpdgKernel;

  kernel_t AxGatherKernel;
  kernel_t AxGatherncKernel;
  kernel_t AxGatherSmoothKernel;
  kernel_t AxInterpGatherKernel;
  kernel_t AxInterpGatherSmoothKernel;
  kernel_t AxTrilinearGatherKernel;
  kernel_t AxTrilinearGatherSmoothKernel;
  kernel_t AxGatherDotKernel;
  kernel_t AxTrilinearGatherDotKernel;
  kernel_t AxGatherDotncKernel;
  
  kernel_t floatPartialAxKernel;
  kernel_t floatAxGatherKernel;
  kernel_t floatAxGatherSmoothKernel;
  kernel_t floatAxTrilinearGatherKernel;
  kernel_t floatAxTrilinearGatherSmoothKernel;
  kernel_t floatAxGatherncKernel;
  kernel_t floatAxGatherDotKernel;  
  kernel_t floatAxTrilinearGatherDotKernel;  
  kernel_t floatAxGatherDotncKernel;
  kernel_t floatPartialGradientKernel;
  kernel_t floatPartialIpdgKernel;
  kernel_t floatAxInterpGatherKernel;
  kernel_t floatAxInterpGatherSmoothKernel;

  // half precision kernels
  kernel_t p2hKernel;
  kernel_t h2pKernel;
  kernel_t d2hKernel;
  kernel_t norm2Kernel;
  kernel_t setKernel;
  kernel_t axpySetHalfKernel;
  kernel_t axpyDoubleKernel;
  kernel_t p2dInnerProdHalfKernel;
  kernel_t p2dInnerProdHalfFloatKernel;
  kernel_t updatePCGHalfKernel;
  kernel_t updatePCGHalfFloatKernel;
  kernel_t scaleHalfKernel;
  kernel_t scaleP2hHalfKernel;
  kernel_t zamxHalfKernel;
  kernel_t zamx_Half2_D_Kernel;
  kernel_t zamx_Half2_F_Kernel;
  kernel_t updateCheby4HalfKernel;
  kernel_t updateCheby5HalfKernel;
  kernel_t updateCheby1HalfKernel;
  kernel_t halfAxTrilinearGatherSmoothKernel;
  kernel_t halfAxTrilinearGatherKernel;
  kernel_t halfFloatAxTrilinearGatherKernel;
  
  kernel_t axpySetKernel;
  kernel_t axpySetncKernel;
  kernel_t updatePCGKernel;
  kernel_t p2dInnerProdKernel;
  kernel_t updatePCGncKernel;
  kernel_t p2dInnerProdncKernel;
  kernel_t updatePCGDoubleKernel;

  kernel_t zamx_H_F_Kernel;
  kernel_t zamx_F_F_Kernel;
  kernel_t zamx_D_F_Kernel;
  kernel_t zamx_H_D_Kernel;
  kernel_t zamx_F_D_Kernel;
  kernel_t zamx_D_D_Kernel;

  kernel_t innerProd_H_F_Kernel;
  kernel_t innerProd_F_F_Kernel;
  kernel_t innerProd_D_F_Kernel;
  kernel_t innerProd_H_D_Kernel;
  kernel_t innerProd_F_D_Kernel;
  kernel_t innerProd_D_D_Kernel;
  kernel_t innerProd_Half2_D_Kernel;
  kernel_t innerProd_Half2_F_Kernel;

  kernel_t AxTrilinearGatherDot_H_F_Kernel;
  kernel_t AxTrilinearGatherDot_F_F_Kernel;
  kernel_t AxTrilinearGatherDot_D_F_Kernel;
  kernel_t AxTrilinearGatherDot_H_D_Kernel;
  kernel_t AxTrilinearGatherDot_F_D_Kernel;
  kernel_t AxTrilinearGatherDot_D_D_Kernel;
  
  kernel_t updatePCGDouble_H_Kernel;
  kernel_t updatePCGDouble_F_Kernel;
  kernel_t updatePCGDouble_D_Kernel;
  kernel_t updatePCGFloat_H_Kernel;
  kernel_t updatePCGFloat_F_Kernel;
  kernel_t updatePCGFloat_D_Kernel;

  kernel_t axpy_H_Kernel;
  kernel_t axpy_Half2_Kernel;
  kernel_t axpy_F_Kernel;
  kernel_t axpy_F2_Kernel;
  kernel_t axpy_D_Kernel;

  kernel_t p2dInnerProd_F_F_Kernel;
  kernel_t p2dInnerProd_F_D_Kernel;
  kernel_t p2dInnerProd_D_F_Kernel;
  kernel_t p2dInnerProd_D_D_Kernel;

  kernel_t AxGatherDot_F_F_Kernel;
  kernel_t AxGatherDot_F_F_F_Kernel;
  kernel_t AxGatherDot_F_D_F_Kernel;
  kernel_t AxGatherDot_F_D_Kernel;
  kernel_t AxGatherDot_D_F_Kernel;
  kernel_t AxGatherDot_D_D_Kernel;

  kernel_t AxGatherDotHalfKernel;
  
  kernel_t axpySet_F_F_F_Kernel;
  kernel_t axpySet_F_F_D_Kernel;
  kernel_t axpySet_F_D_F_Kernel;
  kernel_t axpySet_F_D_D_Kernel;
  kernel_t axpySet_D_F_F_Kernel;
  kernel_t axpySet_D_F_D_Kernel;
  kernel_t axpySet_D_D_F_Kernel;
  kernel_t axpySet_D_D_D_Kernel;

  kernel_t updatePCG_F_F_F_F_Kernel;
  kernel_t updatePCG_F_F_D_F_Kernel;
  kernel_t updatePCG_F_D_F_F_Kernel;
  kernel_t updatePCG_F_D_D_F_Kernel;
  kernel_t updatePCG_D_F_F_F_Kernel;
  kernel_t updatePCG_D_F_D_F_Kernel;
  kernel_t updatePCG_D_D_F_F_Kernel;
  kernel_t updatePCG_D_D_D_F_Kernel;
  kernel_t updatePCG_F_F_F_D_Kernel;
  kernel_t updatePCG_F_F_D_D_Kernel;
  kernel_t updatePCG_F_D_F_D_Kernel;
  kernel_t updatePCG_F_D_D_D_Kernel;
  kernel_t updatePCG_D_F_F_D_Kernel;
  kernel_t updatePCG_D_F_D_D_Kernel;
  kernel_t updatePCG_D_D_F_D_Kernel;
  kernel_t updatePCG_D_D_D_D_Kernel;

  kernel_t cubatureH1ErrorKernel;
  kernel_t cubatureH1L2ErrorKernel;
  kernel_t rhsBCKernel;
  kernel_t addBCKernel;
  
  elliptic_t() = default;
  elliptic_t(platform_t &_platform, mesh_t &_mesh,
              settings_t& _settings, dfloat _lambda,
              const int _NBCTypes, const memory<int> _BCType) {
    Setup(_platform, _mesh, _settings, _lambda, _NBCTypes, _BCType);
  }

  //setup
  void Setup(platform_t& _platform, mesh_t& _mesh,
             settings_t& _settings, dfloat _lambda,
             const int _NBCTypes, const memory<int> _BCType);

  void BoundarySetup();

  void Run();

  dfloat ComputeError(deviceMemory<dfloat>& o_x);
  
  int Solve(linearSolver_t<dfloat>& linearSolver, deviceMemory<dfloat> &o_x, deviceMemory<dfloat> &o_r,
            const dfloat tol, const int MAXIT, const int verbose);

  void PlotFields(memory<dfloat>& Q, std::string fileName);

  void Operator(deviceMemory<double>& o_q, deviceMemory<double>& o_Aq, memory<bool> Rvt, const int offsetRvt);
  void Operator(deviceMemory<double>& o_q, deviceMemory<double>& o_Aq);
  void Operator(deviceMemory<float>& o_q, deviceMemory<float>& o_Aq);

  void BuildOperatorMatrixIpdg(parAlmond::parCOO& A);
  void BuildOperatorMatrixContinuous(parAlmond::parCOO& A);

  void BuildOperatorMatrixContinuousTri2D(parAlmond::parCOO& A);
  void BuildOperatorMatrixContinuousTri3D(parAlmond::parCOO& A);
  void BuildOperatorMatrixContinuousQuad2D(parAlmond::parCOO& A);
  void BuildOperatorMatrixContinuousQuad3D(parAlmond::parCOO& A);
  void BuildOperatorMatrixContinuousTet3D(parAlmond::parCOO& A);
  void BuildOperatorMatrixContinuousHex3D(parAlmond::parCOO& A);

  void BuildOperatorMatrixIpdgTri2D(parAlmond::parCOO& A);
  void BuildOperatorMatrixIpdgTri3D(parAlmond::parCOO& A);
  void BuildOperatorMatrixIpdgQuad2D(parAlmond::parCOO& A);
  void BuildOperatorMatrixIpdgQuad3D(parAlmond::parCOO& A);
  void BuildOperatorMatrixIpdgTet3D(parAlmond::parCOO& A);
  void BuildOperatorMatrixIpdgHex3D(parAlmond::parCOO& A);

  void BuildOperatorDiagonal(memory<dfloat>& diagA);

  void BuildOperatorDiagonalContinuousTri2D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalContinuousTri3D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalContinuousQuad2D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalContinuousQuad3D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalContinuousTet3D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalContinuousHex3D(memory<dfloat>& diagA);

  void BuildOperatorDiagonalIpdgTri2D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalIpdgTri3D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalIpdgQuad2D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalIpdgQuad3D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalIpdgTet3D(memory<dfloat>& diagA);
  void BuildOperatorDiagonalIpdgHex3D(memory<dfloat>& diagA);

  elliptic_t SetupNewDegree(mesh_t& meshF);

  elliptic_t SetupRingPatch(mesh_t& meshPatch);

  void ZeroMean(deviceMemory<double> &o_q);
  void ZeroMean(deviceMemory<float> &o_q);
};

#endif

