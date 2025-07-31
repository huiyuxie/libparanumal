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

#ifndef WAVE_HPP
#define WAVE_HPP

#include "core.hpp"
#include "elliptic.hpp"
#include "initialGuess.hpp"
#include "linAlg.hpp"
#include "mesh.hpp"
#include "platform.hpp"
#include "solver.hpp"
#include "timeStepper.hpp"

#define DWAVE LIBP_DIR "/solvers/wave/"

using namespace libp;

class waveSettings_t : public settings_t {
public:
  waveSettings_t(comm_t &_comm);
  void report();
  void parseFromFile(platformSettings_t &platformSettings,
                     meshSettings_t &meshSettings, const std::string filename);

  ellipticSettings_t extractEllipticSettings();
};

class wave_t : public solver_t {
public:
  mesh_t mesh;

  ellipticSettings_t ellipticSettings;
  elliptic_t elliptic;

  linearSolver_t<dfloat> linearSolver;

  int Nfields = 1;
  int disc_c0 = 0;

  int Niter;
  int iostep;
  int maxIter;
  int verbose;
  dfloat tol;

  dlong Nall;
  dlong NglobalDofs;

  dfloat ellipticTOL;
  dfloat tau;

  int Nstages;
  int embedded;
  dfloat gamma;
  dfloat invGamma;
  dfloat invGammaDt;
  dfloat invDt;
  int Nsteps;
  dfloat dt;
  dfloat startTime;
  dfloat finalTime;

  dfloat lambdaSolve;
  dfloat omega;
  dfloat sigma;

  /* flux source info */
  dfloat xsource;
  dfloat ysource;
  dfloat zsource;
  dfloat fsource;

  memory<int> patchLabels;
  deviceMemory<int> o_EToPatch;

  linAlgMatrix_t<dfloat> alpha, beta, betahat, esdirkC, alphatilde, gammatilde;

  deviceMemory<dfloat> o_alphatilde;
  deviceMemory<dfloat> o_gammatilde;
  deviceMemory<dfloat> o_betatilde;
  deviceMemory<dfloat> o_betahattilde;
  deviceMemory<dfloat> o_alpha;
  deviceMemory<dfloat> o_beta;
  deviceMemory<dfloat> o_betaAlpha;
  deviceMemory<dfloat> o_betahatAlpha;
  deviceMemory<dfloat> o_betahat;
  deviceMemory<dfloat> o_gamma;
  deviceMemory<dfloat> o_esdirkC;

  memory<dfloat> DL;
  memory<dfloat> PL;
  memory<dfloat> DrhsL;
  memory<dfloat> DhatL;
  memory<dfloat> PhatL;

  memory<dfloat> WJ;
  memory<dfloat> invWJ;

  deviceMemory<dfloat> o_DL;
  deviceMemory<dfloat> o_PL;
  deviceMemory<dfloat> o_DrhsL;
  deviceMemory<dfloat> o_DhatL;
  deviceMemory<dfloat> o_PhatL;

  deviceMemory<dfloat> o_DtildeL;
  deviceMemory<dfloat> o_Dtilde;

  deviceMemory<dfloat> o_Drhs;
  deviceMemory<dfloat> o_scratch1;
  deviceMemory<dfloat> o_scratch2;
  deviceMemory<dfloat> o_scratch1L;
  deviceMemory<dfloat> o_scratch2L;

  deviceMemory<dfloat> o_FL;
  deviceMemory<dfloat> o_filtPL;

  deviceMemory<dfloat> o_invMM;
  deviceMemory<dfloat> o_MM;
  deviceMemory<dfloat> o_invWJ;
  deviceMemory<dfloat> o_WJ;

  stoppingCriteria_t<dfloat> *stoppingCriteria = NULL;
  ellipticStoppingCriteria<dfloat> *esc = NULL;

  kernel_t waveStageUpdateKernel;
  kernel_t waveCombineKernel;
  kernel_t waveErrorEstimateKernel;
  kernel_t waveStepInitializeKernel;
  kernel_t waveStepFinalizeKernel;
  kernel_t waveStageInitializeKernel;
  kernel_t waveStageFinalizeKernel;
  kernel_t waveInitialConditionsKernel;
  kernel_t waveForcingKernel;

  kernel_t waveStepInitializeKernelV2;
  kernel_t waveStageInitializeKernelV2;
  kernel_t waveStageFinalizeKernelV2;
  kernel_t waveStageRHSKernelV2;

  kernel_t waveSurfaceSourceKernel;

  wave_t() = default;
  wave_t(platform_t &_platform, mesh_t &_mesh, waveSettings_t &_settings) {
    Setup(_platform, _mesh, _settings);
  }

  // setup
  void Setup(platform_t &_platform, mesh_t &_mesh, waveSettings_t &_settings);

  void Solve(deviceMemory<dfloat> &_o_DL, deviceMemory<dfloat> &_o_PL,
             deviceMemory<dfloat> &_o_FL);

  // skip v2 for now
  //  void SolveV2(deviceMemory<dfloat> &_o_DL,
  //               deviceMemory<dfloat> &_o_PL,
  //               deviceMemory<dfloat> &_o_FL);

  void Operator(deviceMemory<dfloat> &inPL, deviceMemory<dfloat> &outPL);

  void waveHoltz(deviceMemory<dfloat> &o_qL);
  // skip v2 for now
  //  void waveHoltzV2(deviceMemory<dfloat> &o_qL);

  void Run();

  void Report(dfloat time, int tstep);

  void ReportError(dfloat t, dfloat elapsedTime, int iterations,
                   deviceMemory<dfloat> &DL, deviceMemory<dfloat> &PL);

  void PlotFields(libp::memory<dfloat> &DL, libp::memory<dfloat> &PL,
                  std::string fileName);
};

#endif
