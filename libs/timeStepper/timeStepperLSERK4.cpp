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

#include "core.hpp"
#include "timeStepper.hpp"

namespace libp {

namespace TimeStepper {

lserk4::lserk4(dlong Nelements, dlong NhaloElements,
               int Np, int Nfields,
               platform_t& _platform, comm_t _comm):
  lserk4(Nelements, 0, NhaloElements,
         Np, Nfields, 0,
         _platform, _comm) {}

lserk4::lserk4(dlong Nelements, dlong NpmlElements, dlong NhaloElements,
               int Np, int Nfields, int Npmlfields,
               platform_t& _platform, comm_t _comm):
  timeStepperBase_t(Nelements, NpmlElements, NhaloElements,
                    Np, Nfields, Npmlfields,
                    _platform, _comm) {

  Nrk = 5;

  properties_t kernelInfo = platform.props(); //copy base occa properties from solver

  const int blocksize=256;

  kernelInfo["defines/" "p_blockSize"] = blocksize;

  updateKernel = platform.buildKernel(TIMESTEPPER_DIR "/okl/"
                                    "timeStepperLSERK4.okl",
                                    "lserk4Update",
                                    kernelInfo);

  // initialize LSERK4 time stepping coefficients
  dfloat _rka[5] = {0.0,
       -567301805773.0/1357537059087.0 ,
       -2404267990393.0/2016746695238.0 ,
       -3550918686646.0/2091501179385.0  ,
       -1275806237668.0/842570457699.0};
  dfloat _rkb[5] = { 1432997174477.0/9575080441755.0 ,
        5161836677717.0/13612068292357.0 ,
        1720146321549.0/2090206949498.0  ,
        3134564353537.0/4481467310338.0  ,
        2277821191437.0/14882151754819.0};
  // added one more for advanced time step
  dfloat _rkc[6] = {0.0  ,
       1432997174477.0/9575080441755.0 ,
       2526269341429.0/6820363962896.0 ,
       2006345519317.0/3224310063776.0 ,
       2802321613138.0/2924317926251.0 ,
       1.0};

  rka.malloc(Nrk);
  rkb.malloc(Nrk);
  rkc.malloc(Nrk+1);
  rka.copyFrom(_rka);
  rkb.copyFrom(_rkb);
  rkc.copyFrom(_rkc);
}

void lserk4::Run(solver_t& solver,
                 deviceMemory<dfloat> o_q,
                 std::optional<deviceMemory<dfloat>> o_pmlq,
                 dfloat start, dfloat end) {

  /*Pre-reserve memory pool space to avoid some unnecessary re-sizing*/
  platform.reserve<dfloat>(3 * N + 3 * Npml
                           + 6 * platform.memPoolAlignment<dfloat>());

  dfloat time = start;

  solver.Report(time,0);

  dfloat outputInterval=0.0;
  solver.settings.getSetting("OUTPUT INTERVAL", outputInterval);

  dfloat outputTime = time + outputInterval;

  int tstep=0;
  dfloat stepdt;
  while (time < end) {

    if (time<outputTime && time+dt>=outputTime) {
      //save current state
      deviceMemory<dfloat> o_saveq = platform.reserve<dfloat>(N);
      deviceMemory<dfloat> o_savepmlq  = platform.reserve<dfloat>(Npml);
      o_saveq.copyFrom(o_q, N, 0, properties_t("async", true));
      if (o_pmlq.has_value()) {
        o_savepmlq.copyFrom(o_pmlq.value(), Npml, 0, properties_t("async", true));
      }

      stepdt = outputTime-time;

      //take small time step
      Step(solver, o_q, o_pmlq, time, stepdt);

      //report state
      solver.Report(outputTime,tstep);

      //restore previous state
      o_q.copyFrom(o_saveq, N, 0, properties_t("async", true));
      if (o_pmlq.has_value()) {
        o_pmlq.value().copyFrom(o_savepmlq, Npml, 0, properties_t("async", true));
      }

      outputTime += outputInterval;
    }

    //check for final timestep
    if (time+dt > end){
      stepdt = end-time;
    } else {
      stepdt = dt;
    }

    Step(solver, o_q, o_pmlq, time, stepdt);
    time += stepdt;
    tstep++;

    if(!(tstep%100))
       std::cout << "Time: " << time << ", tstep: " << tstep << ", dt: " << dt << std::endl;
  }
}

void lserk4::Step(solver_t& solver,
                  deviceMemory<dfloat> o_q,
                  std::optional<deviceMemory<dfloat>> o_pmlq,
                  dfloat time, dfloat _dt) {

  deviceMemory<dfloat> o_resq = platform.reserve<dfloat>(N);
  deviceMemory<dfloat> o_rhsq = platform.reserve<dfloat>(N);
  deviceMemory<dfloat> o_respmlq = platform.reserve<dfloat>(Npml);
  deviceMemory<dfloat> o_rhspmlq = platform.reserve<dfloat>(Npml);

  // Low storage explicit Runge Kutta (5 stages, 4th order)
  for(int rk=0;rk<Nrk;++rk){

    dfloat currentTime = time + rkc[rk]*_dt;

    //evaluate ODE rhs = f(q,t)
    if (o_pmlq.has_value()) {
      solver.rhsf_pml(o_q, o_pmlq.value(), o_rhsq, o_rhspmlq, currentTime);
    } else {
      solver.rhsf(o_q, o_rhsq, currentTime);
    }

    // update solution using Runge-Kutta
    updateKernel(N, _dt, rka[rk], rkb[rk],
                 o_rhsq, o_resq, o_q);
    if (o_pmlq.has_value()) {
      updateKernel(Npml, _dt, rka[rk], rkb[rk],
                   o_rhspmlq, o_respmlq, o_pmlq.value());
    }
  }
}

} //namespace TimeStepper

} //namespace libp
