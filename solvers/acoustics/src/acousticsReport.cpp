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

#include "acoustics.hpp"

void acoustics_t::Report(dfloat time, int tstep){

  static int frame=0;

  //compute q.M*q
  dlong Nentries = mesh.Nelements*mesh.Np*Nfields;
  deviceMemory<dfloat> o_Mq = platform.reserve<dfloat>(Nentries);
  mesh.MassMatrixApply(o_q, o_Mq);

  dfloat norm2 = sqrt(platform.linAlg().innerProd(Nentries, o_q, o_Mq, mesh.comm));

  if(mesh.rank==0)
    printf("%5.2f (%d), %5.2f (time, timestep, norm)\n", time, tstep, norm2);

#if 0
  int errorStep = 1000;
  settings.getSetting("OUTPUT ERROR INTERVAL", errorStep);
  if(errorStep>0 && tstep>0){
    if((tstep%errorStep)==0){
      ReportError(time, 0, 0, 0, o_q);
    }
  }
#endif
  
  if (settings.compareSetting("OUTPUT TO FILE","TRUE")) {

    // copy data back to host
    o_q.copyTo(q);

    // output field files
    std::string name;
    settings.getSetting("OUTPUT FILE NAME", name);
    char fname[BUFSIZ];
    sprintf(fname, "%s_%04d_%04d.vtu", name.c_str(), mesh.rank, frame++);

    PlotFields(q, std::string(fname));
  }
}
