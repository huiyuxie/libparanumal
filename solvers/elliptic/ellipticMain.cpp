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

int main(int argc, char **argv){

  // start up MPI
  Comm::Init(argc, argv);

  LIBP_ABORT("Usage: ./ellipticMain setupfile", argc!=2);

  { /*Scope so everything is destructed before MPI_Finalize */
    comm_t comm(Comm::World().Dup());

    //create default settings
    platformSettings_t platformSettings(comm);
    meshSettings_t meshSettings(comm);
    ellipticSettings_t ellipticSettings(comm);
    ellipticAddRunSettings(ellipticSettings);

    //load settings from file
    ellipticSettings.parseFromFile(platformSettings, meshSettings,
                                   argv[1]);

    // set up platform
    platform_t platform(platformSettings);

    platformSettings.report();
    meshSettings.report();
    ellipticSettings.report();

    // set up mesh
    mesh_t mesh(platform, meshSettings, comm);

    dfloat lambda = 0.0;
    ellipticSettings.getSetting("LAMBDA", lambda);

    // Boundary Type translation. Just defaults.
    int NBCTypes = 3;
    memory<int> BCType(3);
    BCType[0] = 0;
    BCType[1] = 1;
    BCType[2] = 2;

    // set up elliptic solver
    elliptic_t elliptic(platform, mesh, ellipticSettings,
                        lambda, NBCTypes, BCType);

    // run
    elliptic.Run();
//    elliptic.WaveSolver();
  }

  // close down MPI
  Comm::Finalize();
  return LIBP_SUCCESS;
}
