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

#define ellipticForcing3D(x, y, z, lambda, f) \
  { f = 0.; }

/* Homogeneous Dirichlet boundary condition   */
#define ellipticDirichletCondition3D(x,   \
                                     y,   \
                                     z,   \
                                     nx,  \
                                     ny,  \
                                     nz,  \
                                     uM,  \
                                     uxM, \
                                     uyM, \
                                     uzM, \
                                     uB,  \
                                     uxB, \
                                     uyB, \
                                     uzB) \
  {                                       \
    uB  = 0.f;                            \
    uxB = uxM;                            \
    uyB = uyM;                            \
    uzB = uzM;                            \
  }

/* Homogeneous Neumann boundary condition   */
#define ellipticNeumannCondition3D(                       \
    x, y, z, nx, ny, nz, uM, uxM, uyM, uzM, uB, uxB, uzB) \
  {                                                       \
    uB  = uM;                                             \
    uxB = 0.f;                                            \
    uyB = 0.f;                                            \
    uzB = 0.f;                                            \
  }

#define waveForcingFunction3D(x, y, z, sigma, omega, f) \
  {                                                     \
    /*    f = exp(-80.*(x*x+(y-.5)*(y-.5)));    */      \
    f = 0.;                                             \
  }

#define waveInitialConditionsFunction3D(t, x, y, z, d, p)  \
  {                                                        \
    d = 0.;                                                \
    p = exp(-80. * (x * x + (y - .5) * (y - .5) + z * z)); \
  }
