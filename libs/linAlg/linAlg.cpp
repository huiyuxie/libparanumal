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

#include "linAlg.hpp"
#include "platform.hpp"

#define p_Nwarps 8
#define p_warpSize 32
#define p_blockSize (p_Nwarps * p_warpSize)

namespace libp {

/*********************/
/* vector operations */
/*********************/

// o_a[n] = alpha
template <>
void linAlg_t::set(const dlong N, const float alpha, deviceMemory<float> o_a) {
  setKernelFloat(N, alpha, o_a, 0);
}

template <>
void linAlg_t::set(const dlong          N,
                   const double         alpha,
                   deviceMemory<double> o_a) {
  setKernelDouble(N, alpha, o_a, 0);
}

// o_a[n] = alpha
template <>
void linAlg_t::set(const dlong         N,
                   const float         alpha,
                   deviceMemory<float> o_a,
                   const int           dir) {
  setKernelFloat(N, alpha, o_a, dir);
}

template <>
void linAlg_t::set(const dlong          N,
                   const double         alpha,
                   deviceMemory<double> o_a,
                   const int            dir) {
  setKernelDouble(N, alpha, o_a, dir);
}
template <>
void linAlg_t::setnc(const dlong         N,
                     const float         alpha,
                     deviceMemory<float> o_a,
                     const int           dir) {
  setncKernelFloat(N, alpha, o_a, dir);
}

template <>
void linAlg_t::setnc(const dlong          N,
                     const double         alpha,
                     deviceMemory<double> o_a,
                     const int            dir) {
  setncKernelDouble(N, alpha, o_a, dir);
}

// o_a[n] += alpha
template <>
void linAlg_t::add(const dlong N, const float alpha, deviceMemory<float> o_a) {
  addKernelFloat(N, alpha, o_a);
}
template <>
void linAlg_t::add(const dlong          N,
                   const double         alpha,
                   deviceMemory<double> o_a) {
  addKernelDouble(N, alpha, o_a);
}

// o_a[n] *= alpha
template <>
void linAlg_t::scale(const dlong         N,
                     const float         alpha,
                     deviceMemory<float> o_a) {
  scaleKernelFloat(N, alpha, o_a);
}
template <>
void linAlg_t::scale(const dlong          N,
                     const double         alpha,
                     deviceMemory<double> o_a) {
  scaleKernelDouble(N, alpha, o_a);
}

// o_y[n] = beta*o_y[n] + alpha*o_x[n]
template <>
void linAlg_t::axpy(const dlong         N,
                    const float         alpha,
                    deviceMemory<float> o_x,
                    const float         beta,
                    deviceMemory<float> o_y,
                    int                 reverse) {
  axpyKernelFloat(N, alpha, o_x, beta, o_y, reverse);
}
template <>
void linAlg_t::axpy(const dlong          N,
                    const double         alpha,
                    deviceMemory<double> o_x,
                    const double         beta,
                    deviceMemory<double> o_y,
                    int                  reverse) {
  axpyKernelDouble(N, alpha, o_x, beta, o_y, reverse);
}

// o_y[n] = beta*o_y[n] + alpha*o_x[n], with non-cache load/store
template <>
void linAlg_t::axpync(const dlong         N,
                      const float         alpha,
                      deviceMemory<float> o_x,
                      const float         beta,
                      deviceMemory<float> o_y,
                      int                 reverse) {
  axpyncKernelFloat(N, alpha, o_x, beta, o_y, reverse);
}
template <>
void linAlg_t::axpync(const dlong          N,
                      const double         alpha,
                      deviceMemory<double> o_x,
                      const double         beta,
                      deviceMemory<double> o_y,
                      int                  reverse) {
  axpyncKernelDouble(N, alpha, o_x, beta, o_y, reverse);
}

// o_y[n] = beta*o_y[n] + alpha*o_x[n]
template <>
void linAlg_t::axpy(const dlong         N,
                    const float         alpha,
                    deviceMemory<float> o_x,
                    const float         beta,
                    deviceMemory<float> o_y) {
  int reverse = 0;
  axpyKernelFloat(N, alpha, o_x, beta, o_y, reverse);
}
template <>
void linAlg_t::axpy(const dlong          N,
                    const double         alpha,
                    deviceMemory<double> o_x,
                    const double         beta,
                    deviceMemory<double> o_y) {
  int reverse = 0;
  axpyKernelDouble(N, alpha, o_x, beta, o_y, reverse);
}

// o_z[n] = beta*o_y[n] + alpha*o_x[n]
template <>
void linAlg_t::zaxpy(const dlong         N,
                     const float         alpha,
                     deviceMemory<float> o_x,
                     const float         beta,
                     deviceMemory<float> o_y,
                     deviceMemory<float> o_z) {
  zaxpyKernelFloat(N, alpha, o_x, beta, o_y, o_z);
}
template <>
void linAlg_t::zaxpy(const dlong          N,
                     const double         alpha,
                     deviceMemory<double> o_x,
                     const double         beta,
                     deviceMemory<double> o_y,
                     deviceMemory<double> o_z) {
  zaxpyKernelDouble(N, alpha, o_x, beta, o_y, o_z);
}

// o_x[n] = alpha*o_a[n]*o_x[n]
template <>
void linAlg_t::amx(const dlong         N,
                   const float         alpha,
                   deviceMemory<float> o_a,
                   deviceMemory<float> o_x) {
  amxKernelFloat(N, alpha, o_a, o_x, 0);
}
template <>
void linAlg_t::amx(const dlong          N,
                   const double         alpha,
                   deviceMemory<double> o_a,
                   deviceMemory<double> o_x) {
  amxKernelDouble(N, alpha, o_a, o_x, 0);
}
// o_x[n] = alpha*o_a[n]*o_x[n]
template <>
void linAlg_t::amx(const dlong         N,
                   const float         alpha,
                   deviceMemory<float> o_a,
                   deviceMemory<float> o_x,
                   int                 dir) {
  amxKernelFloat(N, alpha, o_a, o_x, dir);
}
template <>
void linAlg_t::amx(const dlong          N,
                   const double         alpha,
                   deviceMemory<double> o_a,
                   deviceMemory<double> o_x,
                   int                  dir) {
  amxKernelDouble(N, alpha, o_a, o_x, dir);
}
// o_x[n] = alpha*o_a[n]*o_x[n]
template <>
void linAlg_t::amxnc(const dlong         N,
                     const float         alpha,
                     deviceMemory<float> o_a,
                     deviceMemory<float> o_x,
                     int                 dir) {
  amxncKernelFloat(N, alpha, o_a, o_x, dir);
}
template <>
void linAlg_t::amxnc(const dlong          N,
                     const double         alpha,
                     deviceMemory<double> o_a,
                     deviceMemory<double> o_x,
                     int                  dir) {
  amxncKernelDouble(N, alpha, o_a, o_x, dir);
}

// o_y[n] = alpha*o_a[n]*o_x[n] + beta*o_y[n]
template <>
void linAlg_t::amxpy(const dlong         N,
                     const float         alpha,
                     deviceMemory<float> o_a,
                     deviceMemory<float> o_x,
                     const float         beta,
                     deviceMemory<float> o_y) {
  amxpyKernelFloat(N, alpha, o_a, o_x, beta, o_y, 0);
}
template <>
void linAlg_t::amxpy(const dlong          N,
                     const double         alpha,
                     deviceMemory<double> o_a,
                     deviceMemory<double> o_x,
                     const double         beta,
                     deviceMemory<double> o_y) {
  amxpyKernelDouble(N, alpha, o_a, o_x, beta, o_y, 0);
}

// o_y[n] = alpha*o_a[n]*o_x[n] + beta*o_y[n], can reverse direction
template <>
void linAlg_t::amxpy(const dlong         N,
                     const float         alpha,
                     deviceMemory<float> o_a,
                     deviceMemory<float> o_x,
                     const float         beta,
                     deviceMemory<float> o_y,
                     int                 reverse) {
  amxpyKernelFloat(N, alpha, o_a, o_x, beta, o_y, reverse);
}
template <>
void linAlg_t::amxpy(const dlong          N,
                     const double         alpha,
                     deviceMemory<double> o_a,
                     deviceMemory<double> o_x,
                     const double         beta,
                     deviceMemory<double> o_y,
                     int                  reverse) {
  amxpyKernelDouble(N, alpha, o_a, o_x, beta, o_y, reverse);
}

// o_y[n] = alpha*o_a[n]*o_x[n] + beta*o_y[n], can reverse direction, with
// non-cache load/store
template <>
void linAlg_t::amxpync(const dlong         N,
                       const float         alpha,
                       deviceMemory<float> o_a,
                       deviceMemory<float> o_x,
                       const float         beta,
                       deviceMemory<float> o_y,
                       int                 reverse) {
  amxpyncKernelFloat(N, alpha, o_a, o_x, beta, o_y, reverse);
}
template <>
void linAlg_t::amxpync(const dlong          N,
                       const double         alpha,
                       deviceMemory<double> o_a,
                       deviceMemory<double> o_x,
                       const double         beta,
                       deviceMemory<double> o_y,
                       int                  reverse) {
  amxpyncKernelDouble(N, alpha, o_a, o_x, beta, o_y, reverse);
}

// o_z[n] = alpha*o_a[n]*o_x[n] + beta*o_y[n]
template <>
void linAlg_t::zamxpy(const dlong         N,
                      const float         alpha,
                      deviceMemory<float> o_a,
                      deviceMemory<float> o_x,
                      const float         beta,
                      deviceMemory<float> o_y,
                      deviceMemory<float> o_z) {
  zamxpyKernelFloat(N, alpha, o_a, o_x, beta, o_y, o_z);
}
template <>
void linAlg_t::zamxpy(const dlong          N,
                      const double         alpha,
                      deviceMemory<double> o_a,
                      deviceMemory<double> o_x,
                      const double         beta,
                      deviceMemory<double> o_y,
                      deviceMemory<double> o_z) {
  zamxpyKernelDouble(N, alpha, o_a, o_x, beta, o_y, o_z);
}

// o_x[n] = alpha*o_x[n]/o_a[n]
template <>
void linAlg_t::adx(const dlong         N,
                   const float         alpha,
                   deviceMemory<float> o_a,
                   deviceMemory<float> o_x) {
  adxKernelFloat(N, alpha, o_a, o_x);
}
template <>
void linAlg_t::adx(const dlong          N,
                   const double         alpha,
                   deviceMemory<double> o_a,
                   deviceMemory<double> o_x) {
  adxKernelDouble(N, alpha, o_a, o_x);
}

// o_y[n] = alpha*o_x[n]/o_a[n] + beta*o_y[n]
template <>
void linAlg_t::adxpy(const dlong         N,
                     const float         alpha,
                     deviceMemory<float> o_a,
                     deviceMemory<float> o_x,
                     const float         beta,
                     deviceMemory<float> o_y) {
  adxpyKernelFloat(N, alpha, o_a, o_x, beta, o_y);
}
template <>
void linAlg_t::adxpy(const dlong          N,
                     const double         alpha,
                     deviceMemory<double> o_a,
                     deviceMemory<double> o_x,
                     const double         beta,
                     deviceMemory<double> o_y) {
  adxpyKernelDouble(N, alpha, o_a, o_x, beta, o_y);
}

// o_z[n] = alpha*o_x[n]/o_a[n] + beta*o_y[n]
template <>
void linAlg_t::zadxpy(const dlong         N,
                      const float         alpha,
                      deviceMemory<float> o_a,
                      deviceMemory<float> o_x,
                      const float         beta,
                      deviceMemory<float> o_y,
                      deviceMemory<float> o_z) {
  zadxpyKernelFloat(N, alpha, o_a, o_x, beta, o_y, o_z);
}
template <>
void linAlg_t::zadxpy(const dlong          N,
                      const double         alpha,
                      deviceMemory<double> o_a,
                      deviceMemory<double> o_x,
                      const double         beta,
                      deviceMemory<double> o_y,
                      deviceMemory<double> o_z) {
  zadxpyKernelDouble(N, alpha, o_a, o_x, beta, o_y, o_z);
}

// update Chebyshev
template <>
void linAlg_t::updateCheby1(const dlong         N,
                            const float         alpha,
                            const float         beta,
                            deviceMemory<float> o_x,
                            deviceMemory<float> o_S,
                            deviceMemory<float> o_y,
                            int                 reverse) {
  updateCheby1KernelFloat(N, alpha, beta, o_x, o_S, o_y, reverse);
}
template <>
void linAlg_t::updateCheby1nc(const dlong         N,
                              const float         alpha,
                              const float         beta,
                              deviceMemory<float> o_x,
                              deviceMemory<float> o_S,
                              deviceMemory<float> o_y,
                              int                 reverse) {
  updateCheby1ncKernelFloat(N, alpha, beta, o_x, o_S, o_y, reverse);
}

template <>
void linAlg_t::updateCheby2(const dlong         N,
                            const float         alpha,
                            deviceMemory<float> o_r,
                            deviceMemory<float> o_invDiagA,
                            deviceMemory<float> o_Ax,
                            deviceMemory<float> o_x0,
                            deviceMemory<float> o_x,
                            int                 reverse) {
  updateCheby2KernelFloat(N, alpha, o_r, o_invDiagA, o_Ax, o_x0, o_x, reverse);
}
template <>
void linAlg_t::updateCheby2nc(const dlong         N,
                              const float         alpha,
                              deviceMemory<float> o_r,
                              deviceMemory<float> o_invDiagA,
                              deviceMemory<float> o_Ax,
                              deviceMemory<float> o_x0,
                              deviceMemory<float> o_x,
                              int                 reverse) {
  updateCheby2ncKernelFloat(
      N, alpha, o_r, o_invDiagA, o_Ax, o_x0, o_x, reverse);
}

template <>
void linAlg_t::updateCheby3(const dlong         N,
                            const float         a1,
                            const float         a0,
                            const float         d0,
                            deviceMemory<float> o_Ax,
                            deviceMemory<float> o_invDiagA,
                            deviceMemory<float> o_r,
                            deviceMemory<float> o_x0,
                            deviceMemory<float> o_x1,
                            deviceMemory<float> o_x2,
                            int                 reverse) {
  updateCheby3KernelFloat(
      N, a1, a0, d0, o_Ax, o_invDiagA, o_r, o_x0, o_x1, o_x2, reverse);
}
template <>
void linAlg_t::updateCheby3nc(const dlong         N,
                              const float         a1,
                              const float         a0,
                              const float         d0,
                              deviceMemory<float> o_Ax,
                              deviceMemory<float> o_invDiagA,
                              deviceMemory<float> o_r,
                              deviceMemory<float> o_x0,
                              deviceMemory<float> o_x1,
                              deviceMemory<float> o_x2,
                              int                 reverse) {
  updateCheby3ncKernelFloat(
      N, a1, a0, d0, o_Ax, o_invDiagA, o_r, o_x0, o_x1, o_x2, reverse);
}
template <>
void linAlg_t::updateCheby4(const dlong         N,
                            const float         c1,
                            const float         c2,
                            const float         c3,
                            deviceMemory<float> o_x1,
                            deviceMemory<float> o_x2,
                            deviceMemory<float> o_x3,
                            deviceMemory<float> o_y,
                            int                 reverse) {
  updateCheby4KernelFloat(N, c1, c2, c3, o_x1, o_x2, o_x3, o_y, reverse);
}
template <>
void linAlg_t::updateCheby4nc(const dlong         N,
                              const float         c1,
                              const float         c2,
                              const float         c3,
                              deviceMemory<float> o_x1,
                              deviceMemory<float> o_x2,
                              deviceMemory<float> o_x3,
                              deviceMemory<float> o_y,
                              int                 reverse) {
  updateCheby4ncKernelFloat(N, c1, c2, c3, o_x1, o_x2, o_x3, o_y, reverse);
}
template <>
void linAlg_t::updateCheby5(const dlong         N,
                            const float         c1,
                            const float         c2,
                            const float         c3,
                            deviceMemory<float> o_x0,
                            deviceMemory<float> o_x1,
                            deviceMemory<float> o_x2,
                            deviceMemory<float> o_x3,
                            deviceMemory<float> o_y,
                            int                 reverse) {
  updateCheby5KernelFloat(N, c1, c2, c3, o_x0, o_x1, o_x2, o_x3, o_y, reverse);
}
template <>
void linAlg_t::updateCheby5nc(const dlong         N,
                              const float         c1,
                              const float         c2,
                              const float         c3,
                              deviceMemory<float> o_x0,
                              deviceMemory<float> o_x1,
                              deviceMemory<float> o_x2,
                              deviceMemory<float> o_x3,
                              deviceMemory<float> o_y,
                              int                 reverse) {
  updateCheby5ncKernelFloat(
      N, c1, c2, c3, o_x0, o_x1, o_x2, o_x3, o_y, reverse);
}

// update Chebyshev
template <>
void linAlg_t::updateCheby1(const dlong          N,
                            const double         alpha,
                            const double         beta,
                            deviceMemory<double> o_x,
                            deviceMemory<double> o_S,
                            deviceMemory<double> o_y,
                            int                  reverse) {
  updateCheby1KernelDouble(N, alpha, beta, o_x, o_S, o_y, reverse);
}
template <>
void linAlg_t::updateCheby1nc(const dlong          N,
                              const double         alpha,
                              const double         beta,
                              deviceMemory<double> o_x,
                              deviceMemory<double> o_S,
                              deviceMemory<double> o_y,
                              int                  reverse) {
  updateCheby1ncKernelDouble(N, alpha, beta, o_x, o_S, o_y, reverse);
}

template <>
void linAlg_t::updateCheby2(const dlong          N,
                            const double         alpha,
                            deviceMemory<double> o_r,
                            deviceMemory<double> o_invDiagA,
                            deviceMemory<double> o_Ax,
                            deviceMemory<double> o_x0,
                            deviceMemory<double> o_x,
                            int                  reverse) {
  updateCheby2KernelDouble(N, alpha, o_r, o_invDiagA, o_Ax, o_x0, o_x, reverse);
}
template <>
void linAlg_t::updateCheby2nc(const dlong          N,
                              const double         alpha,
                              deviceMemory<double> o_r,
                              deviceMemory<double> o_invDiagA,
                              deviceMemory<double> o_Ax,
                              deviceMemory<double> o_x0,
                              deviceMemory<double> o_x,
                              int                  reverse) {
  updateCheby2ncKernelDouble(
      N, alpha, o_r, o_invDiagA, o_Ax, o_x0, o_x, reverse);
}

template <>
void linAlg_t::updateCheby3(const dlong          N,
                            const double         a1,
                            const double         a0,
                            const double         d0,
                            deviceMemory<double> o_Ax,
                            deviceMemory<double> o_invDiagA,
                            deviceMemory<double> o_r,
                            deviceMemory<double> o_x0,
                            deviceMemory<double> o_x1,
                            deviceMemory<double> o_x2,
                            int                  reverse) {
  updateCheby3KernelDouble(
      N, a1, a0, d0, o_Ax, o_invDiagA, o_r, o_x0, o_x1, o_x2, reverse);
}
template <>
void linAlg_t::updateCheby3nc(const dlong          N,
                              const double         a1,
                              const double         a0,
                              const double         d0,
                              deviceMemory<double> o_Ax,
                              deviceMemory<double> o_invDiagA,
                              deviceMemory<double> o_r,
                              deviceMemory<double> o_x0,
                              deviceMemory<double> o_x1,
                              deviceMemory<double> o_x2,
                              int                  reverse) {
  updateCheby3ncKernelDouble(
      N, a1, a0, d0, o_Ax, o_invDiagA, o_r, o_x0, o_x1, o_x2, reverse);
}
template <>
void linAlg_t::updateCheby4(const dlong          N,
                            const double         c1,
                            const double         c2,
                            const double         c3,
                            deviceMemory<double> o_x1,
                            deviceMemory<double> o_x2,
                            deviceMemory<double> o_x3,
                            deviceMemory<double> o_y,
                            int                  reverse) {
  updateCheby4KernelDouble(N, c1, c2, c3, o_x1, o_x2, o_x3, o_y, reverse);
}
template <>
void linAlg_t::updateCheby4nc(const dlong          N,
                              const double         c1,
                              const double         c2,
                              const double         c3,
                              deviceMemory<double> o_x1,
                              deviceMemory<double> o_x2,
                              deviceMemory<double> o_x3,
                              deviceMemory<double> o_y,
                              int                  reverse) {
  updateCheby4ncKernelDouble(N, c1, c2, c3, o_x1, o_x2, o_x3, o_y, reverse);
}
template <>
void linAlg_t::updateCheby5(const dlong          N,
                            const double         c1,
                            const double         c2,
                            const double         c3,
                            deviceMemory<double> o_x0,
                            deviceMemory<double> o_x1,
                            deviceMemory<double> o_x2,
                            deviceMemory<double> o_x3,
                            deviceMemory<double> o_y,
                            int                  reverse) {
  updateCheby5KernelDouble(N, c1, c2, c3, o_x0, o_x1, o_x2, o_x3, o_y, reverse);
}
template <>
void linAlg_t::updateCheby5nc(const dlong          N,
                              const double         c1,
                              const double         c2,
                              const double         c3,
                              deviceMemory<double> o_x0,
                              deviceMemory<double> o_x1,
                              deviceMemory<double> o_x2,
                              deviceMemory<double> o_x3,
                              deviceMemory<double> o_y,
                              int                  reverse) {
  updateCheby5ncKernelDouble(
      N, c1, c2, c3, o_x0, o_x1, o_x2, o_x3, o_y, reverse);
}

// \min o_a
template <typename T>
T linAlg_t::min(const dlong N, deviceMemory<T> o_a, comm_t comm) {
  int Nblock = (N + p_blockSize - 1) / p_blockSize;
  Nblock     = (Nblock > p_blockSize) ? p_blockSize
                                      : Nblock; // limit to p_blockSize entries

  // pinned scratch buffer
  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  if constexpr(std::is_same_v<T, float>) {
    minKernelFloat(Nblock, N, o_a, o_scratch);
  } else {
    minKernelDouble(Nblock, N, o_a, o_scratch);
  }

  T globalmin;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globalmin = h_scratch[0];
  } else {
    globalmin = std::numeric_limits<T>::max();
  }

  comm.Allreduce(globalmin, Comm::Min);

  return globalmin;
}

template float linAlg_t::min(const dlong         N,
                             deviceMemory<float> o_a,
                             comm_t              comm);

template double linAlg_t::min(const dlong          N,
                              deviceMemory<double> o_a,
                              comm_t               comm);

// \max o_a
template <typename T>
T linAlg_t::max(const dlong N, deviceMemory<T> o_a, comm_t comm) {
  int Nblock = (N + p_blockSize - 1) / p_blockSize;
  Nblock     = (Nblock > p_blockSize) ? p_blockSize
                                      : Nblock; // limit to p_blockSize entries

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  if constexpr(std::is_same_v<T, float>) {
    maxKernelFloat(Nblock, N, o_a, o_scratch);
  } else {
    maxKernelDouble(Nblock, N, o_a, o_scratch);
  }

  T globalmax;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globalmax = h_scratch[0];
  } else {
    globalmax = -std::numeric_limits<T>::max();
  }

  comm.Allreduce(globalmax, Comm::Max);

  return globalmax;
}

template float linAlg_t::max(const dlong         N,
                             deviceMemory<float> o_a,
                             comm_t              comm);

template double linAlg_t::max(const dlong          N,
                              deviceMemory<double> o_a,
                              comm_t               comm);

// \sum o_a
template <typename T>
T linAlg_t::sum(const dlong N, deviceMemory<T> o_a, comm_t comm) {
  int Nblock = (N + p_blockSize - 1) / p_blockSize;
  Nblock     = (Nblock > p_blockSize) ? p_blockSize
                                      : Nblock; // limit to p_blockSize entries

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  if constexpr(std::is_same_v<T, float>) {
    sumKernelFloat(Nblock, N, o_a, o_scratch);
  } else {
    sumKernelDouble(Nblock, N, o_a, o_scratch);
  }

  T globalsum;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globalsum = h_scratch[0];
  } else {
    globalsum = 0.0;
  }

  comm.Allreduce(globalsum, Comm::Sum);

  return globalsum;
}

template float linAlg_t::sum(const dlong         N,
                             deviceMemory<float> o_a,
                             comm_t              comm);

template double linAlg_t::sum(const dlong          N,
                              deviceMemory<double> o_a,
                              comm_t               comm);

// ||o_a||_2
template <typename T>
T linAlg_t::norm2(const dlong      N,
                  deviceMemory<T>& o_a,
                  int              reverse,
                  comm_t           comm) {
  dlong Nreads = 16;
  dlong Nblock = (N + p_blockSize * Nreads - 1) / (p_blockSize * Nreads);

  if(0) {
    printf("\n *** Nblock=%d\n", Nblock);

    pinnedMemory<T> h_tmp = platform->hostReserve<T>(N);
    h_tmp.copyFrom(o_a);
    T res = 0;
    for(int n = 0; n < N; ++n) {
      printf("h_tmp[%d]=%g\n", n, h_tmp[n]);
      res += h_tmp[n] * h_tmp[n];
    }
    printf("***host calculated norm: %g\n", sqrt(res));
  }

  memory<T>       h_scratch(Nblock);
  deviceMemory<T> o_scratch =
      platform->malloc<T>(Nblock); // platform->reserve<T>(Nblock);

  set<T>(Nblock, (dfloat)0.0, o_scratch);

  if constexpr(std::is_same_v<T, float>) {
    norm2KernelFloat(Nblock, N, Nreads, o_a, o_scratch, (int)reverse);
  } else {
    norm2KernelDouble(Nblock, N, Nreads, o_a, o_scratch, (int)reverse);
  }

  platform->device.finish();

  T globalnorm = 0;
  if(Nblock > 0) {
    o_scratch.copyTo(h_scratch);
    globalnorm = h_scratch[0];
  }

  comm.Allreduce(globalnorm, Comm::Sum);

  return sqrt(globalnorm);
}

template float linAlg_t::norm2(const dlong          N,
                               deviceMemory<float>& o_a,
                               int                  reverse,
                               comm_t               comm);

template double linAlg_t::norm2(const dlong           N,
                                deviceMemory<double>& o_a,
                                int                   reverse,
                                comm_t                comm);

// ||o_a||_2
template <typename T>
T linAlg_t::norm2(const dlong N, deviceMemory<T>& o_a, comm_t comm) {

  int   bsize  = 8 * 32;
  dlong Nread  = 1;
  dlong dir    = 0;
  int   Nblock = (N + Nread * bsize - 1) / (Nread * bsize);

  if(0) {
    printf("\n-- norm2: Nblock = %d\n", Nblock);

    T               res   = 0;
    pinnedMemory<T> h_tmp = platform->hostReserve<T>(N);
    for(int n = 0; n < N; ++n) { h_tmp[n] = 0; }
    h_tmp.copyFrom(o_a);
    platform->device.finish();
    for(int n = 0; n < N; ++n) { res += h_tmp[n] * h_tmp[n]; }
    printf("host calculated norm: %g\n", sqrt(res));
  }

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch =
      platform->malloc<T>(p_blockSize); // platform->reserve<T>(Nblock);

  set<T>(p_blockSize, (dfloat)0.0, o_scratch);

  if constexpr(std::is_same_v<T, float>) {
    norm2KernelFloat(Nblock, N, Nread, o_a, o_scratch, dir);
  } else {
    norm2KernelDouble(Nblock, N, Nread, o_a, o_scratch, dir);
  }

  platform->device.finish();

  T globalnorm = 0;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globalnorm = h_scratch[0];
  }

  comm.Allreduce(globalnorm, Comm::Sum);

  return sqrt(globalnorm);
}

template float linAlg_t::norm2(const dlong          N,
                               deviceMemory<float>& o_a,
                               comm_t               comm);

template double linAlg_t::norm2(const dlong           N,
                                deviceMemory<double>& o_a,
                                comm_t                comm);

// o_x.o_y, reverse direction
template <typename T>
T linAlg_t::innerProd(const dlong     N,
                      deviceMemory<T> o_x,
                      deviceMemory<T> o_y,
                      int             reverse,
                      comm_t          comm) {
  dlong Nreads = 16;
  dlong Nblock = (N + p_blockSize * Nreads - 1) / (p_blockSize * Nreads);
  // Nblock = (Nblock>p_blockSize) ? p_blockSize : Nblock; //limit to
  // p_blockSize entries

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  set<T>(p_blockSize, (dfloat)0.0, o_scratch);

  if constexpr(std::is_same_v<T, float>) {
    innerProdKernelFloat(Nblock, N, Nreads, o_x, o_y, o_scratch, reverse);
  } else {
    innerProdKernelDouble(Nblock, N, Nreads, o_x, o_y, o_scratch, reverse);
  }

  T globaldot;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globaldot = h_scratch[0];
  } else {
    globaldot = 0.0;
  }

  comm.Allreduce(globaldot, Comm::Sum);

  return globaldot;
}

template float linAlg_t::innerProd(const dlong         N,
                                   deviceMemory<float> o_x,
                                   deviceMemory<float> o_y,
                                   int                 reverse,
                                   comm_t              comm);

template double linAlg_t::innerProd(const dlong          N,
                                    deviceMemory<double> o_x,
                                    deviceMemory<double> o_y,
                                    int                  reverse,
                                    comm_t               comm);

// o_x.o_y, reverse direction + non-cache load/store
template <typename T>
T linAlg_t::innerProdnc(const dlong     N,
                        deviceMemory<T> o_x,
                        deviceMemory<T> o_y,
                        int             reverse,
                        comm_t          comm) {
  dlong Nreads = 16;
  dlong Nblock = (N + p_blockSize * Nreads - 1) / (p_blockSize * Nreads);
  // Nblock = (Nblock>p_blockSize) ? p_blockSize : Nblock; //limit to
  // p_blockSize entries

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  set<T>(p_blockSize, (dfloat)0.0, o_scratch);

  if constexpr(std::is_same_v<T, float>) {
    innerProdncKernelFloat(Nblock, N, Nreads, o_x, o_y, o_scratch, reverse);
  } else {
    innerProdncKernelDouble(Nblock, N, Nreads, o_x, o_y, o_scratch, reverse);
  }

  T globaldot;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globaldot = h_scratch[0];
  } else {
    globaldot = 0.0;
  }

  comm.Allreduce(globaldot, Comm::Sum);

  return globaldot;
}

template float linAlg_t::innerProdnc(const dlong         N,
                                     deviceMemory<float> o_x,
                                     deviceMemory<float> o_y,
                                     int                 reverse,
                                     comm_t              comm);

template double linAlg_t::innerProdnc(const dlong          N,
                                      deviceMemory<double> o_x,
                                      deviceMemory<double> o_y,
                                      int                  reverse,
                                      comm_t               comm);

// o_x.o_y, reverse direction
template <typename T>
T linAlg_t::innerProd(const dlong     N,
                      deviceMemory<T> o_x,
                      deviceMemory<T> o_y,
                      comm_t          comm) {
  dlong Nreads = 1;
  dlong Nblock = (N + p_blockSize * Nreads - 1) / (p_blockSize * Nreads);
  // Nblock = (Nblock>p_blockSize) ? p_blockSize : Nblock; //limit to
  // p_blockSize entries

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  set<T>(p_blockSize, (dfloat)0.0, o_scratch);

  if constexpr(std::is_same_v<T, float>) {
    innerProdKernelFloat(Nblock, N, Nreads, o_x, o_y, o_scratch, 0);
  } else {
    innerProdKernelDouble(Nblock, N, Nreads, o_x, o_y, o_scratch, 0);
  }

  T globaldot;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globaldot = h_scratch[0];
  } else {
    globaldot = 0.0;
  }

  comm.Allreduce(globaldot, Comm::Sum);

  return globaldot;
}

template float linAlg_t::innerProd(const dlong         N,
                                   deviceMemory<float> o_x,
                                   deviceMemory<float> o_y,
                                   comm_t              comm);

template double linAlg_t::innerProd(const dlong          N,
                                    deviceMemory<double> o_x,
                                    deviceMemory<double> o_y,
                                    comm_t               comm);

// o_w.o_x.o_y
template <typename T>
T linAlg_t::weightedInnerProd(const dlong     N,
                              deviceMemory<T> o_w,
                              deviceMemory<T> o_x,
                              deviceMemory<T> o_y,
                              comm_t          comm) {
  int Nblock = (N + p_blockSize - 1) / p_blockSize;
  Nblock     = (Nblock > p_blockSize) ? p_blockSize
                                      : Nblock; // limit to p_blockSize entries

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  if constexpr(std::is_same_v<T, float>) {
    weightedInnerProdKernelFloat(Nblock, N, o_w, o_x, o_y, o_scratch);
  } else {
    weightedInnerProdKernelDouble(Nblock, N, o_w, o_x, o_y, o_scratch);
  }

  T globaldot;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globaldot = h_scratch[0];
  } else {
    globaldot = 0.0;
  }

  comm.Allreduce(globaldot, Comm::Sum);

  return globaldot;
}

template float linAlg_t::weightedInnerProd(const dlong         N,
                                           deviceMemory<float> o_w,
                                           deviceMemory<float> o_x,
                                           deviceMemory<float> o_y,
                                           comm_t              comm);

template double linAlg_t::weightedInnerProd(const dlong          N,
                                            deviceMemory<double> o_w,
                                            deviceMemory<double> o_x,
                                            deviceMemory<double> o_y,
                                            comm_t               comm);

// ||o_a||_w2
template <typename T>
T linAlg_t::weightedNorm2(const dlong     N,
                          deviceMemory<T> o_w,
                          deviceMemory<T> o_a,
                          comm_t          comm) {

  int Nblock = (N + p_blockSize - 1) / p_blockSize;
  Nblock     = (Nblock > p_blockSize) ? p_blockSize
                                      : Nblock; // limit to p_blockSize entries

  pinnedMemory<T> h_scratch = platform->hostReserve<T>(p_blockSize);
  deviceMemory<T> o_scratch = platform->reserve<T>(p_blockSize);

  if constexpr(std::is_same_v<T, float>) {
    weightedNorm2KernelFloat(Nblock, N, o_w, o_a, o_scratch);
  } else {
    weightedNorm2KernelDouble(Nblock, N, o_w, o_a, o_scratch);
  }

  T globalnorm = 0;
  if(Nblock > 0) {
    h_scratch.copyFrom(o_scratch);
    globalnorm = h_scratch[0];
  } else {
    globalnorm = 0.0;
  }

  comm.Allreduce(globalnorm, Comm::Sum);

  return sqrt(globalnorm);
}

template float linAlg_t::weightedNorm2(const dlong         N,
                                       deviceMemory<float> o_w,
                                       deviceMemory<float> o_a,
                                       comm_t              comm);

template double linAlg_t::weightedNorm2(const dlong          N,
                                        deviceMemory<double> o_w,
                                        deviceMemory<double> o_a,
                                        comm_t               comm);

void linAlg_t::p2d(const dlong          N,
                   deviceMemory<pfloat> o_p,
                   deviceMemory<dfloat> o_d,
                   int                  reverse) {
  p2dKernel(N, o_p, o_d, reverse);
}

void linAlg_t::d2p(const dlong          N,
                   deviceMemory<dfloat> o_d,
                   deviceMemory<pfloat> o_p,
                   int                  reverse) {
  d2pKernel(N, o_d, o_p, reverse);
}

void linAlg_t::p2d(const dlong          N,
                   deviceMemory<pfloat> o_p,
                   deviceMemory<dfloat> o_d) {
  p2dKernel(N, o_p, o_d, 0);
}

void linAlg_t::d2p(const dlong          N,
                   deviceMemory<dfloat> o_d,
                   deviceMemory<pfloat> o_p) {
  d2pKernel(N, o_d, o_p, 0);
}

} // namespace libp
