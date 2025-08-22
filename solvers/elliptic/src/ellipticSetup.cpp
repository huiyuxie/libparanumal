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
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>

#define p_mask 0xffffffff
#define p_Nwarp 4
#define p_warpSize 32
#define PCG_BLOCKSIZE (p_Nwarp * p_warpSize)
#define p_Nloads 1

// Define a macro to build a kernel and assign it to a given variable
#define BUILD_KERNEL_zamx(tzStr, trStr, varName)                            \
  {                                                                         \
    std::string tzStr_                     = tzStr;                         \
    std::string trStr_                     = trStr;                         \
    kernelNewName                          = "zamx_" + tzStr_ + trStr_;     \
    kernelInfoAMP                          = kernelInfo;                    \
    kernelInfoAMP["okl/enabled"]           = false;                         \
    kernelInfoAMP["defines/pfloat2hfloat"] = pfloat2hfloatString;           \
    kernelInfoAMP["defines/hfloat2pfloat"] = hfloat2pfloatString;           \
    kernelInfoAMP["defines/Tz"]            = tzStr;                         \
    kernelInfoAMP["defines/Tr"]            = trStr;                         \
    kernelInfoAMP["defines/zamx"]          = kernelNewName;                 \
    varName = platform.buildKernel(fileName, kernelNewName, kernelInfoAMP); \
  }

#define BUILD_KERNEL_axpy(tzStr, varName)                                   \
  {                                                                         \
    std::string tzStr_                     = tzStr;                         \
    kernelNewName                          = "axpy_" + tzStr_;              \
    kernelInfoAMP                          = kernelInfo;                    \
    kernelInfoAMP["okl/enabled"]           = false;                         \
    kernelInfoAMP["defines/pfloat2hfloat"] = pfloat2hfloatString;           \
    kernelInfoAMP["defines/hfloat2pfloat"] = hfloat2pfloatString;           \
    kernelInfoAMP["defines/Tz"]            = tzStr;                         \
    kernelInfoAMP["defines/Tr"]            = "double";                      \
    kernelInfoAMP["defines/axpy"]          = kernelNewName;                 \
    varName = platform.buildKernel(fileName, kernelNewName, kernelInfoAMP); \
  }

#define BUILD_KERNEL_innerProd(tzStr, trStr, varName)                        \
  {                                                                          \
    std::string tzStr_                     = tzStr;                          \
    std::string trStr_                     = trStr;                          \
    kernelNewName                          = "innerProd_" + tzStr_ + trStr_; \
    kernelInfoAMP                          = kernelInfo;                     \
    kernelInfoAMP["okl/enabled"]           = false;                          \
    kernelInfoAMP["defines/pfloat2hfloat"] = pfloat2hfloatString;            \
    kernelInfoAMP["defines/hfloat2pfloat"] = hfloat2pfloatString;            \
    kernelInfoAMP["defines/Tz"]            = tzStr;                          \
    kernelInfoAMP["defines/Tr"]            = trStr;                          \
    kernelInfoAMP["defines/innerProd"]     = kernelNewName;                  \
    varName = platform.buildKernel(fileName, kernelNewName, kernelInfoAMP);  \
  }

#define BUILD_KERNEL_updatePCGDouble(tzStr, varName)                        \
  {                                                                         \
    std::string tzStr_                       = tzStr;                       \
    kernelNewName                            = "updatePCGDouble_" + tzStr_; \
    kernelInfoAMP                            = kernelInfo;                  \
    kernelInfoAMP["okl/enabled"]             = false;                       \
    kernelInfoAMP["defines/pfloat2hfloat"]   = pfloat2hfloatString;         \
    kernelInfoAMP["defines/hfloat2pfloat"]   = hfloat2pfloatString;         \
    kernelInfoAMP["defines/Tz"]              = tzStr;                       \
    kernelInfoAMP["defines/Tr"]              = "double";                    \
    kernelInfoAMP["defines/updatePCGDouble"] = kernelNewName;               \
    varName = platform.buildKernel(fileName, kernelNewName, kernelInfoAMP); \
  }

#define BUILD_KERNEL_updatePCGFloat(tzStr, varName)                         \
  {                                                                         \
    std::string tzStr_                      = tzStr;                        \
    kernelNewName                           = "updatePCGFloat_" + tzStr_;   \
    kernelInfoAMP                           = kernelInfo;                   \
    kernelInfoAMP["okl/enabled"]            = false;                        \
    kernelInfoAMP["defines/pfloat2hfloat"]  = pfloat2hfloatString;          \
    kernelInfoAMP["defines/hfloat2pfloat"]  = hfloat2pfloatString;          \
    kernelInfoAMP["defines/Tz"]             = tzStr;                        \
    kernelInfoAMP["defines/Tr"]             = "float";                      \
    kernelInfoAMP["defines/updatePCGFloat"] = kernelNewName;                \
    varName = platform.buildKernel(fileName, kernelNewName, kernelInfoAMP); \
  }

#define BUILD_KERNEL_Ax(tzStr, trStr, varName)                                   \
  {                                                                              \
    std::string tzStr_ = tzStr;                                                  \
    std::string trStr_ = trStr;                                                  \
    kernelNewName      = "ellipticAxTrilinearGatherDotHex3D_" + tzStr_ + trStr_; \
    kernelInfoAMP      = kernelInfo;                                             \
    kernelInfoAMP["okl/enabled"]           = false;                              \
    kernelInfoAMP["defines/pfloat2hfloat"] = pfloat2hfloatString;                \
    kernelInfoAMP["defines/hfloat2pfloat"] = hfloat2pfloatString;                \
    kernelInfoAMP["defines/Tz"]            = tzStr;                              \
    kernelInfoAMP["defines/Tr"]            = trStr;                              \
    kernelInfoAMP["defines/ellipticAxTrilinearGatherDotHex3D"] =                 \
        kernelNewName;                                                           \
    varName = platform.buildKernel(fileName, kernelNewName, kernelInfoAMP);      \
  }

void elliptic_t::Setup(platform_t&       _platform,
                       mesh_t&           _mesh,
                       settings_t&       _settings,
                       dfloat            _lambda,
                       const int         _NBCTypes,
                       const memory<int> _BCType) {

  platform = _platform;
  mesh     = _mesh;
  comm     = _mesh.comm;
  settings = _settings;
  lambda   = _lambda;

  Nfields = 1;

  // Trigger JIT kernel builds
  ogs::InitializeKernels(platform, ogs::Dfloat, ogs::Add);
  ogs::InitializeKernels(platform, ogs::Pfloat, ogs::Add);

  disc_ipdg = settings.compareSetting("DISCRETIZATION", "IPDG");
  disc_c0   = settings.compareSetting("DISCRETIZATION", "CONTINUOUS");

  // setup linear algebra module
  platform.linAlg().InitKernels(
      {"add",   "sum",    "scale", "set",    "setnc",     "axpy",
       "zaxpy", "axpync", "amx",   "amxpy",  "zamxpy",    "amxpync",
       "amxnc", "adx",    "adxpy", "zadxpy", "innerProd", "innerProdnc",
       "norm2", "d2p",    "p2d"});
  /*setup trace halo exchange */
  traceHalo = mesh.HaloTraceSetup(Nfields);

  // Boundary Type translation. Just defaults.
  NBCTypes = _NBCTypes;
  BCType.malloc(NBCTypes);
  BCType.copyFrom(_BCType);

  // setup boundary flags and make mask and masked ogs
  BoundarySetup();

  if(settings.compareSetting("DISCRETIZATION", "IPDG")) {
    // tau (penalty term in IPDG)
    if(mesh.elementType == Mesh::TRIANGLES ||
       mesh.elementType == Mesh::QUADRILATERALS) {
      tau = 2.0 * (mesh.N + 1) * (mesh.N + 2) / 2.0;
      if(mesh.dim == 3) { tau *= 1.5; }
    } else {
      tau = 2.0 * (mesh.N + 1) * (mesh.N + 3);
    }
  } else {
    tau = 0.0;
  }

  // OCCA build stuff
  properties_t kernelInfo = mesh.props; // copy base occa properties

  // set kernel name suffix
  std::string suffix = mesh.elementSuffix();

  std::string oklFilePrefix = DELLIPTIC "/okl/";
  std::string oklFileSuffix = ".okl";

  std::string fileName, kernelName, kernelNameDegreeFloat,
      kernelNameDegreeDouble, kernelNameDegreeHalf, kernelNewName;

  // add standard boundary functions
  std::string boundaryHeaderFileName;
  if(mesh.dim == 2)
    boundaryHeaderFileName =
        std::string(DELLIPTIC "/data/ellipticBoundary2D.h");
  else if(mesh.dim == 3)
    boundaryHeaderFileName =
        std::string(DELLIPTIC "/data/ellipticBoundary3D.h");
  kernelInfo["includes"] += boundaryHeaderFileName;
  kernelInfo["defines/"
             "ncLoad"]  = "__ldcs";
  kernelInfo["defines/"
             "ncStore"] = "__stcs";

  int blockMax = 256;
  if(platform.device.mode() == "CUDA") blockMax = 512;

  int NblockV               = std::max(1, blockMax / mesh.Np);
  kernelInfo["defines/"
             "p_NblockV"]   = NblockV;
  kernelInfo["defines/"
             "p_blockSize"] = (int)PCG_BLOCKSIZE;
  kernelInfo["defines/"
             "p_Nloads"]    = (int)p_Nloads;
  kernelInfo["defines/"
             "p_Nwarp"]     = (int)p_Nwarp;
  kernelInfo["defines/"
             "p_warpSize"]  = (int)p_warpSize;

  properties_t kernelInfoDouble       = kernelInfo;
  kernelInfoDouble["defines/dfloat"]  = "double";
  kernelInfoDouble["defines/dfloat4"] = "double4";

  properties_t kernelInfoFloat       = kernelInfo;
  kernelInfoFloat["defines/dfloat"]  = "float";
  kernelInfoFloat["defines/dfloat4"] = "float4";

  properties_t kernelInfoHalf = kernelInfo;

  // PCG kernel
  properties_t kernelInfoPCG = kernelInfo;
  //  fileName = oklFilePrefix + "linearSolverUpdatePCG" + oklFileSuffix;

  properties_t kernelInfoAMP = kernelInfo;
  fileName                   = oklFilePrefix + "linearSolverUpdatePCG.cu";

  BUILD_KERNEL_zamx("__half", "float", zamx_H_F_Kernel);
  BUILD_KERNEL_zamx("float", "float", zamx_F_F_Kernel);
  BUILD_KERNEL_zamx("double", "float", zamx_D_F_Kernel);
  BUILD_KERNEL_zamx("__half", "double", zamx_H_D_Kernel);
  BUILD_KERNEL_zamx("float", "double", zamx_F_D_Kernel);
  BUILD_KERNEL_zamx("double", "double", zamx_D_D_Kernel);

  kernelInfoAMP                          = kernelInfo;
  kernelInfoAMP["okl/enabled"]           = false;
  kernelInfoAMP["defines/pfloat2hfloat"] = pfloat2hfloatString;
  kernelInfoAMP["defines/hfloat2pfloat"] = hfloat2pfloatString;
  kernelInfoAMP["defines/Tz"]            = "double";
  kernelInfoAMP["defines/Tr"]            = "double";
  kernelNewName                          = "zamx_Half2_D";
  kernelInfoAMP["defines/zamx_Half2"]    = kernelNewName;
  zamx_Half2_D_Kernel =
      platform.buildKernel(fileName, kernelNewName, kernelInfoAMP);
  kernelInfoAMP                          = kernelInfo;
  kernelInfoAMP["okl/enabled"]           = false;
  kernelInfoAMP["defines/pfloat2hfloat"] = pfloat2hfloatString;
  kernelInfoAMP["defines/hfloat2pfloat"] = hfloat2pfloatString;
  kernelInfoAMP["defines/Tz"]            = "double";
  kernelInfoAMP["defines/Tr"]            = "float";
  kernelNewName                          = "zamx_Half2_F";
  kernelInfoAMP["defines/zamx_Half2"]    = kernelNewName;
  zamx_Half2_F_Kernel =
      platform.buildKernel(fileName, kernelNewName, kernelInfoAMP);

  kernelInfoAMP                          = kernelInfo;
  kernelInfoAMP["okl/enabled"]           = false;
  kernelInfoAMP["defines/pfloat2hfloat"] = pfloat2hfloatString;
  kernelInfoAMP["defines/hfloat2pfloat"] = hfloat2pfloatString;
  kernelInfoAMP["defines/Tz"]            = "double";
  kernelInfoAMP["defines/Tr"]            = "double";
  axpy_H_Kernel = platform.buildKernel(fileName, "axpy_H", kernelInfoAMP);
  axpy_F_Kernel = platform.buildKernel(fileName, "axpy_F", kernelInfoAMP);
  axpy_D_Kernel = platform.buildKernel(fileName, "axpy_D", kernelInfoAMP);
  axpy_Half2_Kernel =
      platform.buildKernel(fileName, "axpy_Half2", kernelInfoAMP);
  axpy_F2_Kernel = platform.buildKernel(fileName, "axpy_F2", kernelInfoAMP);

  BUILD_KERNEL_updatePCGDouble("__half", updatePCGDouble_H_Kernel);
  BUILD_KERNEL_updatePCGDouble("float", updatePCGDouble_F_Kernel);
  BUILD_KERNEL_updatePCGDouble("double", updatePCGDouble_D_Kernel);
  BUILD_KERNEL_updatePCGFloat("__half", updatePCGFloat_H_Kernel);
  BUILD_KERNEL_updatePCGFloat("float", updatePCGFloat_F_Kernel);
  BUILD_KERNEL_updatePCGFloat("double", updatePCGFloat_D_Kernel);

  BUILD_KERNEL_innerProd("__half", "float", innerProd_H_F_Kernel);
  BUILD_KERNEL_innerProd("float", "float", innerProd_F_F_Kernel);
  BUILD_KERNEL_innerProd("double", "float", innerProd_D_F_Kernel);
  BUILD_KERNEL_innerProd("__half", "double", innerProd_H_D_Kernel);
  BUILD_KERNEL_innerProd("float", "double", innerProd_F_D_Kernel);
  BUILD_KERNEL_innerProd("double", "double", innerProd_D_D_Kernel);
  kernelInfoAMP                            = kernelInfo;
  kernelInfoAMP["okl/enabled"]             = false;
  kernelInfoAMP["defines/pfloat2hfloat"]   = pfloat2hfloatString;
  kernelInfoAMP["defines/hfloat2pfloat"]   = hfloat2pfloatString;
  kernelInfoAMP["defines/Tz"]              = "double";
  kernelInfoAMP["defines/Tr"]              = "double";
  kernelNewName                            = "innerProd_Half2_D";
  kernelInfoAMP["defines/innerProd_Half2"] = kernelNewName;
  innerProd_Half2_D_Kernel =
      platform.buildKernel(fileName, kernelNewName, kernelInfoAMP);
  kernelInfoAMP                            = kernelInfo;
  kernelInfoAMP["okl/enabled"]             = false;
  kernelInfoAMP["defines/pfloat2hfloat"]   = pfloat2hfloatString;
  kernelInfoAMP["defines/hfloat2pfloat"]   = hfloat2pfloatString;
  kernelInfoAMP["defines/Tz"]              = "double";
  kernelInfoAMP["defines/Tr"]              = "float";
  kernelNewName                            = "innerProd_Half2_F";
  kernelInfoAMP["defines/innerProd_Half2"] = kernelNewName;
  innerProd_Half2_F_Kernel =
      platform.buildKernel(fileName, kernelNewName, kernelInfoAMP);

  fileName = oklFilePrefix + "ellipticAxTrilinearAMPHex3D.cu";
  BUILD_KERNEL_Ax("__half", "float", AxTrilinearGatherDot_H_F_Kernel);
  BUILD_KERNEL_Ax("float", "float", AxTrilinearGatherDot_F_F_Kernel);
  BUILD_KERNEL_Ax("double", "float", AxTrilinearGatherDot_D_F_Kernel);
  BUILD_KERNEL_Ax("__half", "double", AxTrilinearGatherDot_H_D_Kernel);
  BUILD_KERNEL_Ax("float", "double", AxTrilinearGatherDot_F_D_Kernel);
  BUILD_KERNEL_Ax("double", "double", AxTrilinearGatherDot_D_D_Kernel);

  fileName = oklFilePrefix + "ellipticAx" + suffix + oklFileSuffix;
  kernelInfoFloat["defines/TAp"]   = "float";
  kernelInfoFloat["defines/Tp"]    = "float";
  kernelInfoFloat["defines/Tggeo"] = "float";
  kernelName                       = "ellipticAxGatherDot" + suffix;
  floatAxGatherDotKernel =
      platform.buildKernel(fileName, kernelName, kernelInfoFloat);

  // half precision
  properties_t tmpKernelInfo   = kernelInfoPCG;
  fileName                     = oklFilePrefix + "linearSolverUpdatePCG.cu";
  tmpKernelInfo["okl/enabled"] = false;
  //  tmpKernelInfo["defines/hfloat"] = hfloatString;
  tmpKernelInfo["defines/pfloat2hfloat"] = pfloat2hfloatString;
  tmpKernelInfo["defines/hfloat2pfloat"] = hfloat2pfloatString;
  tmpKernelInfo["defines/Tr"]            = "double";
  tmpKernelInfo["defines/TAp"]           = "double";
  tmpKernelInfo["defines/Tz"]            = "float";
  //  axpySetHalfKernel = platform.buildKernel(fileName, "axpySet",
  //  tmpKernelInfo);
  // p2dInnerProdHalfKernel = platform.buildKernel(fileName, "p2dInnerProd",
  // tmpKernelInfo); p2dInnerProdHalfFloatKernel =
  // platform.buildKernel(fileName, "p2dInnerProdFloat", tmpKernelInfo);
  //  updatePCGHalfKernel = platform.buildKernel(fileName, "updatePCG",
  //  tmpKernelInfo); updatePCGHalfFloatKernel = platform.buildKernel(fileName,
  //  "updatePCGFloat", tmpKernelInfo); scaleHalfKernel =
  //  platform.buildKernel(fileName, "scale", tmpKernelInfo);
  // scaleP2hHalfKernel = platform.buildKernel(fileName, "scaleP2h",
  // tmpKernelInfo);
  //  axpyDoubleKernel = platform.buildKernel(fileName, "axpyDouble",
  //  tmpKernelInfo);

  properties_t AxHalfKernelInfo   = kernelInfoPCG;
  fileName                        = oklFilePrefix + "ellipticAxHex3D.cu";
  AxHalfKernelInfo["okl/enabled"] = false;
  AxHalfKernelInfo["defines/pfloat2hfloat"] = pfloat2hfloatString;
  AxHalfKernelInfo["defines/hfloat2pfloat"] = hfloat2pfloatString;
  AxHalfKernelInfo["defines/TAp"]           = "double";
  AxHalfKernelInfo["defines/Tggeo"]         = "double";
  //  AxGatherDotHalfKernel = platform.buildKernel(fileName,
  //  "ellipticAxGatherDotHalfHex3D", AxHalfKernelInfo);

  fileName = oklFilePrefix + "linAlgVector.cu";
  //  d2hKernel = platform.buildKernel(fileName, "d2h", tmpKernelInfo);
  p2hKernel   = platform.buildKernel(fileName, "p2h", tmpKernelInfo);
  h2pKernel   = platform.buildKernel(fileName, "h2p", tmpKernelInfo);
  d2hKernel   = platform.buildKernel(fileName, "d2h", tmpKernelInfo);
  norm2Kernel = platform.buildKernel(fileName, "norm2", tmpKernelInfo);
  setKernel   = platform.buildKernel(fileName, "set", tmpKernelInfo);

  fileName = oklFilePrefix + "multigridUpdateCheby.cu";
  //  zamxHalfKernel = platform.buildKernel(fileName, "zamx", tmpKernelInfo);
  // updateCheby4HalfKernel = platform.buildKernel(fileName, "updateCheby4",
  // tmpKernelInfo); updateCheby1HalfKernel = platform.buildKernel(fileName,
  // "updateCheby1", tmpKernelInfo); updateCheby5HalfKernel =
  // platform.buildKernel(fileName, "updateCheby5", tmpKernelInfo);

  fileName             = oklFilePrefix + "ellipticAxTrilinear" + suffix + ".cu";
  kernelName           = "ellipticAxTrilinearGatherSmooth" + suffix;
  kernelNameDegreeHalf = kernelName + std::string("Half") + std::string("N") +
                         std::to_string(mesh.N);
  kernelInfoHalf["defines/" + kernelName] = kernelNameDegreeHalf;
  kernelInfoHalf["okl/enabled"]           = false;
  //  kernelInfoHalf["defines/hfloat"] = hfloatString;
  kernelInfoHalf["defines/pfloat2hfloat"] = pfloat2hfloatString;
  kernelInfoHalf["defines/hfloat2pfloat"] = hfloat2pfloatString;
  //  halfAxTrilinearGatherSmoothKernel = platform.buildKernel(fileName,
  //  kernelNameDegreeHalf,
  //							   kernelInfoHalf);
  kernelName           = "ellipticAxTrilinearGather" + suffix;
  kernelNameDegreeHalf = kernelName + std::string("Half") + std::string("N") +
                         std::to_string(mesh.N);
  kernelInfoHalf["defines/" + kernelName] = kernelNameDegreeHalf;
  //  halfAxTrilinearGatherKernel = platform.buildKernel(fileName,
  //  kernelNameDegreeHalf,
  //							   kernelInfoHalf);
  kernelName           = "ellipticAxTrilinearGatherFloat" + suffix;
  kernelNameDegreeHalf = kernelName + std::string("Half") + std::string("N") +
                         std::to_string(mesh.N);
  kernelInfoHalf["defines/" + kernelName] = kernelNameDegreeHalf;
  //  halfFloatAxTrilinearGatherKernel = platform.buildKernel(fileName,
  //  kernelNameDegreeHalf,
  //							   kernelInfoHalf);

  // updatePCG double
  fileName   = oklFilePrefix + "linearSolverUpdatePCG" + oklFileSuffix;
  kernelName = "updatePCG";
  updatePCGDoubleKernel =
      platform.buildKernel(fileName, kernelName, kernelInfoDouble);

  // Ax kernel
  if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    fileName = oklFilePrefix + "ellipticAx" + suffix + oklFileSuffix;
    if(mesh.elementType == Mesh::HEXAHEDRA) {
      if(mesh.settings.compareSetting("ELEMENT MAP", "TRILINEAR"))
        kernelName = "ellipticPartialAxTrilinear" + suffix;
      else
        kernelName = "ellipticPartialAx" + suffix;
    } else {
      kernelName = "ellipticPartialAx" + suffix;
    }

    partialAxKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoDouble);

    floatPartialAxKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoFloat);

    kernelName = "ellipticAxGather" + suffix;
    // AxGatherKernel = platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxGatherKernel = platform.buildKernel(fileName, kernelName,
    //                                             kernelInfoFloat);
    kernelName = "ellipticAxGatherSmooth" + suffix;
    // AxGatherSmoothKernel = platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxGatherSmoothKernel = platform.buildKernel(fileName, kernelName,
    //                                             kernelInfoFloat);

    kernelName = "ellipticAxGathernc" + suffix;
    // AxGatherncKernel = platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxGatherncKernel = platform.buildKernel(fileName, kernelName,
    //                                             kernelInfoFloat);
    // fileName = oklFilePrefix + "ellipticAx" + suffix + oklFileSuffix;
    // kernelName = "ellipticAxGatherDot" + suffix;
    // AxGatherDotKernel = platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxGatherDotKernel = platform.buildKernel(fileName, kernelName,
    //                                             kernelInfoFloat);

    // kernelName = "ellipticAxGatherDotnc" + suffix;
    // AxGatherDotncKernel = platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxGatherDotncKernel = platform.buildKernel(fileName, kernelName,
    //                                             kernelInfoFloat);

    fileName = oklFilePrefix + "ellipticAxTrilinear" + suffix + oklFileSuffix;
    // kernelName = "ellipticAxTrilinearGather" + suffix;
    // kernelNameDegreeFloat = kernelName + std::string("Float") +
    // std::string("N") + std::to_string(mesh.N); kernelInfoFloat["defines/"+
    // kernelName] = kernelNameDegreeFloat; AxTrilinearGatherKernel =
    // platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxTrilinearGatherKernel = platform.buildKernel(fileName,
    // kernelNameDegreeFloat,
    //                                             kernelInfoFloat);

    kernelName = "ellipticAxTrilinearGatherDot" + suffix;
    AxTrilinearGatherDotKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoDouble);
    floatAxTrilinearGatherDotKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoFloat);

    // kernelName = "ellipticAxTrilinearGatherSmooth" + suffix;
    // kernelNameDegreeFloat = kernelName + std::string("Float") +
    // std::string("N") + std::to_string(mesh.N); kernelInfoFloat["defines/"+
    // kernelName] = kernelNameDegreeFloat; AxTrilinearGatherSmoothKernel =
    // platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxTrilinearGatherSmoothKernel = platform.buildKernel(fileName,
    // kernelNameDegreeFloat,
    //                                             kernelInfoFloat);

    // kernelName = "ellipticAxInterpGather" + suffix;
    // kernelNameDegreeFloat = kernelName + std::string("Float") +
    // std::string("N") + std::to_string(mesh.N); kernelInfoFloat["defines/"+
    // kernelName] = kernelNameDegreeFloat; AxInterpGatherKernel =
    // platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // floatAxInterpGatherKernel = platform.buildKernel(fileName,
    // kernelNameDegreeFloat,
    //                                             kernelInfoFloat);

    // kernelName = "ellipticAxInterpGatherSmooth" + suffix;
    // kernelNameDegreeFloat = kernelName + std::string("Float") +
    // std::string("N") + std::to_string(mesh.N); kernelInfoFloat["defines/"+
    // kernelName] = kernelNameDegreeFloat; AxInterpGatherSmoothKernel =
    // platform.buildKernel(fileName, kernelName, 						      kernelInfoDouble);

    // floatAxInterpGatherSmoothKernel = platform.buildKernel(fileName,
    // 							   kernelNameDegreeFloat,
    // 							   kernelInfoFloat);

  } else if(settings.compareSetting("DISCRETIZATION", "IPDG")) {
    int Nmax                   = std::max(mesh.Np, mesh.Nfaces * mesh.Nfp);
    kernelInfoDouble["defines/"
                     "p_Nmax"] = Nmax;
    kernelInfoFloat["defines/"
                    "p_Nmax"]  = Nmax;

    fileName   = oklFilePrefix + "ellipticGradient" + suffix + oklFileSuffix;
    kernelName = "ellipticPartialGradient" + suffix;
    partialGradientKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoDouble);

    floatPartialGradientKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoFloat);

    fileName   = oklFilePrefix + "ellipticAxIpdg" + suffix + oklFileSuffix;
    kernelName = "ellipticPartialAxIpdg" + suffix;

    partialIpdgKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoDouble);

    floatPartialIpdgKernel =
        platform.buildKernel(fileName, kernelName, kernelInfoFloat);
  }

  /* build kernels */
  // properties_t kernelInfo = platform.props(); // copy base properties

  mesh.CubatureSetup();
  mesh.CubaturePhysicalNodes();
  properties_t kernelInfoCubNp = kernelInfo;
  dlong        p_NblockC       = (mesh.dim == 3) ? 1 : (512 / (mesh.cubNp));
  kernelInfoCubNp["defines/"
                  "p_NblockC"] = p_NblockC;
  kernelInfoCubNp["defines/"
                  "p_cubNp"]   = mesh.cubNp;
  kernelInfoCubNp["defines/"
                  "p_cubNq"]   = mesh.cubNq;
  kernelInfoCubNp["defines/"
                  "p_NblockC"] = p_NblockC;

  // TW: H1 error estimate stuff

  int         NblocksC = (mesh.Nelements + p_NblockC - 1) / p_NblockC;
  std::string dataFileName;
  settings.getSetting("DATA FILE", dataFileName);
  kernelInfoCubNp["includes"] += dataFileName;
  //    std::cout  << kernelInfoCubNp;
  fileName =
      oklFilePrefix + "/ellipticCubatureH1Error" + suffix + oklFileSuffix;
  kernelName = "ellipticCubatureH1Error" + suffix;
  cubatureH1ErrorKernel =
      platform.buildKernel(fileName, kernelName, kernelInfoCubNp);
  kernelName = "ellipticCubatureH1L2Error" + suffix;
  cubatureH1L2ErrorKernel =
      platform.buildKernel(fileName, kernelName, kernelInfoCubNp);

  /* Preconditioner Setup */
  if(settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    Ndofs = ogsMasked.Ngather * Nfields;
    Nhalo = gHalo.Nhalo * Nfields;
  } else {
    Ndofs = mesh.Nelements * mesh.Np * Nfields;
    Nhalo = mesh.totalHaloPairs * mesh.Np * Nfields;
  }

  // setup optimization for linear solver
  if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "NONE"))
    this->multigridDriver = 0;
  else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "FUSE"))
    this->multigridDriver = 1;
  else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "AXTRILINEAR"))
    this->multigridDriver = 2;
  else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "AXINTERP"))
    this->multigridDriver = 3;
  else if(settings.compareSetting("LINEAR SOLVER OPTIMIZATION", "ALT"))
    this->multigridDriver = 4;

  // if (settings.compareSetting("PRECONDITIONER", "JACOBI"))
  //   precon.Setup<JacobiPrecon>(*this);
  // else if (settings.compareSetting("PRECONDITIONER", "MASSMATRIX"))
  //   precon.Setup<MassMatrixPrecon>(*this);
  // else if (settings.compareSetting("PRECONDITIONER", "PARALMOND"))
  //   precon.Setup<ParAlmondPrecon>(*this);
  // else if (settings.compareSetting("PRECONDITIONER", "MULTIGRID"))
  //   precon.Setup<MultiGridPrecon>(*this);
  // else if (settings.compareSetting("PRECONDITIONER", "SEMFEM"))
  //   precon.Setup<SEMFEMPrecon>(*this);
  // else if (settings.compareSetting("PRECONDITIONER", "OAS"))
  //   precon.Setup<OASPrecon>(*this);
  // else if (settings.compareSetting("PRECONDITIONER", "NONE"))
  //   precon.Setup<IdentityPrecon>(Ndofs);

  // set mesh.o_EXYZ
  memory<dfloat> EXYZ(mesh.Nelements * mesh.Nverts * mesh.dim);
  int            offset;
  for(int e = 0; e < mesh.Nelements; e++) {
    offset = e * mesh.Nverts;
    for(int i = 0; i < mesh.Nverts; i++) {
      EXYZ[offset * mesh.dim + i]                   = mesh.EX[offset + i];
      EXYZ[offset * mesh.dim + mesh.Nverts + i]     = mesh.EY[offset + i];
      EXYZ[offset * mesh.dim + mesh.Nverts * 2 + i] = mesh.EZ[offset + i];
    }
  }
  mesh.o_EXYZ = platform.malloc<dfloat>(EXYZ);
  mesh.o_pfloat_EXYZ =
      platform.malloc<pfloat>(mesh.Nelements * mesh.Nverts * mesh.dim);
  platform.linAlg().d2p(mesh.Nelements * mesh.Nverts * mesh.dim,
                        mesh.o_EXYZ,
                        mesh.o_pfloat_EXYZ,
                        0);

  // set mesh.o_gllzw
  memory<dfloat> gllzw(2 * mesh.Nq);
  for(int i = 0; i < mesh.Nq; i++) {
    gllzw[i]           = mesh.gllz[i];
    gllzw[i + mesh.Nq] = mesh.gllw[i];
  }
  mesh.o_gllzw        = platform.malloc<dfloat>(gllzw);
  mesh.o_pfloat_gllzw = platform.malloc<pfloat>(2 * mesh.Nq);
  platform.linAlg().d2p(2 * mesh.Nq, mesh.o_gllzw, mesh.o_pfloat_gllzw, 0);

  int            base;
  int            id;
  int            Nggeo  = mesh.Nggeo;
  int            Nvgeo  = mesh.Nvgeo;
  int            Nverts = mesh.Nverts;
  int            Nq     = mesh.Nq;
  int            Np     = mesh.Np;
  memory<dfloat> ggeoV(mesh.Nelements * (Nggeo + 1) * Nverts);
  int            index(int offset, int Nq, int i);

  for(int e = 0; e < mesh.Nelements; ++e) {
    for(int i = 0; i < Nggeo; ++i) {
      base   = e * (Nggeo + 1) * Nverts + i * Nverts;
      offset = e * Nggeo * Np + i * Np;
      for(int k = 0; k < Nverts; ++k) {
        id              = index(offset, Nq, k);
        ggeoV[base + k] = mesh.ggeoNoW[id];
      }
    }
    base = e * (Nggeo + 1) * Nverts + Nggeo * Nverts;
    for(int k = 0; k < Nverts; ++k) {
      id              = index(0, Nq, k);
      ggeoV[base + k] = mesh.wJ[id];
    }
  }

  mesh.o_ggeoV = platform.malloc<dfloat>(ggeoV);
  mesh.o_pfloat_ggeoV =
      platform.malloc<pfloat>(mesh.Nelements * (Nggeo + 1) * Nverts);
  platform.linAlg().d2p(mesh.Nelements * (Nggeo + 1) * Nverts,
                        mesh.o_ggeoV,
                        mesh.o_pfloat_ggeoV,
                        0);

  // 10 vgeo in total, 9 partial diff, plus J
  memory<dfloat> vgeoV(mesh.Nelements * (9 + 1) * Nverts);
  for(int e = 0; e < mesh.Nelements; ++e) {
    for(int i = 0; i < 9; ++i) {
      base   = e * (9 + 1) * Nverts + i * Nverts;
      offset = e * Nvgeo * Np + i * Np;
      for(int k = 0; k < Nverts; ++k) {
        id              = index(offset, Nq, k);
        vgeoV[base + k] = mesh.vgeo[id];
      }
    }
    base   = e * (9 + 1) * Nverts + 9 * Nverts;
    offset = e * Nvgeo * Np + 9 * Np;
    for(int k = 0; k < Nverts; ++k) {
      id              = index(offset, Nq, k);
      vgeoV[base + k] = mesh.vgeo[id];
    }
  }
  mesh.o_vgeoV = platform.malloc<dfloat>(vgeoV);
  mesh.o_pfloat_vgeoV =
      platform.malloc<pfloat>(mesh.Nelements * (9 + 1) * Nverts);
  platform.linAlg().d2p(
      mesh.Nelements * (9 + 1) * Nverts, mesh.o_vgeoV, mesh.o_pfloat_vgeoV, 0);

  // reorder mesh.o_localGatherElementList
  void elementNeighborhood(mesh_t & mesh, int& maxDegE, memory<hlong>& eNbr);
  void separateElements(mesh_t & mesh,
                        int            maxDegE,
                        memory<hlong>& eNbr,
                        int&           NsetE,
                        int&           maxNumE,
                        memory<hlong>& NeleS,
                        memory<hlong>& setE);

  // build sets of separate nodes
  int           maxDegE, maxNumE;
  memory<hlong> eNbr;
  memory<hlong> setE;

  elementNeighborhood(mesh, maxDegE, eNbr);
  separateElements(mesh, maxDegE, eNbr, NsetE, maxNumE, NeleS, setE);

  // for(int Ns = 0; Ns< NsetE; ++Ns){
  //   for (hlong s = 0; s < NeleS[Ns]; ++s)
  //     {
  // 	int id = Ns * maxNumE + s;
  // 	printf("setE(%d, %d) = %d \n", Ns, s, setE[id]); // add the element
  // newMarked[s] to the set NsetE
  //   }
  // }

  // keep the original elementList in elliptic
  for(int e = 0; e < mesh.Nelements; ++e) {
    mesh.localGatherElementList[e] = e;
  }
  o_localGatherElementList =
      platform.malloc<dlong>(mesh.localGatherElementList);

  offset = 0;
  for(int Ns = 0; Ns < NsetE; ++Ns) {
    for(hlong s = 0; s < NeleS[Ns]; ++s) {
      int id                                  = Ns * maxNumE + s;
      mesh.localGatherElementList[s + offset] = setE[id];
    }
    offset += NeleS[Ns];
  }
  mesh.o_localGatherElementList.copyFrom(mesh.localGatherElementList);
  // for(int e=0; e<mesh.Nelements; e++){
  //   printf("mesh.localGatherElementList[%d] = %d \n", e,
  //   mesh.localGatherElementList[e]);
  // }

  // construct o_GlobalToLocalColor
  memory<dlong> GlobalToLocalColor(mesh.Nelements * mesh.Np);
  // first copy
  for(int e = 0; e < mesh.Nelements; ++e) {
    for(int i = 0; i < mesh.Np; ++i) {
      int id                 = mesh.Np * e + i;
      GlobalToLocalColor[id] = GlobalToLocal[id];
    }
  }
  // map some global index to -2-gid
  memory<int> isFirst(mesh.Nelements * mesh.Np);
  for(int n = 0; n < (mesh.Nelements * mesh.Np); ++n) { isFirst[n] = 1; }
  for(int e = 0; e < mesh.Nelements; ++e) {
    int element = mesh.localGatherElementList[e];
    // printf("element = %d ", element);
    for(int i = 0; i < mesh.Np; ++i) {
      id        = mesh.Np * element + i;
      dlong gid = GlobalToLocalColor[id];
      //      printf(" %d ", gid);
      if(gid >= 0) { // boundary nodes is not affected by the mapping
        if(isFirst[gid] == 1) {
          isFirst[gid] = 0;
          // otherwise do nothing
        } else {
          gid                    = -2 - gid;
          GlobalToLocalColor[id] = gid;
        }
        //	printf(" %d ", gid);
      }
    }
    //      printf("\n");
  }
  o_GlobalToLocalColor = platform.malloc<dlong>(GlobalToLocalColor);
}

int index(int offset, int Nq, int i) {
  int id = 0;
  if(i == 0)
    id = 0;
  else if(i == 1)
    id = Nq - 1;
  else if(i == 2)
    id = Nq * (Nq - 1) + Nq - 1;
  else if(i == 3)
    id = Nq * (Nq - 1);
  else if(i == 4)
    id = Nq * Nq * (Nq - 1);
  else if(i == 5)
    id = Nq * Nq * (Nq - 1) + 0 * Nq + (Nq - 1);
  else if(i == 6)
    id = Nq * Nq * (Nq - 1) + Nq * (Nq - 1) + (Nq - 1);
  else if(i == 7)
    id = Nq * Nq * (Nq - 1) + Nq * (Nq - 1) + 0;

  id = id + offset;
  return id;
}

void elementNeighborhood(mesh_t& mesh, int& maxDegE1, memory<hlong>& eNbr1) {

  // pass by copy
  dlong Nelements = mesh.Nelements;
  // int Nfaces = mesh.Nfaces;
  int   Nverts = mesh.Nverts;
  hlong Nnodes = mesh.Nnodes;

  // build VToE
  memory<int> NVToE(Nnodes, 0);

  // count the degree of vertexes
  for(dlong e = 0; e < Nelements; ++e) {
    for(int v = 0; v < Nverts; ++v) {
      const hlong id = mesh.EToV[v + e * Nverts];
      NVToE[id] += 1;
    }
  }

  // max(degreeV)
  int           maxDegV = *std::max_element(NVToE.begin(), NVToE.end());
  memory<hlong> VToE(Nnodes * maxDegV, -1); // assign to -1 if empty

  ::memset(NVToE.begin(), 0, NVToE.size());

  // find all elements containing the vertex
  for(dlong e = 0; e < Nelements; ++e) {
    for(int v = 0; v < Nverts; ++v) {
      const hlong id                 = mesh.EToV[v + e * Nverts];
      VToE[id * maxDegV + NVToE[id]] = e;
      NVToE[id] += 1;
    }
  }

#if 0
  for(int i=0;i<Nnodes;++i){
    printf("Vertex %d is contained in %d elements\n", i, NVToE[i]);
  }
#endif

  // count size(eNbr1) (the first layer of neighbor of the element)
  memory<int>   NeNbr1(Nelements, 0);
  memory<hlong> tmp(Nverts * maxDegV);

  for(dlong e = 0; e < Nelements; ++e) {
    int i = 0;
    for(int v = 0; v < Nverts; ++v) { // loop all vertex
      const dlong id = mesh.EToV[v + e * Nverts];
      for(int s = 0; s < NVToE[id];
          ++s) { // loop all elements containing this vertex
        tmp[i] = VToE[id * maxDegV + s];
        i += 1;
      }
    }

    // unique(tmp)
    std::sort(tmp.begin(), tmp.begin() + i);
    hlong* last = std::unique(tmp.begin(), tmp.begin() + i);

    NeNbr1[e] = static_cast<int>(last - tmp.begin());
  }

  // max(degreeE1), allocate mem
  maxDegE1 = *std::max_element(NeNbr1.begin(), NeNbr1.end());
  eNbr1.malloc(Nelements * maxDegE1, -1); // assign -1 if the entry is empty

  // build eNbr1
  for(dlong e = 0; e < Nelements; ++e) {
    int i = 0;
    for(int v = 0; v < Nverts; ++v) { // loop all vertex
      const dlong id = mesh.EToV[v + e * Nverts];
      for(int s = 0; s < NVToE[id];
          ++s) { // loop all elements containing this vertex
        tmp[i] = VToE[id * maxDegV + s];
        i += 1;
      }
    }

    // unique(tmp) and set repeated entries to be -1
    std::sort(tmp.begin(), tmp.begin() + i);
    std::unique(tmp.begin(), tmp.begin() + i);

    for(int n = 0; n < NeNbr1[e]; ++n) { eNbr1[n + e * maxDegE1] = tmp[n]; }
  }
  //  printf("maxDegE1 = %d, maxDegE2 = %d, maxDegE3 = %d \n ", maxDegE1,
  //  maxDegE2, maxDegE3);
}

void separateElements(mesh_t&        mesh,
                      int            maxDegE,
                      memory<hlong>& eNbr,
                      int&           NsetE,
                      int&           maxNumE,
                      memory<hlong>& NeleS,
                      memory<hlong>& setE) {

  hlong Nelements = mesh.Nelements;

  memory<hlong> markedElement(Nelements, 0);
  memory<hlong> newMarked(Nelements, 0);
  memory<hlong> nbr(Nelements, 0);
  NeleS.malloc(maxDegE, 0);

  hlong sumMarkedElements = 0;
  NsetE                   = 0;
  int i                   = 0;

  while(sumMarkedElements < Nelements) {
    for(int k = 0; k < Nelements; ++k) {
      if((nbr[k] == 0) && (markedElement[k] == 0)) {
        markedElement[k] = 1; // mark the element k
        newMarked[i]     = k;

        for(hlong s = 0; s < maxDegE; ++s) {
          hlong id = eNbr[k * maxDegE + s];
          if(id > -1) nbr[id] = 1;
        }
        i += 1;
      }
    }

    NeleS[NsetE]      = i;
    sumMarkedElements = std::accumulate(
        markedElement.begin(), markedElement.end(), static_cast<hlong>(0));

    ++NsetE;
    i = 0;
    ::memset(nbr.begin(), 0, nbr.size());
    ::memset(newMarked.begin(), 0, newMarked.size());
  }

  maxNumE = *std::max_element(NeleS.begin(), NeleS.end());
  setE.malloc(NsetE * maxNumE, -1); // assign -1 if the entry is empty
  ::memset(markedElement.begin(), 0, markedElement.size());

  // printf("NsetE = %d, maxNumE = %d \n", NsetE, maxNumE);

  NsetE             = 0;
  sumMarkedElements = 0;

  while(sumMarkedElements < Nelements) {

    for(int k = 0; k < Nelements; ++k) {

      if((nbr[k] == 0) && (markedElement[k] == 0)) {

        markedElement[k] = 1; // mark the element k
        newMarked[i]     = k;

        for(hlong s = 0; s < maxDegE; ++s) {
          hlong id = eNbr[k * maxDegE + s];
          if(id > -1) nbr[id] = 1;
        }
        i += 1;
      }
    }

    NeleS[NsetE]      = i;
    sumMarkedElements = std::accumulate(
        markedElement.begin(), markedElement.end(), static_cast<hlong>(0));

    for(hlong s = 0; s < NeleS[NsetE]; ++s) {
      int id   = NsetE * maxNumE + s;
      setE[id] = newMarked[s]; // add the element newMarked[s] to the set NsetE
    }

    ++NsetE;
    i = 0;
    ::memset(nbr.begin(), 0, nbr.size());
    ::memset(newMarked.begin(), 0, newMarked.size());
  }
}
