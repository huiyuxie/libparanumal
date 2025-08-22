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

#define p_mask 0xffffffff
#define p_Nwarp 4
#define p_warpSize 32
//#define PCG_BLOCKSIZE 128
#define PCG_BLOCKSIZE (p_Nwarp*p_warpSize)  
#define p_Nloads 1

elliptic_t elliptic_t::SetupNewDegree(mesh_t& meshC){

  //if asking for the same degree, return the original solver
  if (meshC.N == mesh.N) return *this;

  //shallow copy
  elliptic_t elliptic = *this;

  elliptic.mesh = meshC;

  /*setup trace halo exchange */
  elliptic.traceHalo = meshC.HaloTraceSetup(Nfields);

  //setup boundary flags and make mask and masked ogs
  elliptic.BoundarySetup();

  //tau (penalty term in IPDG)
  if (settings.compareSetting("DISCRETIZATION","IPDG")) {
    if (meshC.elementType==Mesh::TRIANGLES ||
        meshC.elementType==Mesh::QUADRILATERALS){
      elliptic.tau = 2.0*(meshC.N+1)*(meshC.N+2)/2.0;
      if(meshC.dim==3) {
        elliptic.tau *= 1.5;
      }
    } else {
      elliptic.tau = 2.0*(meshC.N+1)*(meshC.N+3);
    }
  }

  // set mesh.o_EXYZ
  memory<dfloat> EXYZ(meshC.Nelements * meshC.Nverts * meshC.dim);
  int offset;
  for(int e=0; e<meshC.Nelements; e++){
    offset = e*meshC.Nverts;
    for(int i=0;i<meshC.Nverts; i++){
      EXYZ[offset*meshC.dim + i] = meshC.EX[offset + i];
      EXYZ[offset*meshC.dim + meshC.Nverts + i] = meshC.EY[offset + i];
      EXYZ[offset*meshC.dim + meshC.Nverts*2 + i] = meshC.EZ[offset + i];
      // EXYZ[offset*mesh.dim + i] = static_cast<pfloat>(mesh.EX[offset + i]);
      // EXYZ[offset*mesh.dim + mesh.Nverts + i] = static_cast<pfloat>(mesh.EY[offset + i]);
      // EXYZ[offset*mesh.dim + mesh.Nverts*2 + i] = static_cast<pfloat>(mesh.EZ[offset + i]);
    }
  }
  elliptic.mesh.o_EXYZ =  elliptic.platform.malloc<dfloat>(EXYZ);
  elliptic.mesh.o_pfloat_EXYZ =  elliptic.platform.malloc<pfloat>(meshC.Nelements * meshC.Nverts * meshC.dim);
  platform.linAlg().d2p(meshC.Nelements * meshC.Nverts * meshC.dim, elliptic.mesh.o_EXYZ, elliptic.mesh.o_pfloat_EXYZ, 0);

  // set mesh.o_gllzw
  memory<dfloat> gllzw(2*meshC.Nq);
  for(int i=0;i<meshC.Nq;i++){
    gllzw[i] = meshC.gllz[i];
    gllzw[i+meshC.Nq] = meshC.gllw[i];    
  }
  elliptic.mesh.o_gllzw =  elliptic.platform.malloc<dfloat>(gllzw);
  elliptic.mesh.o_pfloat_gllzw =  elliptic.platform.malloc<pfloat>(2 * meshC.Nq);
  platform.linAlg().d2p(2*meshC.Nq, elliptic.mesh.o_gllzw, elliptic.mesh.o_pfloat_gllzw, 0);

  int base;
  int id;
  int Nggeo = meshC.Nggeo;
  int Nvgeo = meshC.Nvgeo;
  int Nverts = meshC.Nverts;
  int Nq = meshC.Nq;
  int Np = meshC.Np;
  memory<dfloat> ggeoV(meshC.Nelements * (Nggeo+1) * Nverts);
  int index(int offset, int Nq, int i);
  for(int e=0;e<meshC.Nelements;++e){
    for(int i=0;i<Nggeo;++i){
      base = e * (Nggeo+1) * Nverts + i*Nverts;
      offset = e * Nggeo * Np + i * Np;
      for (int k=0;k<Nverts;++k){
        id = index(offset, Nq, k);
        ggeoV[base + k] = mesh.ggeoNoW[id];
      }
    }
    base = e * (Nggeo+1) * Nverts + Nggeo*Nverts;
    for (int k=0;k<Nverts;++k){
      id = index(0, Nq, k);
      ggeoV[base + k] = mesh.wJ[id];
    }
  }
  elliptic.mesh.o_ggeoV =  elliptic.platform.malloc<dfloat>(ggeoV);
  elliptic.mesh.o_pfloat_ggeoV =  elliptic.platform.malloc<pfloat>(meshC.Nelements * (Nggeo+1) * Nverts);
  platform.linAlg().d2p(meshC.Nelements * (Nggeo+1) * Nverts, elliptic.mesh.o_ggeoV, elliptic.mesh.o_pfloat_ggeoV, 0);
       
  // 10 vgeo in total, 9 partial diff, plus J
  memory<dfloat> vgeoV(meshC.Nelements * (9+1) * Nverts);
  for(int e=0;e<mesh.Nelements;++e){
    for(int i=0;i<9;++i){
        base = e * (9+1) * Nverts + i*Nverts;
      offset = e * Nvgeo * Np + i * Np;
      for(int k=0;k<Nverts; ++k){
	id = index(offset, Nq, k);
	vgeoV[base + k] = mesh.vgeo[id];
      }
    }
    base = e * (9+1) * Nverts + 9*Nverts;
    offset =  e * Nvgeo * Np + 9 * Np;
    for(int k=0;k<Nverts; ++k){
      id = index(offset, Nq, k);
      vgeoV[base + k] = mesh.vgeo[id];
    }
  }
  elliptic.mesh.o_vgeoV =  elliptic.platform.malloc<dfloat>(vgeoV);
  elliptic.mesh.o_pfloat_vgeoV =  elliptic.platform.malloc<pfloat>(meshC.Nelements * (9+1) * Nverts);
  platform.linAlg().d2p(meshC.Nelements * (9+1) * Nverts, elliptic.mesh.o_vgeoV, elliptic.mesh.o_pfloat_vgeoV, 0);

  // reorder localGlobalElementlist
  meshC.o_localGatherElementList = mesh.o_localGatherElementList;
  meshC.localGatherElementList = mesh.localGatherElementList;

  // construct o_GlobalToLocalColor  
  memory<dlong> GlobalToLocalColor(meshC.Nelements * meshC.Np);
  // first reorder
  for(int e = 0; e< meshC.Nelements; ++e){
    for(int i=0; i<meshC.Np; ++i){
      int newId = meshC.Np * e + i;
      GlobalToLocalColor[newId] = elliptic.GlobalToLocal[newId];
    }
  }
  // then map some global index to -2-gid
  memory<int> isFirst(meshC.Nelements*meshC.Np);
  for(int n=0;n<(meshC.Nelements*meshC.Np);++n){
    isFirst[n] = 1;
  }
  for(int e = 0; e< meshC.Nelements; ++e){
    int element = meshC.localGatherElementList[e];
    //    printf("element = %d ", element);  
    for(int i=0; i<meshC.Np; ++i){
      id = meshC.Np * element + i;
      dlong gid = GlobalToLocalColor[id];
      if(gid>=0){
	if(isFirst[gid]==1){
	  isFirst[gid] = 0;
	  // otherwise do nothing
	}else{ // boundary nodes is not affected by the mapping, gid=-1, gid=-2-gid=-1
	  gid = -2-gid;
	  GlobalToLocalColor[id] = gid;
	}
	//	printf(" %d ", gid);
      }
    }
    //    printf("\n");  
  }
  //  printf("meshC.Np = %d\n", meshC.Np);
  elliptic.o_GlobalToLocalColor = elliptic.platform.malloc<dlong>(GlobalToLocalColor);
  
  // OCCA build stuff
  properties_t kernelInfo = meshC.props; //copy base occa properties

  // set kernel name suffix
  std::string suffix = mesh.elementSuffix();
  std::string oklFilePrefix = DELLIPTIC "/okl/";
  std::string oklFileSuffix = ".okl";

  std::string fileName, kernelName, kernelNameDegree,  kernelNameDegreeDouble, kernelNameDegreeFloat, kernelNameDegreeHalf;

  //add standard boundary functions
  std::string boundaryHeaderFileName;
  if (meshC.dim==2)
    boundaryHeaderFileName = std::string(DELLIPTIC "/data/ellipticBoundary2D.h");
  else if (meshC.dim==3)
    boundaryHeaderFileName = std::string(DELLIPTIC "/data/ellipticBoundary3D.h");
  kernelInfo["includes"] += boundaryHeaderFileName;

  int blockMax = 256;
  if (platform.device.mode() == "CUDA") blockMax = 512;

  int NblockV = std::max(1,blockMax/meshC.Np);
  kernelInfo["defines/" "p_NblockV"]= NblockV;
  kernelInfo["defines/" "ncLoad"]=  "__ldcs";
  kernelInfo["defines/" "ncStore"] = "__stcs";
  kernelInfo["defines/" "p_Nloads"] = (int)p_Nloads;
  kernelInfo["defines/" "p_Nwarp"] = (int)p_Nwarp;
  kernelInfo["defines/" "p_warpSize"] = (int)p_warpSize;
  kernelInfo["defines/" "p_blockSize"] = (int)PCG_BLOCKSIZE;
  kernelInfo["defines/TAp"] = "double";
  kernelInfo["defines/Tp"] = "double";
  kernelInfo["defines/Tggeo"] = "double";
  // kernelInfo["defines/Tx"] = "double";
  // kernelInfo["defines/Tr"] = "double";
  // kernelInfo["defines/Tz"] = "double";
  
  properties_t kernelInfoDouble = kernelInfo;
  kernelInfoDouble["defines/dfloat"]= "double";
  kernelInfoDouble["defines/dfloat4"]= "double4";

  properties_t kernelInfoFloat = kernelInfo;
  kernelInfoFloat["defines/dfloat"]= "float";
  kernelInfoFloat["defines/dfloat4"]= "float4";

  // Ax kernel
  if (settings.compareSetting("DISCRETIZATION","CONTINUOUS")) {
    fileName   = oklFilePrefix + "ellipticAx" + suffix + oklFileSuffix;
    if(meshC.elementType==Mesh::HEXAHEDRA){
      if(mesh.settings.compareSetting("ELEMENT MAP", "TRILINEAR"))
        kernelName = "ellipticPartialAxTrilinear" + suffix;
      else
        kernelName = "ellipticPartialAx" + suffix;
    } else{
      kernelName = "ellipticPartialAx" + suffix;
    }

    elliptic.partialAxKernel = platform.buildKernel(fileName, kernelName,
                                                    kernelInfoDouble);

    elliptic.floatPartialAxKernel = platform.buildKernel(fileName, kernelName,
                                                         kernelInfoFloat);

    kernelName = "ellipticAxGather" + suffix;
    kernelNameDegree = kernelName + std::string("N") + std::to_string(meshC.N);
    kernelInfoDouble["defines/"+ kernelName] = kernelNameDegree;
    elliptic.AxGatherKernel = platform.buildKernel(fileName, kernelNameDegree,
                                           kernelInfoDouble);
    kernelInfoFloat["defines/"+ kernelName] = kernelNameDegree;
    elliptic.floatAxGatherKernel = platform.buildKernel(fileName, kernelNameDegree,
                                                kernelInfoFloat);

    kernelName = "ellipticAxGatherSmooth" + suffix;
    kernelNameDegree = kernelName + std::string("N") + std::to_string(meshC.N);
    kernelInfoDouble["defines/"+ kernelName] = kernelNameDegree;
    elliptic.AxGatherSmoothKernel = platform.buildKernel(fileName, kernelNameDegree,
                                           kernelInfoDouble);
    kernelInfoFloat["defines/"+ kernelName] = kernelNameDegree;
    elliptic.floatAxGatherSmoothKernel = platform.buildKernel(fileName, kernelNameDegree,
                                                kernelInfoFloat);

    // kernelName = "ellipticAxGatherDot" + suffix;
    // elliptic.AxGatherDotKernel = platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // elliptic.floatAxGatherDotKernel = platform.buildKernel(fileName, kernelName,
    //                                             kernelInfoFloat);

    // kernelName = "ellipticAxGatherDotnc" + suffix;
    // elliptic.AxGatherDotncKernel = platform.buildKernel(fileName, kernelName,
    //                                        kernelInfoDouble);
    // elliptic.floatAxGatherDotncKernel = platform.buildKernel(fileName, kernelName,
    //                                             kernelInfoFloat);


    kernelName = "ellipticAxGathernc" + suffix;
    kernelNameDegreeDouble = kernelName + std::string("Double") + std::string("N") + std::to_string(meshC.N);
    kernelNameDegreeFloat = kernelName + std::string("Float") + std::string("N") + std::to_string(meshC.N);
    kernelInfoDouble["defines/"+ kernelName] = kernelNameDegreeDouble;
    kernelInfoFloat["defines/"+ kernelName] = kernelNameDegreeFloat;
    elliptic.AxGatherncKernel = platform.buildKernel(fileName, kernelNameDegreeDouble,
                                           kernelInfoDouble);
    elliptic.floatAxGatherncKernel = platform.buildKernel(fileName, kernelNameDegreeFloat,
                                                kernelInfoFloat);



    fileName = oklFilePrefix + "ellipticAxTrilinear" + suffix + oklFileSuffix;
    kernelName = "ellipticAxTrilinearGather" + suffix;
    kernelNameDegreeDouble = kernelName + std::string("Double") + std::string("N") + std::to_string(meshC.N);
    kernelNameDegreeFloat = kernelName + std::string("Float") + std::string("N") + std::to_string(meshC.N);
    kernelInfoDouble["defines/"+ kernelName] = kernelNameDegreeDouble;
    kernelInfoFloat["defines/"+ kernelName] = kernelNameDegreeFloat;
    elliptic.AxTrilinearGatherKernel = platform.buildKernel(fileName, kernelNameDegreeDouble,
                                           kernelInfoDouble);
    elliptic.floatAxTrilinearGatherKernel = platform.buildKernel(fileName, kernelNameDegreeFloat,
                                                kernelInfoFloat);

    kernelName = "ellipticAxTrilinearGatherSmooth" + suffix;
    kernelNameDegreeDouble = kernelName + std::string("Double") + std::string("N") + std::to_string(meshC.N);
    kernelNameDegreeFloat = kernelName + std::string("Float") + std::string("N") + std::to_string(meshC.N);
    kernelInfoDouble["defines/"+ kernelName] = kernelNameDegreeDouble;
    kernelInfoFloat["defines/"+ kernelName] = kernelNameDegreeFloat;
    elliptic.AxTrilinearGatherSmoothKernel = platform.buildKernel(fileName, kernelNameDegreeDouble,
                                           kernelInfoDouble);
    elliptic.floatAxTrilinearGatherSmoothKernel = platform.buildKernel(fileName, kernelNameDegreeFloat,
                                                kernelInfoFloat);

    kernelName = "ellipticAxInterpGather" + suffix;
    kernelNameDegreeDouble = kernelName + std::string("Double") + std::string("N") + std::to_string(meshC.N);
    kernelNameDegreeFloat = kernelName + std::string("Float") + std::string("N") + std::to_string(meshC.N);
    kernelInfoDouble["defines/"+ kernelName] = kernelNameDegreeDouble;
    kernelInfoFloat["defines/"+ kernelName] = kernelNameDegreeFloat;
    elliptic.AxInterpGatherKernel = platform.buildKernel(fileName, kernelNameDegreeDouble,
                                           kernelInfoDouble);
    elliptic.floatAxInterpGatherKernel = platform.buildKernel(fileName, kernelNameDegreeFloat,
                                                kernelInfoFloat);

    kernelName = "ellipticAxInterpGatherSmooth" + suffix;
    kernelNameDegreeDouble = kernelName + std::string("Double") + std::string("N") + std::to_string(meshC.N);
    kernelNameDegreeFloat = kernelName + std::string("Float") + std::string("N") + std::to_string(meshC.N);
    kernelInfoDouble["defines/"+ kernelName] = kernelNameDegreeDouble;
    kernelInfoFloat["defines/"+ kernelName] = kernelNameDegreeFloat;
    elliptic.AxInterpGatherSmoothKernel = platform.buildKernel(fileName, kernelNameDegreeDouble,
                                           kernelInfoDouble);
    elliptic.floatAxInterpGatherSmoothKernel = platform.buildKernel(fileName, kernelNameDegreeFloat,
                                                kernelInfoFloat);


    properties_t tmpKernelInfo = kernelInfo;
    tmpKernelInfo["okl/enabled"] = false;
    tmpKernelInfo["defines/pfloat2hfloat"] = pfloat2hfloatString;
    tmpKernelInfo["defines/hfloat2pfloat"] = hfloat2pfloatString;  
    fileName = oklFilePrefix + "linAlgVector.cu";
    elliptic.p2hKernel = platform.buildKernel(fileName, "p2h", tmpKernelInfo);    
    elliptic.h2pKernel = platform.buildKernel(fileName, "h2p", tmpKernelInfo);    
    elliptic.norm2Kernel = platform.buildKernel(fileName, "norm2", tmpKernelInfo);    
    elliptic.setKernel = platform.buildKernel(fileName, "set", tmpKernelInfo);    
    elliptic.scaleP2hHalfKernel = platform.buildKernel(fileName, "scaleP2h", tmpKernelInfo);    

    fileName = oklFilePrefix + "multigridUpdateCheby.cu";
    elliptic.zamxHalfKernel = platform.buildKernel(fileName, "zamx", tmpKernelInfo);    
    
    fileName = oklFilePrefix + "multigridUpdateCheby.cu";
    elliptic.zamxHalfKernel = platform.buildKernel(fileName, "zamx", tmpKernelInfo);    
    elliptic.updateCheby4HalfKernel = platform.buildKernel(fileName, "updateCheby4", tmpKernelInfo);    
    elliptic.updateCheby1HalfKernel = platform.buildKernel(fileName, "updateCheby1", tmpKernelInfo);    
    elliptic.updateCheby5HalfKernel = platform.buildKernel(fileName, "updateCheby5", tmpKernelInfo);    
    
    properties_t kernelInfoHalf = kernelInfo;
    kernelInfoHalf["okl/enabled"] = false;
    kernelInfoHalf["defines/pfloat2hfloat"] = pfloat2hfloatString;
    kernelInfoHalf["defines/hfloat2pfloat"] = hfloat2pfloatString;
    fileName = oklFilePrefix + "ellipticAxTrilinear" + suffix + ".cu";

    kernelName = "ellipticAxTrilinearGatherSmooth" + suffix;
    kernelNameDegreeHalf = kernelName + std::string("Half") +  std::string("N") + std::to_string(meshC.N);
    kernelInfoHalf["defines/"+ kernelName] = kernelNameDegreeHalf;
    elliptic.halfAxTrilinearGatherSmoothKernel = platform.buildKernel(fileName, kernelNameDegreeHalf,
							   kernelInfoHalf);
    kernelName = "ellipticAxTrilinearGather" + suffix;
    kernelNameDegreeHalf = kernelName + std::string("Half") +  std::string("N") + std::to_string(meshC.N);
    kernelInfoHalf["defines/"+ kernelName] = kernelNameDegreeHalf;
    elliptic.halfAxTrilinearGatherKernel = platform.buildKernel(fileName, kernelNameDegreeHalf,
							   kernelInfoHalf);
  

    
  } else if (settings.compareSetting("DISCRETIZATION","IPDG")) {
    int Nmax = std::max(meshC.Np, meshC.Nfaces*meshC.Nfp);
    kernelInfoDouble["defines/" "p_Nmax"]= Nmax;
    kernelInfoFloat["defines/p_Nmax"]= Nmax;
    fileName   = oklFilePrefix + "ellipticGradient" + suffix + oklFileSuffix;
    kernelName = "ellipticPartialGradient" + suffix;
    elliptic.partialGradientKernel = platform.buildKernel(fileName, kernelName,
                                                          kernelInfoDouble);

    elliptic.floatPartialGradientKernel = platform.buildKernel(fileName, kernelName,
                                                               kernelInfoFloat);


    fileName   = oklFilePrefix + "ellipticAxIpdg" + suffix + oklFileSuffix;
    kernelName = "ellipticPartialAxIpdg" + suffix;
    elliptic.partialIpdgKernel = platform.buildKernel(fileName, kernelName,
                                                      kernelInfoDouble);
    elliptic.floatPartialIpdgKernel = platform.buildKernel(fileName, kernelName,
                                                           kernelInfoFloat);
  }

  if (settings.compareSetting("DISCRETIZATION", "CONTINUOUS")) {
    elliptic.Ndofs = elliptic.ogsMasked.Ngather*Nfields;
    elliptic.Nhalo = elliptic.gHalo.Nhalo*Nfields;
  } else {
    elliptic.Ndofs = meshC.Nelements*meshC.Np*Nfields;
    elliptic.Nhalo = meshC.totalHaloPairs*meshC.Np*Nfields;
  }

  elliptic.precon = precon_t();

  return elliptic;
}
