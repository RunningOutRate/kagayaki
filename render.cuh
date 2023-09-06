#ifndef CUDA_MYPT_RENDER_CUH
#define CUDA_MYPT_RENDER_CUH

#include "struct_Ray.cuh"
#include "struct_Camera.cuh"
#include "struct_Xorshift.cuh"
#include "struct_SVGFBuffers.cuh"
#include "RayTrace.cuh"
#include "SVGF.cuh"

/*
 * レンダリング処理を行う
 * 各ピクセルごとに以下の独立の処理を行うのでここをCUDAで並列処理する
 *
 * 1: レイの生成(RayGeneration)
 * 2: レイの発射(RayTrace)
 * 3: 結果を画像バッファに記録
 */
__device__ unsigned long long randSeeds[2000*2000];

__global__ void render(Camera* PrevCAM, Camera* CAM, Vec* RESULT, Triangle* Objects, const Material* Materials, const Texture* Textures, const BVH_node_Device* BVH_tree,
                       const float* LightStrengthSum_FaceLight, const float* LightStrengthSum_HDR, const Texture* HDRData, SVGFBuffer* SVGFBuffer,
                       const int LoopID, const int OneLoopSize) {
    // 1: レイの生成(RayGeneration)
    int pos = blockIdx.x*blockDim.x + threadIdx.x + OneLoopSize * LoopID;
    int Xpos = pos%image_sizeX_device;
    int Ypos = pos/image_sizeX_device;
    SVGFBuffer[pos].init();
    Xorshift rand(randSeeds[pos] + 142857*pos + 998244353/pos + 1000000007%pos);
    Ray FirstRay = Generate1stRay(CAM, Xpos, Ypos);
    Vec RESULTofThisPos = {0,0,0};
    // 2: レイの発射(RayTrace)
    // 3: 結果を画像バッファに記録
    for(int i = 0; i < sample; i++) {
        //printf("sample %d done.\n", i);
        RESULTofThisPos = RESULTofThisPos + RayTrace(PrevCAM, FirstRay, &rand, Objects, Materials, Textures, BVH_tree, LightStrengthSum_FaceLight, LightStrengthSum_HDR,
                                                    HDRData, SVGFBuffer, pos);
        //RESULTofThisPos = RESULTofThisPos + Vec{0.2,0.2,0.2};
       //RESULT[pos] = RESULT[pos] + Vec{0.2,0.2,0.2};
    }

    randSeeds[pos] = rand.seed;
    //printf("position (%d, %d) done.\n", Xpos, Ypos);
    RESULTofThisPos = RESULTofThisPos / (float)sample;
    //RESULTofThisPos = {(float)Xpos/(float)image_sizeX_device, (float)Ypos/(float)image_sizeY_device, 0};

    RESULT[pos] = RESULTofThisPos;
    __syncthreads();
}

#endif //CUDA_MYPT_RENDER_CUH
