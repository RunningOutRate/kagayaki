#ifndef CUDA_MYPT_GLOBAL_VALUES_CUH
#define CUDA_MYPT_GLOBAL_VALUES_CUH

#include "struct_Vec.cuh"
#include "struct_BVH_node.cuh"


// 一般的定数
__constant__ float sINF = 1e-4;
__constant__ float bINF = 1e38;
__constant__ float PI = 3.14159265358979323846263383;
__device__ const float DEG2RAD_device = 0.01745329f;
const float DEG2RAD = 0.01745329f;
const float PI_HOST = 3.14159265358979323846263383;

// レンダリングに関する設定
const int image_sizeX = 1920;
const int image_sizeY = 1920/16*9;
const int super_sample = 1;
const int frame_num = 120;

__constant__ int image_sizeX_device = 1920;
__constant__ int image_sizeY_device = 1920/16*9;
__constant__ const int sample = 1;

bool HOST_Exists_NoLight = false;
bool HOST_Exists_NoHDR = false;
__device__ bool Exists_NoLight = false;
__device__ bool Exists_NoHDR = false;

// シーン情報の定数
__device__ const float IOR_air = 1.00;
__device__ int ObjectsNum;

// BVHのデータ
std::vector<BVH_node_Host> HOST_BVH_tree;
BVH_node_Device* HOST_BVH_tree_ConvertedForDevice; // data
BVH_node_Device* DEVICE_BVH_tree; // d_data

#endif //CUDA_MYPT_GLOBAL_VALUES_CUH
