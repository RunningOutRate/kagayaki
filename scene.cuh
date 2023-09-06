#ifndef CUDA_MYPT_SCENE_CUH
#define CUDA_MYPT_SCENE_CUH

#include "struct_Triangle.cuh"
#include "struct_Texture.cuh"
#include "struct_Material.cuh"

// メッシュデータ
std::vector<Triangle> HOST_ObjectsData;
Triangle* HOST_ObjectsData_Array;
Triangle* DEVICE_ObjectsData;

// マテリアルデータ
std::vector<Material> HOST_MaterialsData = {
        Material(0, -1, {0.75, 0.75, 0.75}, {0, 0, 0}, 0.0, 0.0, 0.0,
                 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.45,
                 0.01, 0.01, 0.5), // 予約マテリアル
};
Material* HOST_MaterialsData_Array;
Material* DEVICE_MaterialsData;

// テクスチャデータ
std::vector<Texture> HOST_TexturesData;
Texture* HOST_TexturesData_Array;
Texture* DEVICE_TexturesData;

// HDRデータ
Texture* HOST_HDRData;
Texture* DEVICE_HDRData;

// NEEに関するデータ
std::vector<float> HOST_Light_strength_sum(0);
float* HOST_Light_strength_sum_Array;
float* DEVICE_Light_strength_sum;
__device__ int Light_strength_sum_size;

std::vector<float> HOST_Light_strength_sum_HDR(0);
float* HOST_Light_strength_sum_HDR_Array;
float* DEVICE_Light_strength_sum_HDR;
__device__ int Light_strength_sum_HDR_size;

#endif //CUDA_MYPT_SCENE_CUH
