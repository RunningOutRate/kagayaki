#ifndef CUDA_MYPT_STRUCT_HITPOINT_CUH
#define CUDA_MYPT_STRUCT_HITPOINT_CUH

#include "struct_Vec.cuh"
#include "global_values.cuh"

struct HitPoint{
    float distance;
    Vec position;
    Vec normal;
    Vec normal_NotSmoothed;
    float u, v;
    __device__ HitPoint() : distance(bINF), normal(), normal_NotSmoothed(), position(), u(), v() {}
};

#endif //CUDA_MYPT_STRUCT_HITPOINT_CUH
