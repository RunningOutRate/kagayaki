#ifndef CUDA_MYPT_STRUCT_INTERSECTION_CUH
#define CUDA_MYPT_STRUCT_INTERSECTION_CUH

#include "struct_HitPoint.cuh"

struct Intersection{
    HitPoint hitpoint;
    int face_id;
    __device__ Intersection() : face_id(-1) {};
};

#endif //CUDA_MYPT_STRUCT_INTERSECTION_CUH
