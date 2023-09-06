#ifndef CUDA_MYPT_STRUCT_RAY_CUH
#define CUDA_MYPT_STRUCT_RAY_CUH

#include "struct_Vec.cuh"

/*
 * レイの情報を持つ構造体
 *
 * dir: レイの方向ベクトル
 * org: レイの原点位置ベクトル
 */
struct Ray {
    Vec dir;
    Vec org;
    double _dummy; // 構造体のデータサイズ調整 -> 256byte (不要？)

    __host__ __device__ Ray(const Vec &dir, const Vec &org, const double &_dummy) : org(org), dir(dir), _dummy(_dummy) {}
};

#endif //CUDA_MYPT_STRUCT_RAY_CUH
