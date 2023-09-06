#ifndef CUDA_MYPT_NEXTEVENTESTIMATION_CUH
#define CUDA_MYPT_NEXTEVENTESTIMATION_CUH

#include "Algorithms.cuh"
#include "struct_Xorshift.cuh"
#include "sampling.cuh"

#include "NextEventEstimation_FaceLight.cuh"
#include "NextEventEstimation_HDR.cuh"
#include "NextEventEstimation_PathTracing.cuh"

// NextEventEstimationの処理
__device__ void NextEventEstimation(Vec &Radiance, Vec ThroughPut, Xorshift *rand, const Ray CurrentRay, const Material CurrentMat,
                                    const HitPoint CurrentHitPoint, const Triangle* Objects, const Material* Materials, const Vec Normal, const Vec NormalNotSmoothed,
                                    const BVH_node_Device *BVH_Tree, const float* LightStrengthSum_FaceLight, const float* LightStrengthSum_HDR, const Texture* HDRData,
                                    float aspect, float alpha_sheen, float alpha_specular_x, float alpha_specular_y, float alpha_clearcoat, const float* BRDF_weight_array) {

    NextEventEstimation_FaceLight(Radiance, ThroughPut, rand, CurrentRay, CurrentMat,
                                  CurrentHitPoint, Objects, Materials, Normal, NormalNotSmoothed,
                                  BVH_Tree, LightStrengthSum_FaceLight, LightStrengthSum_HDR, HDRData,
                                  aspect, alpha_sheen, alpha_specular_x, alpha_specular_y, alpha_clearcoat, BRDF_weight_array);

    NextEventEstimation_HDR(Radiance, ThroughPut, rand, CurrentRay, CurrentMat,
                            CurrentHitPoint, Objects, Materials, Normal, NormalNotSmoothed,BVH_Tree,
                            LightStrengthSum_FaceLight, LightStrengthSum_HDR, HDRData,
                            aspect, alpha_sheen, alpha_specular_x, alpha_specular_y, alpha_clearcoat, BRDF_weight_array);

    NextEventEstimation_PathTracing(Radiance, ThroughPut, rand, CurrentRay, CurrentMat,
                                    CurrentHitPoint, Objects, Materials, Normal, NormalNotSmoothed,
                                    BVH_Tree, LightStrengthSum_FaceLight, LightStrengthSum_HDR, HDRData,
                                    aspect, alpha_sheen, alpha_specular_x, alpha_specular_y, alpha_clearcoat, BRDF_weight_array);
}

#endif //CUDA_MYPT_NEXTEVENTESTIMATION_CUH
