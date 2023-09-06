#ifndef CUDA_MYPT_SAMPLING_CUH
#define CUDA_MYPT_SAMPLING_CUH

#include "struct_Vec.cuh"
#include "global_values.cuh"
#include "Disney_BRDF.cuh"

// BXDFの係数列の和をここではMAXとかく．
// (0, MAX)の乱数Rを用いて，BXDFの累積和を保存している配列に対して二分探索を行う．
// 即ちR以上の最も小さい要素を差すindex番号をreturnする．
__device__ inline int sample_BXDF(Xorshift *rand, const float* BXDF_coefficient, int BXDF_coefficient_size, float &probability_of_this_sample) {
    float R = random_rangeMinMax(rand->seed, 0.0f, 1.0f);

    // BXDFの係数の累積和を取る
    float BXDF_coefficient_Sum[5];
    BXDF_coefficient_Sum[0] = BXDF_coefficient[0];
    for(int i = 1; i < BXDF_coefficient_size; i++) {
        BXDF_coefficient_Sum[i] += BXDF_coefficient[i-1];
    }

    R *= BXDF_coefficient_Sum[BXDF_coefficient_size-1];
    int sampled_index_number = lower_bound(BXDF_coefficient_Sum, BXDF_coefficient_size, R);
    probability_of_this_sample = (BXDF_coefficient_Sum[sampled_index_number] - BXDF_coefficient_Sum[sampled_index_number-1]) / BXDF_coefficient[BXDF_coefficient_size-1];

    return sampled_index_number;
}

__device__ inline Ray SampleNextRay(Xorshift *rand, Vec &ThroughPut, const Material CurrentMat, const Ray CurrentRay, const HitPoint CurrentHitPoint,
                             const Vec normal_vector, const Vec normal_vector_NotSmoothed, float aspect, float alpha_sheen, float alpha_specular_x,
                             float alpha_specular_y, float alpha_clearcoat, const float* BRDF_weight_array) {
    Ray NextRay = {{0,0,0}, {0,0,0}, 0};
    // レイと面の交差点における法線ベクトルを高さとする正規直行基底を生成(u,v,w)
    Vec u, v, w;
    GenerateONBFromNormal(normal_vector, u, v, w);

    // Lambert反射モデル----------------------
    if(CurrentMat.MATERIAL_TYPE_ID == 0) {
        NextRay.dir = sample_ray_cosine_weighted(rand, u, v, w);
        NextRay.org = CurrentHitPoint.position;
        ThroughPut = multiply(CurrentMat.diffuse, ThroughPut);
    }
    else if(CurrentMat.MATERIAL_TYPE_ID == 1) {
        // BRDFを選択する
        // 0: diffuse
        // 1: subsurface
        // 2: sheen
        // 3: specular
        // 4: clearcoat
        float probability_of_this_BRDF_sample;
        int sampled_BRDF_ID = sample_BXDF(rand, BRDF_weight_array, 5, probability_of_this_BRDF_sample);
        sample_ray_Disney_BRDF(rand,sampled_BRDF_ID, CurrentRay, NextRay, u, v, w, CurrentHitPoint,
                               alpha_sheen, alpha_specular_x, alpha_specular_y, alpha_clearcoat);
        NextRay.org = CurrentHitPoint.position;
        // BRDFの評価
        Vec Disney_BRDF = evaluate_Disney_BRDF(BRDF_weight_array, CurrentMat, NextRay.dir, -1.0*CurrentRay.dir, normal_vector, u, v, w,
                                               alpha_specular_x, alpha_specular_y, alpha_clearcoat);
        // pdfの評価
        float pdf = evaluate_Disney_BRDF_pdf(BRDF_weight_array, -1*CurrentRay.dir, NextRay.dir, normal_vector, u, v, w,
                                              alpha_specular_x, alpha_specular_y, alpha_sheen, alpha_clearcoat);
        ThroughPut = multiply(ThroughPut, Disney_BRDF * absdot(normal_vector, NextRay.dir) / pdf);
    }

    return NextRay;
}

#endif //CUDA_MYPT_SAMPLING_CUH
