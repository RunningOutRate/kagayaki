#ifndef CUDA_MYPT_NEXTEVENTESTIMATION_FACELIGHT_CUH
#define CUDA_MYPT_NEXTEVENTESTIMATION_FACELIGHT_CUH

#include "scene.cuh"
#include "ImageBasedLighting.cuh"

// 累積和を考える．HOST_Light_strength_sum[i] := Objects[i]までのemission*光源面積の和
__host__ void calculate_Lss() {
    if(!HOST_ObjectsData.empty()) {
        HOST_Light_strength_sum.push_back(HOST_ObjectsData[0].area() * HOST_MaterialsData[HOST_ObjectsData[0].material_id].emission.length());
        for (int i = 1; i < HOST_ObjectsData.size(); i++) {
            HOST_Light_strength_sum.push_back(HOST_Light_strength_sum[i - 1] +
                                              HOST_ObjectsData[i].area() * HOST_MaterialsData[HOST_ObjectsData[i].material_id].emission.length());
            //std::cout << "LSS[" << i << "] = " << HOST_Light_strength_sum[i] << std::endl;
        }
        // 光源が無い場合
        if(HOST_Light_strength_sum[HOST_Light_strength_sum.size()-1] == 0) {
            HOST_Exists_NoLight = true;
        }
    }
}

// 光源オブジェクトIDを選ぶ
__device__ int ChooseLightSource(float R, const float* LightStrengthSum_FaceLight) {
    if(Exists_NoLight) return -1;
    // RをLssの値域に対応させる
    R *= LightStrengthSum_FaceLight[Light_strength_sum_size - 1];
    // LssにおいてRより大きな最小の値を持つindexを求める
    return lower_bound(LightStrengthSum_FaceLight, Light_strength_sum_size, R);
}

// 三角面上の点をサンプリングする
__device__ Vec SamplePointOnTriangle(const float r1, const float r2, const Triangle Primitive) {
    const Vec v0 = Primitive.v0;
    const Vec v1 = Primitive.v1;
    const Vec v2 = Primitive.v2;

    float t1 = r1 / 2.0f, t2 = r2 / 2.0f;
    float t_offset = t2 - t1;

    if (t_offset > 0) t2 += t_offset;
    else t1 -= t_offset;

    return (t1 * v0 + t2 * v1 + (1 - t1 - t2) * v2);
}

__device__ void NextEventEstimation_FaceLight(Vec &Radiance, Vec ThroughPut, Xorshift *rand, const Ray CurrentRay, const Material CurrentMat,
                                              const HitPoint CurrentHitPoint, const Triangle* Objects, const Material* Materials, const Vec Normal, const Vec NormalNotSmoothed,
                                              const BVH_node_Device *BVH_Tree, const float* LightStrengthSum_FaceLight, const float* LightStrengthSum_HDR, const Texture* HDRData,
                                              float aspect, float alpha_sheen, float alpha_specular_x, float alpha_specular_y, float alpha_clearcoat, const float* BRDF_weight_array) {
    // NEEに適していない場合は何もしない
    if(CurrentMat.emission.length() != 0 || CurrentMat.MATERIAL_TYPE_ID >= 2 || Light_strength_sum_size == 0) {
        return;
    }

    // 光源上の点をサンプリング
    const int SampledLightID = ChooseLightSource(random_rangeMinMax(rand->seed, 0, 1), LightStrengthSum_FaceLight);
    Vec SampledLightPosition = SamplePointOnTriangle(random_rangeMinMax(rand->seed, 0, 1), random_rangeMinMax(rand->seed, 0, 1), Objects[SampledLightID]);
    // 光源のサンプル点に向かうベクトル
    Vec LightDir = normalize(SampledLightPosition - CurrentHitPoint.position);
    // 交差点から光源のサンプル点までの距離
    const float LightDistance = (SampledLightPosition - CurrentHitPoint.position).length();
    // シャドウレイを生成
    struct Ray ShadowRay(LightDir, CurrentHitPoint.position, 0);
    Intersection ShadowRayIntersection;

    // レイが面の内部に入り込むかどうか and シャドウレイの交差判定
    // shadow_rayがオブジェクトに遮られずに光源までたどり着けない場合
    if (dot(LightDir, Normal) <= 0 || !willIntersectWithTriangle_nonRecursive(ShadowRay, &ShadowRayIntersection, ObjectsNum, Objects, BVH_Tree) ||
        ShadowRayIntersection.face_id != SampledLightID) {
        return;
    }

    // G項を計算する
    float cos1 = abs(dot(CurrentHitPoint.normal, LightDir)) , cos2 = abs(dot(ShadowRayIntersection.hitpoint.normal, -1.0f * LightDir));
    float G_NEE = cos1 * cos2 / (LightDistance * LightDistance);
    // pdfを計算する. pdf[1/m^2] := (その光源indexを選ぶ確率) * (1/その光源の面積)
    float pdf_NEE;
    if (SampledLightID == 0) {
        pdf_NEE = (LightStrengthSum_FaceLight[0] / LightStrengthSum_FaceLight[Light_strength_sum_size - 1])
                  * (1.0f / Objects[0].area()); // 光源が1つめの時の範囲外参照を防ぐ
    }
    else {
        pdf_NEE = ((LightStrengthSum_FaceLight[SampledLightID] - LightStrengthSum_FaceLight[SampledLightID - 1]) /
                   LightStrengthSum_FaceLight[Light_strength_sum_size - 1])
                  * (1.0f / Objects[SampledLightID].area());
    }

    // Multi Importance Sampling: このレイをptが飛ばす確率とneeが飛ばす確率
    float probability_pt = 0;
    float probability_nee_facelight = pdf_NEE;
    float probability_nee_HDR = 0;
    // HDRが存在する時はそのシャドウレイを，HDRに対するNEEで飛ばす確率を求める
    if(!Exists_NoHDR) probability_nee_HDR = calculate_HDR_pdf(ShadowRay.dir, LightStrengthSum_HDR, HDRData);

    Vec BSDF = {0, 0, 0};
    if (CurrentMat.MATERIAL_TYPE_ID == 0) {
        BSDF = (CurrentMat.diffuse)/PI;
        probability_pt = calculate_Lambert_pdf(Normal, ShadowRay.dir);
    }
    else if(CurrentMat.MATERIAL_TYPE_ID == 1) {
        // レイと面の交差点における法線ベクトルを高さとする正規直行基底を生成(u,v,w)
        Vec u, v, w;
        GenerateONBFromNormal(Normal, u, v, w);
        BSDF = evaluate_Disney_BRDF(BRDF_weight_array, CurrentMat, ShadowRay.dir, -1.0*CurrentRay.dir, Normal, u, v, w,
                                    alpha_specular_x, alpha_specular_y, alpha_clearcoat);
        probability_pt = evaluate_Disney_BRDF_pdf(BRDF_weight_array, -1*CurrentRay.dir, ShadowRay.dir, Normal, u, v, w,
                                                  alpha_specular_x, alpha_specular_y, alpha_sheen, alpha_clearcoat);
    }

    probability_pt *= G_NEE / abs(dot(ShadowRay.dir, CurrentHitPoint.normal)); // pdfの単位を合わせる

    float MISWeight_NEEFaceLight = probability_nee_facelight /
                                   (probability_pt + probability_nee_facelight + probability_nee_HDR);
    Vec Radiance_NEEFaceLight = multiply((BSDF * G_NEE / pdf_NEE),
                                         Materials[Objects[SampledLightID].material_id].emission);
    Radiance_NEEFaceLight = MISWeight_NEEFaceLight * Radiance_NEEFaceLight;

    Radiance = Radiance + multiply(ThroughPut,Radiance_NEEFaceLight);
}

#endif //CUDA_MYPT_NEXTEVENTESTIMATION_FACELIGHT_CUH
