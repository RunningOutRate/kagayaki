#ifndef CUDA_MYPT_NEXTEVENTESTIMATION_HDR_CUH
#define CUDA_MYPT_NEXTEVENTESTIMATION_HDR_CUH

__device__ void NextEventEstimation_HDR(Vec &Radiance, Vec ThroughPut, Xorshift *rand, const Ray CurrentRay, const Material CurrentMat,
                                        const HitPoint CurrentHitPoint, const Triangle* Objects, const Material* Materials, const Vec Normal, const Vec NormalNotSmoothed,
                                        const BVH_node_Device *BVH_Tree, const float* LightStrengthSum_FaceLight, const float* LightStrengthSum_HDR, const Texture* HDRData,
                                        float aspect, float alpha_sheen, float alpha_specular_x, float alpha_specular_y, float alpha_clearcoat, const float* BRDF_weight_array) {
    // HDRが存在する場合はHDRに対するNEEを行う-------------------------------------------------------------------------------
    if(Exists_NoHDR) {
        return;
    }

    // HDR画素上への方向をサンプリング
    const int SampledHDRPixel = choose_HDR_pixel(random_rangeMinMax(rand->seed, 0, 1), LightStrengthSum_HDR);
    float pdf_NEE_HDR;
    if(SampledHDRPixel == 0) pdf_NEE_HDR = LightStrengthSum_HDR[0] / LightStrengthSum_HDR[Light_strength_sum_HDR_size - 1];
    else pdf_NEE_HDR = (LightStrengthSum_HDR[SampledHDRPixel] - LightStrengthSum_HDR[SampledHDRPixel - 1]) / LightStrengthSum_HDR[Light_strength_sum_HDR_size - 1];

    Vec ShadowRayToHDR_Dir = convert_HDRindex_to_Dir(SampledHDRPixel, pdf_NEE_HDR, HDRData);
    Ray ShadowRayToHDR(ShadowRayToHDR_Dir, CurrentHitPoint.position, 0);

    // HDRの方もシャドウレイを飛ばす．何とも交差しない場合について考える．また，現在の面の，入射側とは裏方向にレイを飛ばさないように制限する
    Intersection ShadowRayToHDRIntersection;
    if (dot(ShadowRayToHDR_Dir, Normal) <= 0 || willIntersectWithTriangle_nonRecursive(ShadowRayToHDR, &ShadowRayToHDRIntersection, ObjectsNum, Objects, BVH_Tree)) {
        return;
    }
    // HDRの持つ輝度
    Vec HDR_emission = Vec(HDRData->pixels[SampledHDRPixel*HDRData->bpp],
                           HDRData->pixels[SampledHDRPixel*HDRData->bpp + 1],
                           HDRData->pixels[SampledHDRPixel*HDRData->bpp + 2]);
    Vec BSDF = {0,0,0};

    // Multi Importance Sampling: このレイをptが飛ばす確率とneeが飛ばす確率
    float probability_pt = 0;
    // float probability_nee_facelight = 0; 何とも交差していない時点で絶対に面光源NEEがこの点をサンプリングすることはないと分かる(光源面と交差することになるはずである)
    float probability_nee_HDR = 0;
    if(CurrentMat.MATERIAL_TYPE_ID == 0) {
        BSDF = CurrentMat.diffuse / PI;
        probability_pt = calculate_Lambert_pdf(Normal, ShadowRayToHDR_Dir); // [/sr]
    }
    else if(CurrentMat.MATERIAL_TYPE_ID == 1) {
        // レイと面の交差点における法線ベクトルを高さとする正規直行基底を生成(u,v,w)
        Vec u, v, w;
        GenerateONBFromNormal(Normal, u, v, w);
        BSDF = evaluate_Disney_BRDF(BRDF_weight_array, CurrentMat, ShadowRayToHDR.dir, -1.0*CurrentRay.dir, Normal, u, v, w,
                                    alpha_specular_x, alpha_specular_y, alpha_clearcoat);
        probability_pt = evaluate_Disney_BRDF_pdf(BRDF_weight_array, -1*CurrentRay.dir, ShadowRayToHDR.dir, Normal, u, v, w,
                                                  alpha_specular_x, alpha_specular_y, alpha_sheen, alpha_clearcoat);
    }
    probability_nee_HDR = pdf_NEE_HDR; // [/sr]

    float MISWeight_NEEHDR = probability_nee_HDR / (probability_pt + probability_nee_HDR);

    Vec Radiance_NEEHDR = multiply(BSDF, HDR_emission / pdf_NEE_HDR) * dot(Normal, ShadowRayToHDR_Dir);
    Radiance_NEEHDR = MISWeight_NEEHDR * Radiance_NEEHDR;

    Radiance = Radiance + multiply(Radiance_NEEHDR, ThroughPut);
}

#endif //CUDA_MYPT_NEXTEVENTESTIMATION_HDR_CUH
