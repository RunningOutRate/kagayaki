#ifndef CUDA_MYPT_NEXTEVENTESTIMATION_PATHTRACING_CUH
#define CUDA_MYPT_NEXTEVENTESTIMATION_PATHTRACING_CUH

__device__ void NextEventEstimation_PathTracing (Vec &Radiance, Vec ThroughPut, Xorshift *rand, const Ray CurrentRay, const Material CurrentMat,
                                                 const HitPoint CurrentHitPoint, const Triangle* Objects, const Material* Materials, const Vec Normal, const Vec NormalNotSmoothed,
                                                 const BVH_node_Device *BVH_Tree, const float* LightStrengthSum_FaceLight, const float* LightStrengthSum_HDR, const Texture* HDRData,
                                                 float aspect, float alpha_sheen, float alpha_specular_x, float alpha_specular_y, float alpha_clearcoat, const float* BRDF_weight_array) {
    // PathTracingも行う-------------------------------------------------------------------------------------------------
    // 適していない場合をはじく
    if((CurrentMat.MATERIAL_TYPE_ID != 0)) {
        return;
    }
    // SampleNextRayはTroughPutを更新するため，そうされないように別の変数にコピーしていれる
    Vec ThroughPut_InputOfSampleNextRay = ThroughPut;
    Ray ShadowRayPathTracing = SampleNextRay(rand, ThroughPut_InputOfSampleNextRay, CurrentMat, CurrentRay, CurrentHitPoint, Normal, NormalNotSmoothed,
                                             aspect, alpha_sheen, alpha_specular_x, alpha_specular_y, alpha_clearcoat, BRDF_weight_array);
    Intersection ShadowRayPathTracingIntersection;
    if(dot(ShadowRayPathTracing.dir, Normal) <= 0 || willIntersectWithTriangle_nonRecursive(ShadowRayPathTracing, &ShadowRayPathTracingIntersection, ObjectsNum, Objects, BVH_Tree) ||
       Materials[Objects[ShadowRayPathTracingIntersection.face_id].material_id].emission.length() == 0) {
        return;
    }
    // G項を計算する
    float cos1 = abs(dot(CurrentHitPoint.normal, ShadowRayPathTracing.dir)) , cos2 = abs(dot(ShadowRayPathTracingIntersection.hitpoint.normal, -1.0f * ShadowRayPathTracing.dir));
    float G_PT = cos1 * cos2 / (ShadowRayPathTracingIntersection.hitpoint.distance * ShadowRayPathTracingIntersection.hitpoint.distance);
    // pdfを計算する. pdf[1/m^2] := (その光源indexを選ぶ確率) * (1/その光源の面積)
    float probability_nee;
    if (ShadowRayPathTracingIntersection.face_id == 0) {
        probability_nee = (LightStrengthSum_FaceLight[0] / LightStrengthSum_FaceLight[Light_strength_sum_size - 1])
                          * (1.0f / Objects[0].area()); // 光源が1つめの時の範囲外参照を防ぐ
    }
    else {
        probability_nee = ((LightStrengthSum_FaceLight[ShadowRayPathTracingIntersection.face_id] - LightStrengthSum_FaceLight[ShadowRayPathTracingIntersection.face_id - 1]) /
                           LightStrengthSum_FaceLight[Light_strength_sum_size - 1])
                          * (1.0f / Objects[ShadowRayPathTracingIntersection.face_id].area());
    }

    // Multi Importance Sampling: このレイをptが飛ばす確率とneeが飛ばす確率
    float probability_pt = 0;
    float probability_nee_HDR = 0;
    if(!Exists_NoHDR) probability_nee_HDR = calculate_HDR_pdf(ShadowRayPathTracing.dir, LightStrengthSum_HDR, HDRData);
    Vec BSDF = {0, 0, 0};
    if (CurrentMat.MATERIAL_TYPE_ID == 0) {
        BSDF = (CurrentMat.diffuse)/PI;
        probability_pt = calculate_Lambert_pdf(Normal, ShadowRayPathTracing.dir);
    }
    else if(CurrentMat.MATERIAL_TYPE_ID == 1) {
        // レイと面の交差点における法線ベクトルを高さとする正規直行基底を生成(u,v,w)
        Vec u, v, w;
        GenerateONBFromNormal(Normal, u, v, w);
        BSDF = evaluate_Disney_BRDF(BRDF_weight_array, CurrentMat, ShadowRayPathTracing.dir, -1.0*CurrentRay.dir, Normal, u, v, w,
                                    alpha_specular_x, alpha_specular_y, alpha_clearcoat);
        probability_pt = evaluate_Disney_BRDF_pdf(BRDF_weight_array, -1*CurrentRay.dir, ShadowRayPathTracing.dir, Normal, u, v, w,
                                                  alpha_specular_x, alpha_specular_y, alpha_sheen, alpha_clearcoat);
    }

    probability_nee = probability_nee * (cos1 / G_PT);

    float MISWeight_PathTracing = probability_pt / (probability_nee + probability_pt + probability_nee_HDR);

    Vec Radiance_PathTracing = multiply((BSDF * cos1 / probability_pt),
                                        Materials[Objects[ShadowRayPathTracingIntersection.face_id].material_id].emission);
    Radiance_PathTracing = Radiance_PathTracing * MISWeight_PathTracing;

    Radiance = Radiance + multiply(ThroughPut, Radiance_PathTracing);
}

#endif //CUDA_MYPT_NEXTEVENTESTIMATION_PATHTRACING_CUH
