#ifndef CUDA_MYPT_RAYTRACE_CUH
#define CUDA_MYPT_RAYTRACE_CUH

#include "struct_Intersection.cuh"
#include "sampling.cuh"
#include "RayTrace_Russian_Roulette.cuh"
#include "Disney_BRDF.cuh"

__device__ inline Vec RayTrace(Camera *PrevCAM, Ray Ray, Xorshift *rand, const Triangle* Objects, const Material* Materials, const Texture* Textures, const BVH_node_Device* BVH_tree,
                        const float* LightStrengthSum_FaceLight, const float* LightStrengthSum_HDR, const Texture* HDRData, SVGFBuffer* SVGFBuffer, int pos) {
    const Vec BACKGROUND_COLOR = {1,1,1};
    int depth = -1;
    Vec ThroughPut = {1,1,1};
    Vec Radiance = {sINF,sINF,sINF};

    for(;;) {
        // depthの更新
        depth++;
        // 交差判定
        Intersection ClosestIntersection;
        if(!willIntersectWithTriangle_nonRecursive(Ray, &ClosestIntersection, ObjectsNum, Objects, BVH_tree)) {
            if(depth == 0) {
                // SVGFBufferに「交差しなかった」ことを記録
                SVGFBuffer[pos].ObjectID = -1;
            }

            // HDRへのアクセス
            if(Exists_NoHDR) {
                Radiance = Radiance + multiply(ThroughPut, BACKGROUND_COLOR);
            }
            else {
                Radiance = Radiance + multiply(ThroughPut, get_HDR_emission(Ray.dir,HDRData));
            }
            break;
        }

        // 交差点における面と，その面のマテリアル
        Triangle face_in_question = Objects[ClosestIntersection.face_id];
        HitPoint hitpoint = ClosestIntersection.hitpoint;
        Material material_in_question = Materials[face_in_question.material_id];
        // 交差点における入射光側に向いている法線ベクトルを求める
        Vec normal_vector, normal_vector_NotSmoothed;
        if(dot(hitpoint.normal, Ray.dir) > 0.0) normal_vector = -1.0*hitpoint.normal;
        else normal_vector = hitpoint.normal;
        if(dot(hitpoint.normal_NotSmoothed, Ray.dir) > 0.0) normal_vector_NotSmoothed = -1.0*hitpoint.normal_NotSmoothed;
        else normal_vector_NotSmoothed = hitpoint.normal_NotSmoothed;

        // テクスチャマッピング
        if(material_in_question.TEXTURE_BASECOLOR_ID != -1) {
            material_in_question.diffuse = Textures[material_in_question.TEXTURE_BASECOLOR_ID].get_Color(hitpoint.u, hitpoint.v);
        }
        // 0ベクトルだと不具合が起きるので微小な値を入れる
        if(material_in_question.diffuse.length() < 1e-5) {
            material_in_question.diffuse = Vec{sINF, sINF, sINF};
        }

        /*
         * SVGFBufferへの記録
         * 1次レイとの交差点における次を記録する
         * 1. Albedo: 反射率？
         * 2. Normal: 法線ベクトル
         * 3. ObjectID: オブジェクトID
         * 4. MaterialID: マテリアルID
         * 5. PosLastFrameI: ひとつ前のフレームにおけるその交差点のあったピクセル座標
         */
        if(depth == 0) {
            SVGFBuffer[pos].Albedo = material_in_question.diffuse;
            SVGFBuffer[pos].Emission = material_in_question.emission;
            SVGFBuffer[pos].Normal = normal_vector;
            SVGFBuffer[pos].ObjectID = ClosestIntersection.face_id;
            SVGFBuffer[pos].MaterialID = face_in_question.material_id;
            SVGFBuffer[pos].Depth = ClosestIntersection.hitpoint.distance;
            Camera_ConvertFromHitPointPosToPixelPos(SVGFBuffer[pos].ObjectID, ClosestIntersection.hitpoint.position,
                                                    PrevCAM, Objects, BVH_tree, SVGFBuffer[pos].PosLastFrameI, SVGFBuffer[pos].PosLastFrameF);
           /* if(SVGFBuffer[pos].PosLastFrameI.x != -1) {
                Radiance = {1,1,1};
            }
            else {
                Radiance = {0,0,0};
            }
            break; */
        }

        // Disney_BRDFでつかう変数
        float aspect = sqrt(1.0f - 0.9f * material_in_question.anisotropic);
        float alpha_sheen = material_in_question.roughness*material_in_question.roughness;
        float alpha_specular_x = max(0.001f, material_in_question.roughness*material_in_question.roughness / aspect);
        float alpha_specular_y = max(0.001f, material_in_question.roughness*material_in_question.roughness * aspect);
        float alpha_clearcoat = (1.0f - material_in_question.cloearcoatGloss) * 0.1f + material_in_question.cloearcoatGloss*0.001f;
        // 各BRDFのウェイトを計算する {diffuse, subsurface, sheen, specular, clearcoat}
        float BRDF_weight_array[5] = {0.0f};
        generate_Disney_BRDF_coefficient_array(material_in_question, BRDF_weight_array);

        // ロシアンルーレット
        if(!Russian_Roulette(Radiance, rand, material_in_question, ThroughPut, depth, 2, 3)) {
            break;
        }

        // NextEventEstimation
        NextEventEstimation(Radiance, ThroughPut, rand, Ray, material_in_question, hitpoint, Objects, Materials, normal_vector, normal_vector_NotSmoothed,
                            BVH_tree, LightStrengthSum_FaceLight, LightStrengthSum_HDR, HDRData,
                            aspect, alpha_sheen, alpha_specular_x, alpha_specular_y, alpha_clearcoat, BRDF_weight_array);

        // 次に飛ばすレイを求める
        Ray = SampleNextRay(rand, ThroughPut, material_in_question, Ray, hitpoint, normal_vector, normal_vector_NotSmoothed,
                            aspect, alpha_sheen, alpha_specular_x, alpha_specular_y, alpha_clearcoat, BRDF_weight_array);
    }
    return Radiance;
}

#endif //CUDA_MYPT_RAYTRACE_CUH
