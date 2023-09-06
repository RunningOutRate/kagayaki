#ifndef CUDA_MYPT_STRUCT_MATERIAL_CUH
#define CUDA_MYPT_STRUCT_MATERIAL_CUH

#include "struct_Vec.cuh"

struct Material{
    int MATERIAL_TYPE_ID;
    int TEXTURE_BASECOLOR_ID;
    Vec diffuse, emission;
    float subsurface, metallic, specular, specularTint, roughness, anisotropic, sheen, sheenTint, clearcoat, cloearcoatGloss, IOR,
            mu_a, mu_s, scatter_g;

    /*
     * MATERIAL_TYPE_ID
     * 0: Lambert Diffuse
     * 1: Disney BRDF
     *
     * TEXTURE_BASECOLOR_ID
     * -1: no texture
     * 0~: Texture[num]をテクスチャにもつ
     */
     __host__ __device__ Material(int MATERIAL_TYPE_ID = 0, int TEXTURE_BASECOLOR_ID = -1,
             const Vec &diffuse = {0.8, 0.8, 0.8}, const Vec &emission= {0.0, 0.0, 0.0}, const float &subsurface = 0.0,
             const float &specular= 0.0, const float &specularTint= 0.0,
             const float &metallic = 0.0, const float &roughness = 0.00, const float &anisotropic= 0.0, const float &sheen= 0.0, const float &sheenTint = 0.0,
             const float &clearcoat = 0.0, const float &clearcoatGloss = 0.0, const float &IOR = 1.45,
             const float & mu_a = 0.0, const float &mu_s = 0.0, const float &scatter_g = 0.5) :
            MATERIAL_TYPE_ID(MATERIAL_TYPE_ID), TEXTURE_BASECOLOR_ID(TEXTURE_BASECOLOR_ID), diffuse(diffuse), emission(emission), subsurface(subsurface), specular(specular), specularTint(specularTint),
            metallic(metallic), roughness(roughness), anisotropic(anisotropic), sheen(sheen), sheenTint(sheenTint), clearcoat(clearcoat), cloearcoatGloss(clearcoatGloss), IOR(IOR),
            mu_a{mu_a}, mu_s(mu_s), scatter_g(scatter_g) {}
};

#endif //CUDA_MYPT_STRUCT_MATERIAL_CUH
