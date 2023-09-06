#ifndef CUDA_MYPT_DISNEY_BRDF_CUH
#define CUDA_MYPT_DISNEY_BRDF_CUH

#include "Algorithms.cuh"
#include "Disney_BRDF_sampling.cuh"
#include "LambertBRDF_sampling.cuh"

// Disney_BRDFのBRDFにかかる各係数の配列を生成する．
__device__ void generate_Disney_BRDF_coefficient_array(Material &mat, float* Coefficient_Array) {
    const float alpha_diffuse = (1 - mat.subsurface)*(1-mat.metallic);
    const float alpha_subsurface = mat.subsurface*(1-mat.metallic);
    const float alpha_sheen = 1-mat.metallic;
    const float alpha_specular = 1.0;
    const float alpha_clearcoat = 1.0;

    Coefficient_Array[0] = alpha_diffuse;
    Coefficient_Array[1] = alpha_subsurface;
    Coefficient_Array[2] = alpha_sheen;
    Coefficient_Array[3] = alpha_specular;
    Coefficient_Array[4] = alpha_clearcoat;
}

__device__ float sqr(float a) {
    return a*a;
}

__device__ float Schlick_Fresnel(float u) {
    float m = 1-u;
    clamp(m, 0.0f, 1.0f);
    return m*m*m*m*m;
}

__device__ Vec Fr_schlick(const Vec F, const Vec &normal, const Vec &V) {
    return F + (Vec{1,1,1} - F) * pow(1 - dot(V, normal), 5);
}

__device__ float D_GTR2_anisotropic(const Vec h, const float alphaX, const float alphaY, const Vec u, const Vec v, const Vec w) {
    float h_dot_u = dot(h, u), h_dot_v = dot(h, v), h_dot_w = dot(h,w);
    return 1 / (PI * alphaX * alphaY * pow((h_dot_u*h_dot_u)/(alphaX*alphaX) + (h_dot_v*h_dot_v)/(alphaY*alphaY) + h_dot_w*h_dot_w, 2));
}

__device__ float D_Berry(const Vec h, const Vec normal, float alpha) {
    alpha = clamp(alpha, 0.0001f, 0.9999f);
    float v = ((alpha*alpha - 1) / (2*PI * log(alpha))) / ((alpha*alpha-1)* absdot(h, normal) * absdot(h, normal) + 1);

    return ((alpha*alpha - 1) / (2*PI * log(alpha))) / ((alpha*alpha-1)* absdot(h, normal) * absdot(h, normal) + 1);
}

__device__ float G_GGX_lambda_anisotropic(const Vec w, const Vec h, const float alphaX, const float alphaY) {
    // レイと面の交差点における法線ベクトルを高さとする正規直行基底を生成(u,v,w)
    Vec wx, wy, wz;
    //Vec wx;
    //if (std::abs(wz.x) > sINF) wx = normalize(cross(Vec(0.0, 1.0, 0.0), wz));
    //else wx = normalize(cross(Vec(1.0, 0.0, 0.0), wz));
    //Vec wy = normalize(cross(wz, wx));

    GenerateONBFromNormal(h, wx, wy, wz);
    float wx_dot_w = dot(wx, w), wy_dot_w = dot(wy, w), wz_dot_w = dot(wz, w);
    return 0.5f * (-1 + sqrt(1 + (alphaX*alphaX*wx_dot_w*wx_dot_w + alphaY*alphaY*wy_dot_w*wy_dot_w) / (wz_dot_w*wz_dot_w)));
}

__device__ float G_GGX_anisotropic(const Vec incoming, const Vec outgoing, const float alphaX, const float alphaY) {
    Vec half_vector = normalize(outgoing + incoming);
    return 1 / (1 + G_GGX_lambda_anisotropic(incoming, half_vector, alphaX, alphaY) +
                G_GGX_lambda_anisotropic(outgoing, half_vector, alphaX, alphaY));
}

__device__ float smithG_GGX(const float NdotV, const float alphaG) {
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1 / (NdotV + sqrt(a + b - a*b));
}

__device__ Vec calculate_Disney_BRDF_diffuse(const Material &mat, const Vec &outgoing, const Vec &incoming,const Vec &normal) {
    float FL = Schlick_Fresnel(dot(normal, incoming)), FV = Schlick_Fresnel(dot(normal,outgoing));
    Vec half_vector = normalize(outgoing + incoming);
    float F_D90 = 0.5f + 2 * mat.roughness * absdot(half_vector, incoming) * absdot(half_vector, incoming);
    float FD = ((1-FL) + FL * F_D90) * ((1-FV) + FV*F_D90);
    return (mat.diffuse/PI) * FD;
}

__device__ Vec calculate_Disney_BRDF_subsurface(const Material &mat, const Vec &outgoing, const Vec &incoming,const Vec &normal) {
    Vec half_vector = normalize(outgoing + incoming);
    float FL = Schlick_Fresnel(dot(normal, incoming)), FV = Schlick_Fresnel(dot(normal,outgoing));
    float F_SS90 = mat.roughness * dot(half_vector, incoming) * dot(half_vector, incoming);
    float FSS = ((1-FL) + FL * F_SS90) * ((1-FV) + FV*F_SS90);
    return (mat.diffuse/PI) * 1.25f * (FSS * (1 / (dot(normal, incoming) + dot(normal, outgoing)) - 0.5f) + 0.5f);
}

__device__ Vec calculate_Disney_BRDF_sheen(const Material &mat, const Vec &outgoing, const Vec &incoming,const Vec &normal) {
    Vec Rho_sheen = (1 - mat.sheenTint)*Vec{1,1,1} + mat.sheenTint * normalize(mat.diffuse);
    Vec half_vector = normalize(outgoing + incoming);
    return mat.sheen * Rho_sheen * pow((1 - dot(half_vector, incoming)), 5);
}

__device__ Vec calculate_Disney_BRDF_specular(const Material &mat, const Vec &outgoing, const Vec &incoming,const Vec &normal,
                                              const Vec u, const Vec v, const Vec w, const float alpha_specular_x, const float alpha_specular_y) {
    Vec Rho_specular = (1 - mat.specularTint)*Vec{1,1,1} + mat.specularTint * normalize(mat.diffuse);
    Vec half_vector = normalize(outgoing + incoming);
    Vec F_S0 = (1.0f - mat.metallic) * 0.08f * mat.specular * Rho_specular + mat.metallic * mat.diffuse;
    return Fr_schlick(F_S0, normal, half_vector) *
           D_GTR2_anisotropic(half_vector, alpha_specular_x, alpha_specular_y, u, v, w) *
           G_GGX_anisotropic(incoming,outgoing,alpha_specular_x,alpha_specular_y) /
           (4 * absdot(incoming, normal) * absdot(outgoing, normal));
}

__device__ Vec calculate_Disney_BRDF_clearcoat(const Material &mat, const Vec &outgoing, const Vec &incoming,const Vec &normal, const float alpha_clearcoat) {
    Vec half_vector = normalize(outgoing + incoming, "calculate_Disney_BRDF_clearcoat");
    float N_dot_incoming = dot(normal, incoming);
    float N_dot_outgoing = dot(normal, outgoing);
    return 0.25f * mat.clearcoat * Fr_schlick(Vec{0.04,0.04,0.04}, normal, half_vector) *
           D_Berry(half_vector,normal, alpha_clearcoat) *
           // G_GGX_anisotropic(incoming, outgoing, 0.25, 0.25) /
           // (4 * absdot(incoming, normal) * absdot(outgoing, normal));          // これまでの実装
           smithG_GGX(N_dot_incoming, .25) * smithG_GGX(N_dot_outgoing, .25);
}
// incoming，outgoingは共に交差点「から」飛んでいく方向で考える．
__device__ Vec evaluate_Disney_BRDF(const float* weight_array, const Material &mat, const Vec &outgoing, const Vec incoming,
                                    const Vec &normal, const Vec &u, const Vec &v, const Vec &w,
                                    const float &alpha_specular_x, const float &alpha_specular_y, const float &alpha_clearcoat) {
    Vec BRDF = {0,0,0};

    if(dot(normal, incoming) < 0 || dot(normal, outgoing) < 0) {
        return BRDF;
    }

    BRDF = BRDF + weight_array[0] * calculate_Disney_BRDF_diffuse(mat, outgoing, incoming, normal);
    BRDF = BRDF + weight_array[1] * calculate_Disney_BRDF_subsurface(mat, outgoing, incoming, normal);
    BRDF = BRDF + weight_array[2] * calculate_Disney_BRDF_sheen(mat, outgoing, incoming, normal);
    BRDF = BRDF + weight_array[3] * calculate_Disney_BRDF_specular(mat, outgoing, incoming, normal, u, v, w, alpha_specular_x, alpha_specular_y);
    BRDF = BRDF + weight_array[4] * calculate_Disney_BRDF_clearcoat(mat, outgoing, incoming, normal, alpha_clearcoat);
    return BRDF;
}
// レイのサンプリング
__device__ void sample_ray_Disney_BRDF(Xorshift *rand, int &sampled_BRDF_ID, const Ray Ray, struct Ray &outgoing_ray, Vec &u, Vec &v, Vec &w, const HitPoint &hitpoint,
                            float &alpha_sheen, float &alpha_specular_x, float &alpha_specular_y, float &alpha_clearcoat) {
    // 選択したBRDFに基づいて方向サンプリングする
    // diffuse or subsurface or sheen -> cos重点サンプリング (ID <= 2)
    // specular -> GTR2_anisotropicのサンプリング (ID == 3)
    // clearcoat -> GTR1のサンプリング (ID == 4)
    if(sampled_BRDF_ID <= 2) {
        outgoing_ray.dir = sample_ray_cosine_weighted(rand, u, v, w);
        outgoing_ray.org = hitpoint.position;
    }
    else if(sampled_BRDF_ID == 3) {
        Vec half_vector = sample_half_vector_GTR2_anisotropic(rand, u, v, w, alpha_specular_x, alpha_specular_y);
        outgoing_ray.dir = normalize(Ray.dir - 2*dot(Ray.dir, half_vector)*half_vector);
        outgoing_ray.org = hitpoint.position;
    }
    else {
        Vec half_vector = sample_half_vector_GTR1(rand, u, v, w, alpha_clearcoat);
        outgoing_ray.dir = normalize(Ray.dir - 2*dot(Ray.dir, half_vector)*half_vector);
        outgoing_ray.org = hitpoint.position;
    }
}

// cos重点サンプリング
__device__ float evaluate_Disney_BRDF_diffuse_subsurface_sheen_pdf(Vec &normal, Vec &outgoing) {
    return max(dot(normal, outgoing), 0.0f) / PI;
}
__device__ float evaluate_Disney_BRDF_clearcoat_pdf(Vec &normal, Vec &outgoing, Vec &h, float &alpha) {
    return D_Berry(h, normal, alpha) * absdot(normal, h) / (4 * absdot(normal, outgoing));
}
__device__ float evaluate_Disney_BRDF_specular_pdf(Vec &normal, Vec &outgoing, Vec &h, float &alpha_specular_X, float &alpha_specular_Y, Vec u, Vec v, Vec w) {
    return D_GTR2_anisotropic(h, alpha_specular_X, alpha_specular_Y, u, v, w) * absdot(normal, h) / (4 * absdot(normal, outgoing));
}
// incoming，outgoingは共に交差点「から」飛んでいく方向で考える．
__device__ float evaluate_Disney_BRDF_pdf(const float* weight_array, Vec incoming, Vec outgoing, Vec normal, Vec u, Vec v, Vec w,
                                float alpha_specular_X, float alpha_specular_Y, float alpha_sheen, float alpha_clearcoat) {
    float sum_weight = 0;
    float pdf = 0;
    Vec half_vector = normalize(outgoing+incoming);

    for(int i = 0; i < 5; i++) {
        sum_weight += weight_array[i];
    }

    pdf += (weight_array[0] + weight_array[1] + weight_array[2]) * evaluate_Disney_BRDF_diffuse_subsurface_sheen_pdf(normal, outgoing);
    pdf += weight_array[3] * evaluate_Disney_BRDF_specular_pdf(normal, outgoing, half_vector, alpha_specular_X, alpha_specular_Y, u, v, w);
    pdf += weight_array[4] * evaluate_Disney_BRDF_clearcoat_pdf(normal, outgoing, half_vector, alpha_clearcoat);
    pdf /= sum_weight;
    return pdf;
}


#endif //CUDA_MYPT_DISNEY_BRDF_CUH
