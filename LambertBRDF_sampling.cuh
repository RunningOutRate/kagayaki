#ifndef CUDA_MYPT_LAMBERTBRDF_SAMPLING_CUH
#define CUDA_MYPT_LAMBERTBRDF_SAMPLING_CUH

// pdf = cosθ/π
// 完全拡散面など
__device__ float calculate_Lambert_pdf(const Vec &normal, const Vec &dir) {
    return max(dot(normal, dir), 0.0f) / PI;
}

__device__ Vec sample_ray_cosine_weighted (Xorshift *rand, Vec u, Vec v, Vec w) {
    // 確率密度関数の値はcos(θ)/π | 累積分布関数はθとφについて重積分し，(φ/2π) * (1-cos^2(θ))
    // θ: 法線とレイのなす角 [0, π/2) / φ: 法線を法線方向から見てみると点となる(これは交差点)が，今回その周辺方向には一様にレイが反射するとする．[0, 2π]
    // ゆえに乱数R1, R2を用いると，φ = 2πR1 | θ = acos(sqrt(R2))
    // この時 cos(θ) = sqrt(1-R2), sin(θ) = sqrt(R2)
    float R1 = random_rangeMinMax(rand->seed, 0, 1.0f);
    float R2 = random_rangeMinMax(rand->seed, 0, 0.9999f);

    float theta = acosf(sqrtf(1.0f - R2));
    float phi = 2 * PI * R1;

    // 方向サンプリング
    Vec outgoing_ray_dir = normalize(
              u * sinf(theta) * cosf(phi)
            + v * sinf(theta) * sinf(phi)
            + w * cosf(theta), "sampling cosine weighted");

    //printf("%f %f %f\n", outgoing_ray_dir.x, outgoing_ray_dir.y, outgoing_ray_dir.z);
    return outgoing_ray_dir;
}

#endif //CUDA_MYPT_LAMBERTBRDF_SAMPLING_CUH
