#ifndef CUDA_MYPT_DISNEY_BRDF_SAMPLING_CUH
#define CUDA_MYPT_DISNEY_BRDF_SAMPLING_CUH

__device__ Vec sample_half_vector_GTR2_anisotropic(Xorshift *rand, Vec u, Vec v, Vec w, float alphaX, float alphaY){
    float R1 = random_rangeMinMax(rand->seed, 0.0001f, 0.9999f);
    float R2 = random_rangeMinMax(rand->seed, 0.0001f, 0.9999f);
    Vec h = sqrt(R2/(1-R2)) * (alphaX * cos(2*PI*R1)*u + alphaY * sin(2*PI*R1)*v) + w;
    return normalize(h);
}

__device__ float sample_GTR1_cos_theta_h(Xorshift *rand, Vec u, Vec v, Vec w, float alpha) {
    float R2 = random_rangeMinMax(rand->seed, 0.0001f, 0.9999f);
    alpha = clamp(alpha, -0.9999f, 0.9999f);

    return sqrt((1 - pow(alpha, 2*(1 - R2))) / (1 - alpha*alpha));
}

__device__ float sample_GTR1_phi_h(Xorshift *rand) {
    float R1 = random_rangeMinMax(rand->seed, 0.0001f, 0.9999f);
    return 2*PI*R1;
}

__device__ Vec sample_half_vector_GTR1(Xorshift *rand, Vec u, Vec v, Vec w, float alpha){
    float cosphi = cos(sample_GTR1_phi_h(rand));
    float sinphi = sqrt(1 - cosphi*cosphi);
    float costheta = sample_GTR1_cos_theta_h(rand, u, v, w, alpha);
    float sintheta = sqrt(1 - costheta*costheta);

    return sintheta*cosphi* u + sintheta*sinphi*v + costheta*w;
}

#endif //CUDA_MYPT_DISNEY_BRDF_SAMPLING_CUH
