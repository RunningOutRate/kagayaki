#ifndef CUDA_MYPT_IMAGEBASEDLIGHTING_CUH
#define CUDA_MYPT_IMAGEBASEDLIGHTING_CUH



__device__ Vec calculate_uv_Sphere(Vec Dir) {
    float theta, phi;

    phi = asin(Dir.z);

    if(abs(Dir.z) >= 1.0 - sINF) theta = 0.0f;
    else if(abs(Dir.x) <= sINF) {
        if(Dir.y > 0) theta = PI/2;
        else theta = -PI/2;
    }
    else {
        if(Dir.x > 0) {
            theta = asin(Dir.y / sqrt(1 - Dir.z * Dir.z));
        }
        else if (Dir.x < 0 && Dir.y > 0) {
            theta = PI - asin(Dir.y / sqrt(1 - Dir.z * Dir.z));
        }
        else {
            theta = -1*PI - asin(Dir.y / sqrt(1 - Dir.z * Dir.z));
        }
    }
    theta = clamp(theta/(2*PI) + 0.5f, 0.0f, 1.0f);
    phi   = clamp(phi / PI + 0.5f, 0.0f, 1.0f);

    return {theta, phi, 0.0f};
}

__device__ Vec get_HDR_emission(Vec Dir, const Texture* HDRData) {
    Vec UV_Sphere = calculate_uv_Sphere(Dir);
    int idx = (HDRData->width)*(int)((float)(HDRData->height-1)*(1-UV_Sphere.y)) + (int)((float)(HDRData->width-1)*(1-UV_Sphere.x));
    idx *= HDRData->bpp;
 /*   if(!((!(idx < 0) && !(idx >= HDRData->width*HDRData->height*HDRData->bpp)))) {
        printf("ERROR: \nDir = {%f, %f, %f}\nUV_Sphere = {%f, %f}\nidx = %d\nLimit: 0 <= idx < %d\n",
               Dir.x, Dir.y, Dir.z, UV_Sphere.x, UV_Sphere.y, idx, HDRData->width*HDRData->height*HDRData->bpp);
        idx = 0;
    } */
    Vec emission = Vec( HDRData->pixels[idx], HDRData->pixels[idx+1],HDRData->pixels[idx+2]);
    return emission;
}

__device__ Vec convert_HDRindex_to_Dir(int index, float &pdf_NEE_HDR, const Texture* HDRData) {
    float u, v;
    u = 1.0f - (float)(index % HDRData->width) / (float)HDRData->width;
    v = 1.0f - (float)(index / HDRData->width) / (float)HDRData->height;
    float theta = (u-0.5f)*2*PI;
    float phi = (v-0.5f)*PI;
    if(abs(cos(phi)) < sINF) pdf_NEE_HDR *= 1.0f / (2.0f*PI*PI*sINF);
    else pdf_NEE_HDR *= 1.0f / (2.0f*PI*PI*cos(phi));
    pdf_NEE_HDR *= (float)(HDRData->width*HDRData->height);
    return {(float)cos(phi)*cos(theta), (float)cos(phi)*sin(theta), (float)sin(phi)};
}

// 累積和を考える．HOST_Light_strength_sum_HDR[i] := pixels[i]までの画素値のlengthの和
__host__ void calculate_Lss_HDR() {
    const int n = HOST_HDRData->width*HOST_HDRData->height;
    if(n != 0) {
        float v = 1.0;
        HOST_Light_strength_sum_HDR.push_back(Vec(HOST_HDRData->pixels[0], HOST_HDRData->pixels[1], HOST_HDRData->pixels[2]).length() * cos((v - 0.5f) * PI_HOST));
        for (int i = 1; i < n; i++) {
            v = 1.0f - (float)(i / HOST_HDRData->width) / (float)HOST_HDRData->height;
            HOST_Light_strength_sum_HDR.push_back(HOST_Light_strength_sum_HDR[i - 1] +
                                             Vec(HOST_HDRData->pixels[HOST_HDRData->bpp*i],
                                                 HOST_HDRData->pixels[HOST_HDRData->bpp*i+1],
                                                 HOST_HDRData->pixels[HOST_HDRData->bpp*i+2]).length() * cos((v-0.5f)*PI_HOST));
            //std::cout << "LSS[" << i << "] = " << HOST_Light_strength_sum_HDR[i] << std::endl;
        }
    }
}

// HDRの画素を選ぶ  アクセスする際にbppを掛ける必要があることに注意．
__device__ int choose_HDR_pixel(float R, const float* LightStrengthSum_HDR) {
    // RをLssの値域に対応させる
    R *= LightStrengthSum_HDR[Light_strength_sum_HDR_size-1];
    // LssにおいてRより大きな最小の値を持つindexを求める
    return lower_bound(LightStrengthSum_HDR, Light_strength_sum_HDR_size, R);
}
// 選んだHDRの画素を選ぶ確率を求める
__device__ float calculate_HDR_pdf(const Vec dir, const float* LightStrengthSum_HDR, const Texture* HDRData) {
    Vec UV_Sphere = calculate_uv_Sphere(dir);
    // UV_Sphere.x = u
    // UV_Sphere.y = v
    int idx = (HDRData->width*(int)((float)HDRData->height*(1.0f-UV_Sphere.y)) + (int)((float)HDRData->width*(UV_Sphere.x)));

    if(idx == 0) return LightStrengthSum_HDR[0]/LightStrengthSum_HDR[Light_strength_sum_HDR_size-1];
    else return (LightStrengthSum_HDR[idx] - LightStrengthSum_HDR[idx-1]) / LightStrengthSum_HDR[Light_strength_sum_HDR_size-1];
}

#endif //CUDA_MYPT_IMAGEBASEDLIGHTING_CUH
