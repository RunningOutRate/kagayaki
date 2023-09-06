#ifndef CUDA_MYPT_STRUCT_TEXTURE_CUH
#define CUDA_MYPT_STRUCT_TEXTURE_CUH

#include "struct_Vec.cuh"

struct Texture{
    float* pixels;
    int width, height, bpp;

    __host__ __device__ Texture() : pixels(), width(), height(), bpp() {}

    __device__ Vec get_Color(float &u, float &v) const{
        int idx = (width*(int)((float)height*(1-v)) + (int)((float)width*u));
        idx *= bpp;
        Vec Color = {pixels[idx], pixels[idx+1],pixels[idx+2]};
        //printf("Color.length: %f\n", Color.length());
        return Color;
    }
};

// Textureのホスト側のデータ構造をデバイス側でも使用できるように組み替える
__host__ void Convert_TextureFromHostToDevice(const Texture &Host, Texture &Device) {
    Device.width = Host.width;
    Device.height = Host.height;
    Device.bpp = Host.bpp;
    Device.pixels = (float*)calloc(Host.width*Host.height*Host.bpp, sizeof(float));
    for(int i = 0; i < Host.width*Host.height*Host.bpp; i++) {
        Device.pixels[i] = Host.pixels[i];
    }
}

// Texturesのホスト側のデータをデバイス側でも使用できるように組み替える
__host__ void Convert_TextureVectorFromHostToDevice(const std::vector<Texture>& Host, Texture*& Device) {
    for(int i = 0; i < Host.size(); i++) {
        Convert_TextureFromHostToDevice(Host[i], Device[i]);
    }
}

// Texturesをデバイス側にコピーする
__host__ void MallocAndSetTextureOnDevice(Texture*& HostData, Texture*& DeviceData, const int NumOfElement) {

    Texture* h_data = (Texture*)malloc(NumOfElement * sizeof(Texture));
    memcpy(h_data, HostData,NumOfElement * sizeof(Texture));

    for(int i = 0; i < NumOfElement; i++) {
        cudaMalloc(&(h_data[i].pixels), HostData[i].width * HostData[i].height *  HostData[i].bpp * sizeof(float));
        cudaMemcpy(h_data[i].pixels, HostData[i].pixels, HostData[i].width * HostData[i].height *  HostData[i].bpp * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&DeviceData, sizeof(Texture) * NumOfElement);
    cudaMemcpy(DeviceData, h_data, sizeof(Texture) * NumOfElement, cudaMemcpyHostToDevice);
}



#endif //CUDA_MYPT_STRUCT_TEXTURE_CUH
