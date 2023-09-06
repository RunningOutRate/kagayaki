#ifndef CUDA_MYPT_IO_LOAD_IMAGES_CUH
#define CUDA_MYPT_IO_LOAD_IMAGES_CUH

#define STB_IMAGE_IMPLEMENTATION
#include "external_lib/stb/stb_image.h"
#include "global_values.cuh"

//画像のロード
__host__ void load_image(const std::string& filename, float* &pixels, int &width, int &height, int &bpp) {
    pixels = stbi_loadf(filename.c_str(), &width, &height, &bpp, 0);
    for(int i = 0; i < width*height*bpp; i++) {
        pixels[i] += 1e-4;
    }
}

// HDRのロード
__host__ void load_HDR(float* &pixels, int &width, int &height, int &bpp) {
    pixels = stbi_loadf("HDR.hdr", &width, &height, &bpp, 0);
}

// TexNameToTexIDからHOSTTexturesDataを作成する
__host__ void load_Textures(std::map<std::string, int> &TexNameToTexID) {
    HOST_TexturesData.resize(TexNameToTexID.size());
    for (const auto& NameToID : TexNameToTexID) {
        std::cout << NameToID.second << std::endl;
        load_image(NameToID.first, HOST_TexturesData[NameToID.second].pixels, HOST_TexturesData[NameToID.second].width,
                   HOST_TexturesData[NameToID.second].height, HOST_TexturesData[NameToID.second].bpp);
    }
}

#endif //CUDA_MYPT_IO_LOAD_IMAGES_CUH
