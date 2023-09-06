#ifndef CUDA_MYPT_IO_SAVE_PNG_CUH
#define CUDA_MYPT_IO_SAVE_PNG_CUH

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external_lib/stb/stb_image_write.h"

__host__ void ClampImageBuffer(Vec* Buffer_RenderedImage) {
    for(int i = 0; i < image_sizeX*image_sizeY; i++) {
        /*if(!isfinite(Buffer_RenderedImage[i].x) || !isfinite(Buffer_RenderedImage[i].y) || !isfinite(Buffer_RenderedImage[i].z)) {
            if(isnan(Buffer_RenderedImage[i].x) || isnan(Buffer_RenderedImage[i].y) || isnan(Buffer_RenderedImage[i].z)) {
                Buffer_RenderedImage[i] = {1,0,0};
            }
            else {
                Buffer_RenderedImage[i] = {0.5,0,1};
            }
        }
        if(isnan(Buffer_RenderedImage[i].x) || isnan(Buffer_RenderedImage[i].y) || isnan(Buffer_RenderedImage[i].z)) {
            Buffer_RenderedImage[i] = {1,0,0};
        } */
        Buffer_RenderedImage[i].x = clamp(Buffer_RenderedImage[i].x, 0.0f, 1.0f);
        Buffer_RenderedImage[i].y = clamp(Buffer_RenderedImage[i].y, 0.0f, 1.0f);
        Buffer_RenderedImage[i].z = clamp(Buffer_RenderedImage[i].z, 0.0f, 1.0f);
    }
}

__host__ void GammaConvertImageBuffer(Vec* Buffer_RenderedImage) {
    for(int i = 0; i < image_sizeX*image_sizeY; i++) {
        Buffer_RenderedImage[i] = {powf(Buffer_RenderedImage[i].x, 1/2.2f),
                                   powf(Buffer_RenderedImage[i].y, 1/2.2f),
                                   powf(Buffer_RenderedImage[i].z, 1/2.2f)};
    }
}

__host__ void save_png(Vec* Buffer_RenderedImage, int FrameID, const char* FileName) {
    std::vector<unsigned char> Image(image_sizeX * image_sizeY * 4);
    GammaConvertImageBuffer(Buffer_RenderedImage);
    ClampImageBuffer(Buffer_RenderedImage);
    for(int i = 0; i < image_sizeX*image_sizeY; i++) {
        Image.at(4*i+0) = (unsigned char)(Buffer_RenderedImage[i].x * 255);
        Image.at(4*i+1) = (unsigned char)(Buffer_RenderedImage[i].y * 255);
        Image.at(4*i+2) = (unsigned char)(Buffer_RenderedImage[i].z * 255);
        Image.at(4*i+3) = 255;
        //cout(Buffer_RenderedImage[i] * 255);
    }
    char *Name = (char*)malloc(sizeof(char) * 128);
    if((FrameID-1)/10 < 1) {
        sprintf(Name, "%s00%d.png",FileName, FrameID-1);
    }
    else if((FrameID-1)/10 < 10) {
        sprintf(Name, "%s0%d.png",FileName, FrameID-1);
    }
    else {
        sprintf(Name, "%s%d.png",FileName, FrameID-1);
    }
    stbi_write_png(Name, image_sizeX, image_sizeY, 4, &Image.front(), 0);
}

#endif //CUDA_MYPT_IO_SAVE_PNG_CUH
