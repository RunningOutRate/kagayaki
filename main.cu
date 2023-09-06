#include <iostream>

#include "render_prepare.cuh"
#include "render.cuh"
#include "struct_Camera.cuh"
#include "struct_Triangle.cuh"
#include "struct_BVH_node.cuh"
#include "struct_Material.cuh"
#include "struct_SVGFBuffers.cuh"
#include "host_to_device.cuh"
#include "FileIO/IO_save_png.cuh"
#include "SVGF.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUErrorCheck: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main() {
    cudaEvent_t start, stop;

// 初期化

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

// 開始時間を記録

    cudaEventRecord(start);
    auto* HOST_CAM = (Camera*)calloc(1, sizeof(Camera));
    auto* HOST_RESULT = (Vec*)calloc(image_sizeX*image_sizeY, sizeof(Vec));

    auto* DEVICE_CAM = (Camera*)calloc(1, sizeof(Camera));
    auto* DEVICE_CAM_prev = (Camera*)calloc(1, sizeof(Camera));
    auto* DEVICE_RESULT = (Vec*)calloc(image_sizeX*image_sizeY, sizeof(Vec));
    auto* DEVICE_RESULT_SVGF = (Vec*)calloc(image_sizeX*image_sizeY, sizeof(Vec));

    auto* HOST_SVGFBuffer = (SVGFBuffer*)calloc(image_sizeX*image_sizeY, sizeof(SVGFBuffer));
    for(int i = 0;i < image_sizeX*image_sizeY; i++) {
        HOST_SVGFBuffer[i].init();
    }

    auto* DEVICE_SVGFBuffer = (SVGFBuffer*)calloc(image_sizeX*image_sizeY, sizeof(SVGFBuffer));
    //cudaMalloc(&DEVICE_SVGFBuffer, sizeof(SVGFBuffer) * image_sizeX*image_sizeY);

    auto* DEVICE_PrevSVGFBuffer = (SVGFBuffer*)calloc(image_sizeX*image_sizeY, sizeof(SVGFBuffer));
    //cudaMalloc(&DEVICE_PrevSVGFBuffer, sizeof(SVGFBuffer) * image_sizeX*image_sizeY);


    HOST_HDRData = (Texture*)calloc(1, sizeof(Texture));

    render_prepare(HOST_CAM);

    // std::vectorはデバイス側では使えないのでポインタを使用した配列に変換する必要がある

    HOST_ObjectsData_Array = (Triangle*)calloc(HOST_ObjectsData.size(), sizeof(Triangle));
    HOST_MaterialsData_Array = (Material*)calloc(HOST_MaterialsData.size(), sizeof(Material));
    HOST_TexturesData_Array = (Texture*)calloc(HOST_TexturesData.size(), sizeof(Texture));
    HOST_BVH_tree_ConvertedForDevice = (BVH_node_Device*)calloc(HOST_BVH_tree.size(), sizeof(BVH_node_Device));
    HOST_Light_strength_sum_Array = (float*)calloc(HOST_Light_strength_sum.size(), sizeof(float));
    HOST_Light_strength_sum_HDR_Array = (float*)calloc(HOST_Light_strength_sum_HDR.size(), sizeof(float));

    // std::vectorからポインタ配列へと変換する

    ConvertVectorToPtrArray(HOST_ObjectsData, HOST_ObjectsData_Array);
    ConvertVectorToPtrArray(HOST_MaterialsData, HOST_MaterialsData_Array);
    ConvertVectorToPtrArray(HOST_Light_strength_sum, HOST_Light_strength_sum_Array);
    ConvertVectorToPtrArray(HOST_Light_strength_sum_HDR, HOST_Light_strength_sum_HDR_Array);
    // 構造体に含まれるstd::vectorをポインタ配列に変換し，全体もポインタ配列に変換する

    Convert_TextureVectorFromHostToDevice(HOST_TexturesData, HOST_TexturesData_Array);
    Convert_BVHTreeFromHostToDevice(HOST_BVH_tree, HOST_BVH_tree_ConvertedForDevice);

    // デバイス上にメモリを確保してホスト側のデータを送る

    MallocAndSetOnDevice(HOST_CAM, DEVICE_CAM, 1, true);
    MallocAndSetOnDevice(HOST_CAM, DEVICE_CAM_prev, 1, true);
    MallocAndSetOnDevice(HOST_RESULT, DEVICE_RESULT, image_sizeX*image_sizeY, true);
    MallocAndSetOnDevice(HOST_RESULT, DEVICE_RESULT_SVGF, image_sizeX*image_sizeY, true);
    MallocAndSetOnDevice(HOST_ObjectsData_Array, DEVICE_ObjectsData, (int)HOST_ObjectsData.size(), true);
    MallocAndSetOnDevice(HOST_MaterialsData_Array, DEVICE_MaterialsData, (int)HOST_MaterialsData.size(), true);
    MallocAndSetOnDevice(HOST_Light_strength_sum_Array, DEVICE_Light_strength_sum, (int)HOST_Light_strength_sum.size(), true);
    MallocAndSetOnDevice(HOST_Light_strength_sum_HDR_Array, DEVICE_Light_strength_sum_HDR, (int)HOST_Light_strength_sum_HDR.size(), true);
    MallocAndSetOnDevice(HOST_SVGFBuffer, DEVICE_SVGFBuffer, image_sizeX*image_sizeY, true);
    MallocAndSetOnDevice(HOST_SVGFBuffer, DEVICE_PrevSVGFBuffer, image_sizeX*image_sizeY, true);


    // 特殊な構造体のデータ送信
   // DEVICE_HDRData = (Texture*)calloc(1, sizeof(Texture));
   // DEVICE_TexturesData = (Texture*)calloc(HOST_TexturesData.size(), sizeof(Texture));
   // DEVICE_BVH_tree = (BVH_node_Device*)calloc(HOST_BVH_tree.size(), sizeof(BVH_node_Device));
    // デバイス上にメモリを確保してホスト側のデータを送る(特殊な構造体)

    MallocAndSetTextureOnDevice(HOST_HDRData, DEVICE_HDRData, 1);
    MallocAndSetTextureOnDevice(HOST_TexturesData_Array, DEVICE_TexturesData, (int)HOST_TexturesData.size());
    MallocAndSetBVHtreeOnDevice(HOST_BVH_tree_ConvertedForDevice, DEVICE_BVH_tree, (int)HOST_BVH_tree.size());

    void *ObjectNumPtr; int *ObjectNumPtrCPU = (int*)calloc(1, sizeof(int));
    *ObjectNumPtrCPU = (int)HOST_ObjectsData.size();
    cudaGetSymbolAddress(&ObjectNumPtr, ObjectsNum);
    cudaMemcpy(ObjectNumPtr, ObjectNumPtrCPU, sizeof(int), cudaMemcpyHostToDevice);

    void *LssSizePtr; int* LssSizePtrCPU = (int*)calloc(1, sizeof(int));
    *LssSizePtrCPU = (int)HOST_Light_strength_sum.size();
    cudaGetSymbolAddress(&LssSizePtr, Light_strength_sum_size);
    cudaMemcpy(LssSizePtr, LssSizePtrCPU, sizeof(int), cudaMemcpyHostToDevice);

    void *LssHDRSizePtr; int* LssHDRSizePtrCPU = (int*)calloc(1, sizeof(int));
    *LssHDRSizePtrCPU = (int)HOST_Light_strength_sum_HDR.size();
    cudaGetSymbolAddress(&LssHDRSizePtr, Light_strength_sum_HDR_size);
    cudaMemcpy(LssHDRSizePtr, LssHDRSizePtrCPU, sizeof(int), cudaMemcpyHostToDevice);

    void *NoLightPtr;
    cudaGetSymbolAddress(&NoLightPtr, Exists_NoLight);
    cudaMemcpy(NoLightPtr, &HOST_Exists_NoLight, sizeof(bool), cudaMemcpyHostToDevice);

    void *NoHDRPtr;
    cudaGetSymbolAddress(&NoHDRPtr, Exists_NoHDR);
    cudaMemcpy(NoHDRPtr, &HOST_Exists_NoHDR, sizeof(bool), cudaMemcpyHostToDevice);

    /*for(int i = HOST_ObjectsData.size(); i < HOST_BVH_tree.size(); i++) {
        std::cout << "ID: " << i << " | " << HOST_BVH_tree[i].face_list.size() << " | children:  " << HOST_BVH_tree[i].children.first << " " << HOST_BVH_tree[i].children.second;
        std::cout << std::endl;
    }
    std::cout << HOST_BVH_tree.size() - HOST_ObjectsData.size() << std::endl;
     */

    int blocksize = 256;
    int OneLoopSize = blocksize * 64;
    int LoopNum = image_sizeX*image_sizeY / OneLoopSize;
    dim3 block (blocksize, 1, 1);
    dim3 grid  (OneLoopSize / block.x, 1, 1);
    cudaDeviceSetLimit(cudaLimitStackSize, 1024*32);
    printf("Rendering's Started\n");

    OneLoopSize = image_sizeX*image_sizeY;
    LoopNum = 1;
    block.x = blocksize;
    grid.x = OneLoopSize/block.x;

    for(int i = 0; i < frame_num; i++) {
        for(int j = 0; j < LoopNum; j++) {
            render<<<grid, block>>>(DEVICE_CAM_prev, DEVICE_CAM, DEVICE_RESULT, DEVICE_ObjectsData, DEVICE_MaterialsData, DEVICE_TexturesData, DEVICE_BVH_tree,
                                    DEVICE_Light_strength_sum, DEVICE_Light_strength_sum_HDR, DEVICE_HDRData, DEVICE_SVGFBuffer, j, OneLoopSize);
            gpuErrchk( cudaGetLastError() )
            gpuErrchk( cudaDeviceSynchronize() )
        }
        cudaMemcpy(HOST_RESULT, DEVICE_RESULT, sizeof(Vec) * image_sizeX*image_sizeY,
                   cudaMemcpyDeviceToHost);
        save_png(HOST_RESULT, i+1, "Output");

        OneLoopSize = image_sizeX*image_sizeY;
        LoopNum = 1;
        block.x = 256;
        grid.x = OneLoopSize/block.x;
        for(int j = 0; j < LoopNum; j++) {
            ApplySVGF<<<grid, block>>>(DEVICE_RESULT, DEVICE_RESULT_SVGF, DEVICE_PrevSVGFBuffer, DEVICE_SVGFBuffer, j, OneLoopSize, i);
            gpuErrchk( cudaGetLastError() )
            gpuErrchk( cudaDeviceSynchronize() )
        }

        cudaMemcpy(HOST_RESULT, DEVICE_RESULT_SVGF, sizeof(Vec) * image_sizeX*image_sizeY,
                   cudaMemcpyDeviceToHost);

        save_png(HOST_RESULT, i+1, "SVGF");

        //cudaMemcpy(HOST_RESULT, DEVICE_RESULT, sizeof(Vec) * image_sizeX*image_sizeY,
        //           cudaMemcpyDeviceToHost);

        //save_png(HOST_RESULT, i+1, "ColorHistory");

        std::cout << "Frame " << i+1 << " Done." << std::endl;
        // カメラの更新
        render_update(HOST_CAM, i+1);
        cudaMemcpy(DEVICE_CAM_prev, DEVICE_CAM, sizeof(Camera), cudaMemcpyDeviceToDevice);
        cudaMemcpy(DEVICE_PrevSVGFBuffer, DEVICE_SVGFBuffer, sizeof(SVGFBuffer) * image_sizeX*image_sizeY, cudaMemcpyDeviceToDevice);
        cudaMemcpy(DEVICE_CAM, HOST_CAM, sizeof(Camera), cudaMemcpyHostToDevice);

        // 終了時間を記録
        cudaEventRecord(stop);
        //イベントの終了を待つ。
        cudaEventSynchronize(stop);
        // ms単位でstartとstopの差を計算する。
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Current time: %f ms\n", milliseconds);
        if(milliseconds > 295000) {
            printf("TLE. Aborted\n");
            break;
        }
    }
    printf("Rendering's Done\n");
}
