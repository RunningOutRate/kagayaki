#ifndef CUDA_MYPT_STRUCT_BVH_NODE_CUH
#define CUDA_MYPT_STRUCT_BVH_NODE_CUH

#include "struct_Vec.cuh"
#include "host_to_device.cuh"

struct BVH_node_Host {
    Vec position_min;
    Vec position_max;
    Vec position_center;
    std::pair<int, int> children;
    std::vector<int> face_list;

    __host__ BVH_node_Host(const Vec position_min = {-1e37,-1e37, -1e37}, const Vec position_max = {1e38, 1e38, 1e38}, const Vec position_center = {0,0,0})
            : position_min(position_min), position_max(position_max), position_center(position_center), children() , face_list(){}

    // AABBの表面積を求める
    __host__ float surface_area() {
        float x = position_max.x - position_min.x;
        float y = position_max.y - position_min.y;
        float z = position_max.z - position_min.z;
        return 2.0f*(x*y + y*z + z*x);
    }
};

struct BVH_node_Device {
    Vec position_min;
    Vec position_max;
    Vec position_center;
    int2 children;
    int* face_list;
    int face_list_size;

    __host__ __device__ BVH_node_Device(const Vec position_min = {-1e37,-1e37, -1e37}, const Vec position_max = {1e38, 1e38, 1e38}, const Vec position_center = {0,0,0}, const int face_list_size = 0)
            : position_min(position_min), position_max(position_max), position_center(position_center), children() , face_list(), face_list_size(face_list_size) {}

    // AABBの表面積を求める
    __device__ float surface_area() {
        float x = position_max.x - position_min.x;
        float y = position_max.y - position_min.y;
        float z = position_max.z - position_min.z;
        return 2.0f*(x*y + y*z + z*x);
    }
};

// BVHNodeのホスト側のデータ構造をデバイス側でも使用できるように組み替える
// std::vectorを配列にし，不要な情報を削る
__host__ void Convert_BVHNodeFromHostToDevice(const BVH_node_Host &Host, BVH_node_Device &Device) {
    Device.position_min = Host.position_min;
    Device.position_max = Host.position_max;
    Device.position_center = Host.position_center;
    Device.children.x = Host.children.first;
    Device.children.y = Host.children.second;
    if(Host.children == std::pair{-1, -1}) {
        Device.face_list = (int*)calloc(Host.face_list.size(), sizeof(int));
        Device.face_list_size = (int)Host.face_list.size();
        ConvertVectorToPtrArray(Host.face_list, Device.face_list);
    }
    else {
        Device.face_list_size = 0;
    }
}

// BVHTreeのホスト側のデータ構造をデバイス側でも使用できるように組み替える
__host__ void Convert_BVHTreeFromHostToDevice(const std::vector<BVH_node_Host>& Host, BVH_node_Device*& Device) {
    for(int i = 0; i < Host.size(); i++) {
        Convert_BVHNodeFromHostToDevice(Host[i], Device[i]);
    }
}

// BVHTreeをデバイス側にコピーする
__host__ void MallocAndSetBVHtreeOnDevice(BVH_node_Device*& HostData, BVH_node_Device*& DeviceData, const int NumOfElement) {
    BVH_node_Device* h_data = (BVH_node_Device*)malloc(NumOfElement * sizeof(BVH_node_Device));
    memcpy(h_data, HostData,NumOfElement * sizeof(BVH_node_Device));

    for(int i = 0; i < NumOfElement; i++) {
        cudaMalloc(&(h_data[i].face_list), HostData[i].face_list_size*sizeof(int));
        cudaMemcpy(h_data[i].face_list, HostData[i].face_list, HostData[i].face_list_size*sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&DeviceData, sizeof(BVH_node_Device) * NumOfElement);
    cudaMemcpy(DeviceData, h_data, sizeof(BVH_node_Device) * NumOfElement, cudaMemcpyHostToDevice);
}


#endif //CUDA_MYPT_STRUCT_BVH_NODE_CUH
