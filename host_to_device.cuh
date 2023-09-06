#ifndef CUDA_MYPT_HOST_TO_DEVICE_CUH
#define CUDA_MYPT_HOST_TO_DEVICE_CUH


template<typename T>
__host__ void MallocAndSetOnDevice(T*& HostData, T*& DeviceData, const int NumOfElement, bool willCopyFromHostToDevice) {
    cudaMalloc(&DeviceData, sizeof(T) * NumOfElement);

    if(willCopyFromHostToDevice) {
        cudaMemcpy(DeviceData, HostData, sizeof(T) * NumOfElement, cudaMemcpyHostToDevice);
    }
    else {
        cudaMemset(&DeviceData, 0, sizeof(T) * NumOfElement);
    }
}

template<typename T>
__host__ void ConvertVectorToPtrArray(const std::vector<T> &Vector, T*& Array) {
    for(int i = 0; i < Vector.size(); i++) {
        Array[i] = Vector[i];
    }
}

#endif //CUDA_MYPT_HOST_TO_DEVICE_CUH
