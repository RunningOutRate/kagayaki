#ifndef CUDA_MYPT_ALGORITHMS_CUH
#define CUDA_MYPT_ALGORITHMS_CUH

template<typename T>
// 照準にソートされた配列Arrayにおいて，K以上で最も先頭にある要素へのインデックスを返す
__device__ int lower_bound(const T* Array, const int Length, const T K) {
    int ng = -1;
    int ok = Length-1;
    int mid;
    while(ok - ng > 1) {
        mid = (ok + ng)/2;
        if(Array[mid] >= K) {
            ok = mid;
        }
        else {
            ng = mid;
        }
    }
    return ok;
}

template<typename T>
__host__ __device__ T clamp(T value, T MIN, T MAX) {
    if(value < MIN) {
        return MIN;
    }
    else if(value > MAX) {
        return MAX;
    }
    else {
        return value;
    }
}

// 線形補完 SegA -> SegB
template<typename T>
__host__ __device__ T LinearInterpolate1d(const T &SegA, const T &SegB, const float t) {
    return (1-t)*SegA + t*SegB;
}

// 2次補完 SegA -> SegB
template<typename T>
__host__ __device__ T DoubleInterpolate1d(const T &SegA, const T &SegB, const float t) {
    return (1-t*t)*SegA + t*t*SegB;
}

#endif //CUDA_MYPT_ALGORITHMS_CUH
