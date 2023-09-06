#ifndef CUDA_MYPT_STRUCT_STACK_CUH
#define CUDA_MYPT_STRUCT_STACK_CUH

// 参考: https://hanecci.hatenadiary.org/entry/20110109/1294581354
template<class T, int N>
class stack {
private:
    T buf[N];
    int idx;

public:
    __device__ stack() :
            idx(-1)
    {}

    __device__ T & top() {
        return buf[idx];
    }

    __device__ void push(T const & v) {
        buf[++idx] = v;
    }

    __device__ T pop() {
        return buf[idx--];
    }

    __device__ bool full() {
        return idx == (N - 1);
    }

    __device__ bool empty() {
        return idx == -1;
    }
};

#endif //CUDA_MYPT_STRUCT_STACK_CUH
