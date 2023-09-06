#ifndef CUDA_MYPT_STRUCT_XORSHIFT_CUH
#define CUDA_MYPT_STRUCT_XORSHIFT_CUH

struct Xorshift {
    unsigned long long int seed;
    __host__ __device__ Xorshift(unsigned long long int seed) : seed(seed) {}
};

// [0, ULLONG_MAX]の整数値乱数を生成する
__host__ __device__ unsigned long long int random_ullint(unsigned long long int seed) {
    if(seed==0) {
        seed++;
        printf("Warning in Xorshift::random_ullint: 0 cannot be a seed! Used 1 instead");
    }
    seed = seed ^ (seed << 5);
    seed = seed ^ (seed >> 41);
    seed = seed ^ (seed << 20);

    return seed;
}

// [rangeMin, rangeMax]の実数値乱数を生成する
__host__ __device__ float random_rangeMinMax(unsigned long long int &seed, const float rangeMin, const float rangeMax) {
    seed = random_ullint(seed);
   // printf("%llu\n", seed);
    float mid = (rangeMax+rangeMin) * 0.5f;
    float r = (rangeMax-rangeMin) * 0.5f;
    float ret = 2 * ((float)seed / (float)UINT64_MAX - 0.5f); // ret ∈ [-1, 1]
    ret = r * ret + mid; // ret ∈ [rangeMin, rangeMax]

    return ret;
}



#endif //CUDA_MYPT_STRUCT_XORSHIFT_CUH
