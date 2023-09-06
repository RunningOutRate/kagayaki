#ifndef CUDA_MYPT_STRUCT_VEC_CUH
#define CUDA_MYPT_STRUCT_VEC_CUH

struct Vec{
    float x,y,z;
    float align;
    __host__ __device__ Vec(const float x = 0.0f, const float y = 0.0f, const float z = 0.0f, const float align = 0.0f) : x(x), y(y), z(z), align(align) {}

    __host__ __device__ Vec operator+(const Vec &b) const{
        return {x+b.x, y+b.y, z+b.z};
    }
    __host__ __device__ Vec operator-(const Vec &b) const{
        return {x-b.x, y-b.y, z-b.z};
    }
    __host__ __device__ Vec operator*(const float &b) const{
        return {x*b, y*b, z*b};
    }
    __host__ __device__ Vec operator/(const float &b) const{
        return {x/b, y/b, z/b};
    }
    __host__ __device__ float length_squared() const{
        return x*x + y*y + z*z;
    }
    __host__ __device__ float length() const{
        return sqrtf(length_squared());
    }

};

__host__ __device__ Vec operator*(float t, const Vec v) {
    return v*t;
}
__host__ __device__ bool operator==(const Vec a, const Vec b) {
    return (a.x==b.x && a.y==b.y && a.z==b.z);
}
__host__ __device__ bool operator!=(const Vec a, const Vec b) {
    return (a.x!=b.x || a.y!=b.y || a.z!=b.z);
}

__host__ __device__ inline Vec normalize(const Vec &b, const char* text = "") {
    if(b.length()==0) {
        printf("error: you're normalizing 0-size vector. %s\n", text);
        return {10000, 0, 0};
    }
    return Vec (b/b.length());
}
__host__ __device__ inline const Vec multiply(const Vec &v1, const Vec &v2) {
    return Vec(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
__host__ __device__ inline const Vec divide(const Vec &v1, const Vec &v2) {
    return Vec(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
__host__ __device__ inline  float dot(const Vec &v1, const Vec &v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
__host__ __device__ inline  float absdot(const Vec &v1, const Vec &v2) {
    return std::abs(dot(v1, v2));
}
__host__ __device__ inline  Vec cross(const Vec &v1, const Vec &v2) {
    return Vec (
            (v1.y * v2.z) - (v1.z * v2.y),
            (v1.z * v2.x) - (v1.x * v2.z),
            (v1.x * v2.y) - (v1.y * v2.x));
}
__host__ __device__ inline const Vec exponential(const Vec &v) {
    return Vec(exp(v.x), exp(v.y), exp(v.z));
}
__host__ inline void cout(const Vec &v, std::string coutType = "no_comma_with_br") {
    if(coutType == "no_comma_with_br")std::cout << v.x << " " << v.y << " " << v.z << std::endl;
    else if(coutType == "with_comma_with_br")std::cout << v.x << ", " << v.y << ", " << v.z << std::endl;
    else if(coutType == "no_comma_no_br")std::cout << v.x << " " << v.y << " " << v.z;
    else if(coutType == "with_comma_no_br")std::cout << v.x << ", " << v.y << ", " << v.z;
}

// TextBefore (v.x v.y v.z) TextAfter\n
__device__ inline void printVec(const Vec &v, const char* TextBefore = "", const char* TextAfter = "") {
    printf("%s %f %f %f %s\n", TextBefore, v.x, v.y, v.z, TextAfter);
}

// 法線ベクトルを高さとする正規直行基底を生成(u,v,w)
__device__ void GenerateONBFromNormal(const Vec &N, Vec &u, Vec &v, Vec &w) {
    w = N;
    if (abs(w.x) > 1e-4) u = normalize(cross(Vec(0.0f, 1.0f, 0.0f), w), "GenONB");
    else u = normalize(cross(Vec(1.0f, 0.0f, 0.0f), w), "GenONB");
    v = normalize(cross(w, u), "GenONB");
}

// RGBを輝度に変換
__device__ float ConvertRGBToLuminance(const Vec &RGB) {
    return 0.2126f * RGB.x + 0.7152f * RGB.y + 0.0722f * RGB.z;
}

#endif //CUDA_MYPT_STRUCT_VEC_CUH
