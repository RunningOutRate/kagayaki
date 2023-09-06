#ifndef CUDA_MYPT_STRUCT_SVGFBUFFERS_CUH
#define CUDA_MYPT_STRUCT_SVGFBUFFERS_CUH

#include "struct_Vec.cuh"

// TODO: SoAな実装にする
struct SVGFBuffer{
    Vec Albedo;
    Vec Emission;
    Vec Normal;
    Vec ColorHistory;
    int ObjectID = -1;
    int MaterialID;
    int2 PosLastFrameI = {-1, -1};
    float2 PosLastFrameF = {-1,-1};
    int sampleNum = 0;
    float MomentOne = 0.0f;
    float MomentTwo = 0.0f;
    float Variance;
    float Depth;
    bool willApplyATrousFilter = true;
    bool willApplyTemporalAccumulation = false;

    __host__ __device__ SVGFBuffer(Vec &Albedo, Vec &Emission, Vec &Normal, Vec &ColorHistory, int &ObjectID, int &MaterialID,
                                   int2 &PosLastFrameI, float2 &PosLastFrameF, int &sampleNum, float &MomentOne, float &MomentTwo, float &Variance, float& Depth,
                                   bool &willApplyATrousFilter, bool &willApplyTemporalAccumulation)
                                   : Albedo(Albedo), Emission(Emission), Normal(Normal), ColorHistory(ColorHistory), ObjectID(ObjectID), MaterialID(MaterialID), PosLastFrameI(PosLastFrameI),
                                     PosLastFrameF(PosLastFrameF), sampleNum(sampleNum), MomentOne(MomentOne), MomentTwo(MomentTwo), Variance(Variance), Depth(Depth),
                                   willApplyATrousFilter(willApplyATrousFilter), willApplyTemporalAccumulation(willApplyTemporalAccumulation) {}

    __host__ __device__ void init() {
        Albedo = {0,0,0};
        Emission = {0,0,0};
        Normal = {0,0,0};
        ColorHistory = {0,0,0};
        ObjectID = -1;
        MaterialID = -1;
        PosLastFrameI = {-1,-1};
        PosLastFrameF = {-1,-1};
        sampleNum = 0;
        MomentOne = -1.0f;
        MomentTwo = -1.0f;
        Variance = 0.0f;
        Depth = 0.0f;
        willApplyATrousFilter = true;
        willApplyTemporalAccumulation = false;
    }
};

#endif //CUDA_MYPT_STRUCT_SVGFBUFFERS_CUH
