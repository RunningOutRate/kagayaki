#ifndef CUDA_MYPT_SVGF_CUH
#define CUDA_MYPT_SVGF_CUH

#include "test.cuh"

// Equation(3). SigmaZ := 1.0
// 分子: 2点の間のdepthの差の絶対値にマイナスをつける
// 分母: 2地点を結ぶベクトルと，注目点におけるzバッファの「ピクセル平面（離散） + depth成分（連続）」空間での勾配（隣り合うピクセルでのdepth比較）の内積の絶対値をSigmaZ倍
// これをexpに入れる
__device__ float CalculateDepthWeight(float Depth, float NeighborDepth, int dx, int dy, float dZdx, float dZdy) {
    float top = -1.0f * fabs(Depth - NeighborDepth);
    float bottom = 1.0f * fabs((float)dx * dZdx + (float)dy * dZdy) + sINF;
    if(bottom == 0) {
        printf("depth====\n");
    }
    return expf(top/bottom);
}

// Equation(4). SigmaN := 128
// 注目点の法線ベクトルと，ある点の法線ベクトルの内積をReLUを通し，SigmaN乗する．
__device__ float CalculateNormalWeight(Vec Normal, Vec NeighborNormal) {
    float cosine = max(0.0f, dot(Normal, NeighborNormal));
    cosine *= cosine; cosine *= cosine; cosine *= cosine; cosine *= cosine; // ^16
    cosine *= cosine; cosine *= cosine; cosine *= cosine; // ^128
    return cosine;
}

// Equation(5). SigmaL := 4.0
// 分子: 輝度の差の絶対値にマイナスをつける
// 分母: 3x3のガウシアンフィルターを掛けた分散の平方根，すなわち標準偏差
// これをexpに入れる
__device__ float CalculateLuminanceWeight(float Luminance, float NeighborLuminance, float LocalMeanSD) {
    float top = -1.0f * fabs(Luminance - NeighborLuminance);
    float bottom = 4.0f * LocalMeanSD + sINF;
   /* if(!isfinite(LocalMeanSD)) {
        printf("!Luminance LocalMeanSD\n");
    }
    if(!isfinite(Luminance)) {
        printf("!Luminance Luminance\n");
    }
    if(!isfinite(NeighborLuminance)) {
        printf("!Luminance NeighborLuminance\n");
    } */
    if(bottom == 0) {
        printf("luminance====\n");
    }
    return expf(top/bottom);
}

__device__ void TemporalAccumulation(Vec* RESULT_Buffer, SVGFBuffer* PrevSVGFBuffer, SVGFBuffer* CurrentSVGFBuffer, int pos) {
    // 放射がある場合は無視
    if(CurrentSVGFBuffer[pos].Emission.length() != 0) {
        return;
    }
    // ひとつ前のフレームにおける，現在posピクセルに見えている点のピクセル座標の配列インデックス
    int2 prevPos = CurrentSVGFBuffer[pos].PosLastFrameI;

    // 一つ手前のフレームにおけるピクセル座標の正確な位置から補完することを考える
    float s = CurrentSVGFBuffer[pos].PosLastFrameF.x - (float)CurrentSVGFBuffer[pos].PosLastFrameI.x;
    float t = CurrentSVGFBuffer[pos].PosLastFrameF.y - (float)CurrentSVGFBuffer[pos].PosLastFrameI.y;

    //printf("s, t, Fx, Ix Fy Iy: %f %f %f %d %f %d\n", s, t, CurrentSVGFBuffer[pos].PosLastFrameF.x, CurrentSVGFBuffer[pos].PosLastFrameI.x,
    //       CurrentSVGFBuffer[pos].PosLastFrameF.x, CurrentSVGFBuffer[pos].PosLastFrameI.x);

    // 左上，右上，左下，右下
    int2 offset[] = {{0, 0}, {1, 0},
                     {0, 1}, {1, 1}};
    float weight[] = {(1-s)*(1-t), s*(1-t), (1-s)*t, s*t};

    // 4点の輝度をバイリニア補完
    Vec SumRadiance = {0,0,0,0};
    float Sum_Weight = 0.0f;
    for(int i = 0; i < 4; i++) {
        int2 prevPos_XY = int2{prevPos.x + offset[i].x, prevPos.y + offset[i].y};
        int prevPos_idx = image_sizeX_device * prevPos_XY.y + prevPos_XY.x;

        if(prevPos_XY.x < 0 || prevPos_XY.x >= image_sizeX_device ||
           prevPos_XY.y < 0 || prevPos_XY.y >= image_sizeY_device) {
            continue;
        }

        // マテリアルが異なる場合は無視する
        // ひとつ前のフレームにおけるTemporalAccumulationフラグが立っている場合
        if(PrevSVGFBuffer[prevPos_idx].MaterialID == CurrentSVGFBuffer[pos].MaterialID && PrevSVGFBuffer[prevPos_idx].willApplyTemporalAccumulation) {
            SumRadiance = SumRadiance + weight[i] * PrevSVGFBuffer[prevPos_idx].ColorHistory;
            Sum_Weight += weight[i];
        }
    }

    if(Sum_Weight == 0.0f) {
        //printf("Temporal Accumulation\n");
        return;
    }
    Vec ColorHistory = SumRadiance/Sum_Weight;

    RESULT_Buffer[pos] = 0.2f*RESULT_Buffer[pos] + 0.8*ColorHistory;

}

/*
 * 注目ピクセルにおける輝度の分散を求める(Variance Estimation)
 * 1つ前のフレームにおける情報が利用できる場合は利用して推定する(-temporal)
 * そうでない場合は空間的に分散を推定する(Spatio-)
 *
 * あらかじめその点での輝度と輝度の二乗が並べられたMoment1or2Mapを用意しておく
 * -temporal: 過去のフレームデータと重みを付けて足し合わせる．結果をMomentMapとバッファの両方に記録
 * Spatio-: MomentMapに7x7バイラテラルフィルタをESF付きで与える
 */
__device__ float Moment1Map[1920*1920] = {0.0f};
__device__ float Moment2Map[1920*1920] = {0.0f};

__device__ void VarianceEstimation(Vec* RESULT_Buffer, Vec* RESULT_Filtered, SVGFBuffer* PrevSVGFBuffer, SVGFBuffer* CurrentSVGFBuffer, int pos) {
    // 注目しているピクセル[pos]における輝度の分散を計算する
    int posX = pos%image_sizeX_device;
    int posY = pos/image_sizeX_device;

    // ひとつ前のフレームにおける，現在posピクセルに見えている点のピクセル座標の配列インデックス
    int prevPos = image_sizeX_device * CurrentSVGFBuffer[pos].PosLastFrameI.y + CurrentSVGFBuffer[pos].PosLastFrameI.x;

    // レンダリング結果を1次レイの交差点におけるAlbedoで割った物について計算していく
    Vec DemodulatedColor = divide(RESULT_Buffer[pos], CurrentSVGFBuffer[pos].Albedo + Vec{sINF, sINF, sINF});
    float Luminance_DemodulatedColor = ConvertRGBToLuminance(DemodulatedColor);
    float sqLuminance_DemodulatedColor = Luminance_DemodulatedColor*Luminance_DemodulatedColor;

    // MomentMapをその地点での輝度と輝度の二乗で初期化
    Moment1Map[pos] = Luminance_DemodulatedColor;
    Moment2Map[pos] = sqLuminance_DemodulatedColor;

    // SpatialFilterで使用
    bool willApplySpatialFilter = CurrentSVGFBuffer[pos].PosLastFrameI.x == -1 || PrevSVGFBuffer[prevPos].sampleNum < 3;
    bool willApplyTemporalFilter = !willApplySpatialFilter;
    float Sum_Weight = 0.0f;
    float Sum_Moment1 = 0.0f;
    float Sum_Moment2 = 0.0f;
    // 注目点（pos）における法線ベクトル，depth，そして「ピクセル平面（離散） + depth成分（連続）」空間での勾配を求める
    float depth = CurrentSVGFBuffer[pos].Depth;
    int dx = posX < image_sizeX_device / 2 ? 1 : -1;
    int dy = posY < image_sizeY_device / 2 ? 1 : -1;
    int PosNeighbor_RL = image_sizeX_device*posY + posX + dx;
    int PosNeighbor_UD = image_sizeX_device*(posY+dy) + posX;
    float NeighborDepth_RL = CurrentSVGFBuffer[PosNeighbor_RL].Depth;
    float NeighborDepth_UD = CurrentSVGFBuffer[PosNeighbor_UD].Depth;
    float dZdx = (NeighborDepth_RL - depth) / (float)dx;
    float dZdy = (NeighborDepth_UD - depth) / (float)dy;
    Vec Normal = CurrentSVGFBuffer[pos].Normal;

    __syncthreads();

    // ひとつ前のフレームが妥当である場合，すなわちひとつ前のフレームにおいてカメラからその交差点が見え，かつサンプル数が3以上ある場合: 前のフレームにおける情報を利用
    if(willApplyTemporalFilter) {
        CurrentSVGFBuffer[pos].sampleNum = PrevSVGFBuffer[prevPos].sampleNum + sample;
        constexpr uint32_t threshold = 5;
        float CurrentWeight = 1.0f / (float)threshold; // Exponential Moving Average
        if (CurrentSVGFBuffer[pos].sampleNum < threshold) // Cumulative Moving Average
            CurrentWeight = 1.0f / (float)CurrentSVGFBuffer[pos].sampleNum;

        CurrentSVGFBuffer[pos].MomentOne = (1.0f - CurrentWeight) * PrevSVGFBuffer[prevPos].MomentOne +
                                           CurrentWeight * Moment1Map[pos];
        CurrentSVGFBuffer[pos].MomentTwo = (1.0f - CurrentWeight) * PrevSVGFBuffer[prevPos].MomentTwo +
                                           CurrentWeight * Moment2Map[pos];
        Moment1Map[pos] = CurrentSVGFBuffer[pos].MomentOne;
        Moment2Map[pos] = CurrentSVGFBuffer[pos].MomentTwo;
    }
    __syncthreads();
    // ひとつ前のフレームの情報が利用できない場合: 画像における空間的処理を行う
    if(willApplySpatialFilter) {
        // 7x7バイラテラルフィルタのカーネル
        constexpr float Kernel_BLF[7] = {0.00598f, 0.060626f, 0.241843f, 0.383103f, 0.241843f, 0.060626f, 0.00598f};

        /*
         * 7x7バイラテラルフィルタと同じ要領でカーネルを作用させて計算する
         * カーネルの各マス目に対応するピクセルでの，edge-stopping関数によって計算された重み付けを行う
         * ピクセルから読みだしたBufferの情報でその重みを求める
         */
        for(int kx = -3; kx <= 3; kx++) {
            // カーネルを作用させる点のX座標
            int NeighborPosX = posX + kx;
            // 範囲外参照
            if (NeighborPosX < 0 || NeighborPosX >= image_sizeX_device) {
                continue;
            }
            // カーネルの重み
            float hx = Kernel_BLF[kx + 3];
            for(int ky = -3; ky <= 3; ky++) {
                // カーネルを作用させる点のY座標
                int NeighborPosY = posY + ky;
                // 範囲外参照
                if (NeighborPosY < 0 || NeighborPosY >= image_sizeY_device) {
                    continue;
                }
                // カーネルの重み
                float hy = Kernel_BLF[ky + 3];
                // 自分自身をさしている時
                if(kx == 0 && ky == 0) {
                    Sum_Moment1 += hx*hy*Moment1Map[pos];
                    Sum_Moment2 += hx*hy*Moment2Map[pos];
                    Sum_Weight += hx*hy;
                    continue;
                }

                // カーネルが作用する点の配列インデックス
                int NeighborPos = image_sizeX_device * NeighborPosY + NeighborPosX;

                // カーネルが作用する点におけるバッファを読み込む
                // 作用させる点がそもそもどの面とも交差していない場合，影響を考えない
                if(CurrentSVGFBuffer[NeighborPos].ObjectID == -1) {
                    continue;
                }
                float NeighborDepth = CurrentSVGFBuffer[NeighborPos].Depth;
                Vec NeighborNormal = CurrentSVGFBuffer[NeighborPos].Normal;
                int   NeighborMaterialID = CurrentSVGFBuffer[NeighborPos].MaterialID;

                // edge-stopping functionにより重みを与える
                float wm = (CurrentSVGFBuffer[pos].MaterialID == NeighborMaterialID) ? 1.0f : 0.0f;
                float wz = CalculateDepthWeight(depth, NeighborDepth, kx, ky, dZdx, dZdy);
                float wn = CalculateNormalWeight(Normal, NeighborNormal);
                float weight = hx * hy * wz * wn * wm;

                // 作用
                Sum_Moment1 += weight * Moment1Map[NeighborPos];
                Sum_Moment2 += weight * Moment2Map[NeighborPos];
                Sum_Weight  += weight;
            }
        }
    }
    //値を更新していくので一旦足並みをそろえる
    __syncthreads();
    // ウェイトの総和で割る（同期のため外に出す）
    if(CurrentSVGFBuffer[pos].ObjectID == -1) {
        RESULT_Filtered[pos] = RESULT_Buffer[pos];
        CurrentSVGFBuffer[pos].init();
        CurrentSVGFBuffer[pos].willApplyATrousFilter = false;
    }
    else if(willApplySpatialFilter && Sum_Weight == 0) {
        RESULT_Filtered[pos] = RESULT_Buffer[pos];
        CurrentSVGFBuffer[pos].init();
        CurrentSVGFBuffer[pos].willApplyATrousFilter = false;
    }
    else if(willApplySpatialFilter) {
        CurrentSVGFBuffer[pos].MomentOne = Sum_Moment1 / Sum_Weight;
        CurrentSVGFBuffer[pos].MomentTwo = Sum_Moment2 / Sum_Weight;

        // サンプル数の更新
        // DisOcclusionが起きているのでサンプル棄却
        if(CurrentSVGFBuffer[pos].PosLastFrameI.x == -1) {
            CurrentSVGFBuffer[pos].sampleNum = sample;
        }
        else {
            CurrentSVGFBuffer[pos].sampleNum = PrevSVGFBuffer[prevPos].sampleNum + sample;
        }
    }
    __syncthreads();
/*
    if(!isfinite(CurrentSVGFBuffer[pos].MomentOne + CurrentSVGFBuffer[pos].MomentTwo)) {
        printf("Mom1: %f Mom2: %f, SumWeight: %f, SumMom1: %f, SumMom2: %f\n", CurrentSVGFBuffer[pos].MomentOne, CurrentSVGFBuffer[pos].MomentTwo,
               Sum_Weight, Sum_Moment1, Sum_Moment2);
    }
*/

    // そもそも交差していない場合は無視
    if(CurrentSVGFBuffer[pos].ObjectID == -1) {
        return;
    }

    // V[X] = E[X^2] - E[X]*E[X]
    CurrentSVGFBuffer[pos].Variance = fmax(0.0f, CurrentSVGFBuffer[pos].MomentTwo - CurrentSVGFBuffer[pos].MomentOne*CurrentSVGFBuffer[pos].MomentOne);

    if(CurrentSVGFBuffer[pos].Variance > 1000){
        //RESULT_Filtered[pos] = RESULT_Buffer[pos];
        //CurrentSVGFBuffer[pos].init();
        //CurrentSVGFBuffer[pos].willApplyATrousFilter = false;
        CurrentSVGFBuffer[pos].Variance = 1000;
    }

}
// 分散の安定化のために3x3のガウシアンフィルターを与える
__device__ void ApplyGaussianFilterToVar(Vec* RESULT_Buffer, Vec* RESULT_Filtered, SVGFBuffer* PrevSVGFBuffer, SVGFBuffer* CurrentSVGFBuffer, int pos, float &LocalMeanSD) {
    // 注目しているピクセル[pos]
    int posX = pos%image_sizeX_device;
    int posY = pos/image_sizeX_device;

    // ガウシアンフィルターに掛ける
    float Kernel_GF[3] = {1 / 4.0f, 1 / 2.0f, 1 / 4.0f};
    float Sum_Var = 0.0f;
    float Sum_Weight = 0.0f;

    for(int kx = -1; kx <= 1; kx++) {
        float hx = Kernel_GF[kx+1];
        int NeighborPosX = posX + kx;
        // 範囲外参照
        if(NeighborPosX < 0 || NeighborPosX >= image_sizeX_device) {
            continue;
        }
        for(int ky = -1; ky <= 1; ky++) {
            float hy = Kernel_GF[ky+1];
            int NeighborPosY = posY + ky;
            // 範囲外参照
            if(NeighborPosY < 0 || NeighborPosY >= image_sizeY_device) {
                continue;
            }
            // カーネルが作用する点の配列インデックス
            int NeighborPos = image_sizeX_device * NeighborPosY + NeighborPosX;

            if(CurrentSVGFBuffer[NeighborPos].ObjectID == -1){
                continue;
            }

            float Sum_Var_prev = Sum_Var;

            // カーネルの作用
            float weight = hx * hy;
            Sum_Var += weight * CurrentSVGFBuffer[NeighborPos].Variance;
            Sum_Weight += weight;

           /* if(!isfinite(Sum_Var) && isfinite(Sum_Var_prev)) {
                printf("weight: %f, var: %f\n", hx*hy, CurrentSVGFBuffer[NeighborPos].Variance);
            } */
        }
    }
    __syncthreads();
    // 本当に分散が0であればなにもしない
    if(Sum_Var == 0 && Sum_Weight != 0) {
        RESULT_Filtered[pos] = RESULT_Buffer[pos];
        CurrentSVGFBuffer[pos].willApplyATrousFilter = false;
    }
    // 分散0の原因が重み0の場合は
    else if(Sum_Weight == 0) {
        CurrentSVGFBuffer[pos].MomentOne = -1.0f;
        CurrentSVGFBuffer[pos].MomentTwo = -1.0f;
        CurrentSVGFBuffer[pos].sampleNum = 0;
        CurrentSVGFBuffer[pos].willApplyATrousFilter = false;
    }
    else {
        // フィルタ後の分散の記録
        CurrentSVGFBuffer[pos].Variance = Sum_Var/Sum_Weight;
        // 分散の標準偏差
        LocalMeanSD = sqrtf(Sum_Var/Sum_Weight);

     /*   if(!isfinite(LocalMeanSD)) {
            printf("LocalMeanSD SumVar SumWeight: %f %f %f %f %f %f\n", LocalMeanSD, Sum_Var, Sum_Weight, CurrentSVGFBuffer[pos].MomentOne,
                   CurrentSVGFBuffer[pos].MomentTwo, CurrentSVGFBuffer[pos].Variance);
        } */
    }
}

// A-Trousフィルターを適用する
__device__ void ApplyATrousFilterOneStage(Vec* RESULT_Buffer, Vec* RESULT_Filtered, SVGFBuffer* PrevSVGFBuffer, SVGFBuffer* CurrentSVGFBuffer, int pos, float LocalMeanSD, int stage) {
    // 注目しているピクセル[pos]
    int posX = pos%image_sizeX_device;
    int posY = pos/image_sizeX_device;
    Vec DemodulatedColor = divide(RESULT_Buffer[pos], CurrentSVGFBuffer[pos].Albedo + Vec{sINF,sINF,sINF});
    float Luminance_DemodulatedColor = ConvertRGBToLuminance(DemodulatedColor);

    constexpr float KernelWeight[] = {
            1 / 256.0f,  4 / 256.0f,  6 / 256.0f,  4 / 256.0f, 1 / 256.0f,
            4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
            6 / 256.0f, 24 / 256.0f, 36 / 256.0f, 24 / 256.0f, 6 / 256.0f,
            4 / 256.0f, 16 / 256.0f, 24 / 256.0f, 16 / 256.0f, 4 / 256.0f,
            1 / 256.0f,  4 / 256.0f,  6 / 256.0f,  4 / 256.0f, 1 / 256.0f
    };
    constexpr int2 KernelOffset[] = {
            {-2, -2}, {-1, -2}, {0, -2}, {1, -2}, {2, -2},
            {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {2, -1},
            {-2,  0}, {-1,  0}, {0,  0}, {1,  0}, {2,  0},
            {-2,  1}, {-1,  1}, {0,  1}, {1,  1}, {2,  1},
            {-2,  2}, {-1,  2}, {0,  2}, {1,  2}, {2,  2}
    };

    const int KernelSize = 25;
    const int KernelCenterIndex = 12;

    // 注目点（pos）における法線ベクトル，depth，そして「ピクセル平面（離散） + depth成分（連続）」空間での勾配を求める
    float depth = CurrentSVGFBuffer[pos].Depth;
    int dx = posX < image_sizeX_device / 2 ? 1 : -1;
    int dy = posY < image_sizeY_device / 2 ? 1 : -1;
    int PosNeighbor_RL = image_sizeX_device*posY + posX + dx;
    int PosNeighbor_UD = image_sizeX_device*(posY+dy) + posX;
    float NeighborDepth_RL = CurrentSVGFBuffer[PosNeighbor_RL].Depth;
    float NeighborDepth_UD = CurrentSVGFBuffer[PosNeighbor_UD].Depth;
    float dZdx = (NeighborDepth_RL - depth) / (float)dx;
    float dZdy = (NeighborDepth_UD - depth) / (float)dy;
    Vec Normal = CurrentSVGFBuffer[pos].Normal;

    Vec Sum_Radiance = {0.0f,0.0f,0.0f};
    float Sum_Var = 0.0f;
    float Sum_Weight = 0.0f;
    for(int KernelIndex = 0; KernelIndex < KernelSize; KernelIndex++) {
        // カーネルの重み
        float h = KernelWeight[KernelIndex];
        // カーネルが作用する点の座標
        int2 KernelPos = {posX + (KernelOffset[KernelIndex].x << stage), posY + (KernelOffset[KernelIndex].y << stage)};
        // 範囲外参照
        if(KernelPos.x < 0 || KernelPos.x >= image_sizeX_device ||
           KernelPos.y < 0 || KernelPos.y >= image_sizeY_device) {
            continue;
        }
        // カーネルが作用する点の配列インデックス
        int NeighborPos = image_sizeX_device * KernelPos.y + KernelPos.x;
        // 自分自身をさしている時
        if(KernelIndex == KernelCenterIndex) {
            Sum_Var     += h*h * CurrentSVGFBuffer[pos].Variance;
            Sum_Radiance = Sum_Radiance + h * DemodulatedColor;
            Sum_Weight += h;
            continue;
        }
        // 作用させる点についてバッファから情報を読み込む
        // 作用させる点がそもそもどの面とも交差していない場合，影響を考えない
        if(CurrentSVGFBuffer[NeighborPos].ObjectID == -1) {
            continue;
        }
        float NeighborDepth      = CurrentSVGFBuffer[NeighborPos].Depth;
        Vec   NeighborNormal     = CurrentSVGFBuffer[NeighborPos].Normal;
        Vec   NeighborRadiance   = divide(RESULT_Buffer[NeighborPos], CurrentSVGFBuffer[NeighborPos].Albedo + Vec{sINF, sINF, sINF});
        float NeighborLuminance  = ConvertRGBToLuminance(NeighborRadiance);
        float NeighborVariance   = CurrentSVGFBuffer[NeighborPos].Variance;
        int   NeighborMaterialID = CurrentSVGFBuffer[NeighborPos].MaterialID;

        if((Luminance_DemodulatedColor > 0.5 && NeighborLuminance > 5 * Luminance_DemodulatedColor) ||
           (Luminance_DemodulatedColor > 0.1 && NeighborLuminance > 50 * Luminance_DemodulatedColor) ||
           (Luminance_DemodulatedColor > sINF && NeighborLuminance > 100 * Luminance_DemodulatedColor)) {
            continue;
        }

        // edge-stopping functionにより重みを与える
        float wm = CurrentSVGFBuffer[pos].MaterialID == NeighborMaterialID ? 1 : 0;
        float wz = CalculateDepthWeight(depth, NeighborDepth, KernelPos.x, KernelPos.y, dZdx, dZdy);
        float wn = CalculateNormalWeight(Normal, NeighborNormal);
        float wl = CalculateLuminanceWeight(Luminance_DemodulatedColor, NeighborLuminance, LocalMeanSD);
        float weight;
        if(CurrentSVGFBuffer[pos].willApplyATrousFilter) {
            weight = h * wz * wn * wl * wm;
        }
        else {
            weight = h * wz * wn * wm;
        }

        //if(!isfinite(weight)) {
        //    printf("h wz wn wl: %f %f %f %f\n", h, wz, wn, wl);
        //}

        // カーネルへの作用
        Sum_Radiance = Sum_Radiance + weight * NeighborRadiance;
        Sum_Var     += weight * weight * NeighborVariance;
        Sum_Weight  += weight;
    }
    __syncthreads();
    if(CurrentSVGFBuffer[pos].ObjectID == -1) {
        return;
    }
    if(Sum_Weight < 1e-6f || Sum_Weight*Sum_Weight < 1e-6f) {
        RESULT_Filtered[pos] = RESULT_Buffer[pos];
        CurrentSVGFBuffer[pos].willApplyTemporalAccumulation = true;
        return;
    }

    Vec filteredLighting = Sum_Radiance / Sum_Weight;
    float filteredVar = Sum_Var / (Sum_Weight * Sum_Weight);
   // constexpr uint32_t threshold = 3;
    float CurrentWeight = 0.2f;

    RESULT_Filtered[pos] = RESULT_Buffer[pos];
    RESULT_Filtered[pos] = (CurrentWeight) * multiply(filteredLighting, CurrentSVGFBuffer[pos].Albedo) +
                           (1 - CurrentWeight) * RESULT_Filtered[pos];
    RESULT_Buffer[pos] = RESULT_Filtered[pos];
    if(CurrentSVGFBuffer[pos].willApplyATrousFilter) {
        CurrentSVGFBuffer[pos].Variance = (CurrentWeight) * CurrentSVGFBuffer[pos].Variance +
                                          (1 - CurrentWeight) * filteredVar;
    }
    // Temporal Accumulationのための記録
    if(stage == 0) {
        CurrentSVGFBuffer[pos].ColorHistory = RESULT_Filtered[pos];
        CurrentSVGFBuffer[pos].willApplyTemporalAccumulation = true;
    }
}

__device__ void SVGF(Vec* RESULT_Buffer, Vec* RESULT_Filtered, SVGFBuffer* PrevSVGFBuffer, SVGFBuffer* CurrentSVGFBuffer, int pos) {
    float LocalMeanSD = 0.0f;
    TemporalAccumulation(RESULT_Buffer, PrevSVGFBuffer, CurrentSVGFBuffer, pos);
    __syncthreads();
    VarianceEstimation(RESULT_Buffer, RESULT_Filtered, PrevSVGFBuffer, CurrentSVGFBuffer, pos);
    __syncthreads();
    ApplyGaussianFilterToVar(RESULT_Buffer, RESULT_Filtered, PrevSVGFBuffer, CurrentSVGFBuffer, pos, LocalMeanSD);
    __syncthreads();
    int filterStageMax = 5;
    // Edge-avoiding a-trous wavelet transform
    for(int k = 0; k < 5; k++) {
        for(int iteration = 0; iteration < filterStageMax; iteration++) {
            ApplyATrousFilterOneStage(RESULT_Buffer, RESULT_Filtered, PrevSVGFBuffer, CurrentSVGFBuffer, pos, LocalMeanSD, iteration);
            __syncthreads();
        }
    }
    RESULT_Buffer[pos] = CurrentSVGFBuffer[pos].ColorHistory;
}

__global__ void ApplySVGF(Vec* RESULT_Buffer, Vec* RESULT_Filtered, SVGFBuffer* PrevSVGFBuffer, SVGFBuffer* CurrentSVGFBuffer, const int LoopID, const int OneLoopSize, int FrameID){
    int pos = blockIdx.x*blockDim.x + threadIdx.x + OneLoopSize * LoopID;

    if(!isfinite(RESULT_Buffer[pos].length())) {
        RESULT_Buffer[pos] = {0.0f, 0.0f, 0.0f};
    }

    if(FrameID == 0) {
        PrevSVGFBuffer[pos].init();
    }

    if(PrevSVGFBuffer[pos].MomentOne == -1.0f) {
        Vec Color = divide(RESULT_Buffer[pos], CurrentSVGFBuffer[pos].Albedo);
        CurrentSVGFBuffer[pos].MomentOne = ConvertRGBToLuminance(Color);
        CurrentSVGFBuffer[pos].MomentTwo = CurrentSVGFBuffer[pos].MomentOne * CurrentSVGFBuffer[pos].MomentOne;
    }
    RESULT_Filtered[pos] = {0,0,0};
    __syncthreads();
    SVGF(RESULT_Buffer, RESULT_Filtered, PrevSVGFBuffer, CurrentSVGFBuffer, pos);
    //ApplyNamelessFilter(RESULT_Buffer, RESULT_Filtered, PrevSVGFBuffer, CurrentSVGFBuffer, pos);
    //int prevPos = image_sizeX_device * CurrentSVGFBuffer[pos].PosLastFrameI.y + CurrentSVGFBuffer[pos].PosLastFrameI.x;
    //if(CurrentSVGFBuffer[pos].PosLastFrameI.x != -1){
    //    RESULT_Filtered[pos] = {1,1,0};
   // }
    //else {}
    //RESULT_Filtered[pos] = Vec{CurrentSVGFBuffer[pos].MomentTwo, CurrentSVGFBuffer[pos].MomentTwo, CurrentSVGFBuffer[pos].MomentTwo};
    __syncthreads();
}



#endif //CUDA_MYPT_SVGF_CUH
