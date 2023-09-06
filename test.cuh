#ifndef CUDA_MYPT_TEST_CUH
#define CUDA_MYPT_TEST_CUH

__device__ void VarianceEstimationTest(Vec* RESULT_Buffer, Vec* RESULT_Filtered, SVGFBuffer* PrevSVGFBuffer, SVGFBuffer* CurrentSVGFBuffer, int pos) {
    // そもそも交差していない場合は無視
    if(CurrentSVGFBuffer[pos].ObjectID == -1) {
        RESULT_Filtered[pos] = RESULT_Buffer[pos];
        CurrentSVGFBuffer[pos].willApplyATrousFilter = false;
        return;
    }

    // 注目しているピクセル[pos]における輝度の分散を計算する
    int posX = pos%image_sizeX_device;
    int posY = pos/image_sizeX_device;

    // ひとつ前のフレームにおける，現在posピクセルに見えている点のピクセル座標の配列インデックス
    int prevPos = image_sizeX_device * CurrentSVGFBuffer[pos].PosLastFrameI.y + CurrentSVGFBuffer[pos].PosLastFrameI.x;

    // レンダリング結果を1次レイの交差点におけるAlbedoで割った物について計算していく
    Vec DemodulatedColor = divide(RESULT_Buffer[pos], CurrentSVGFBuffer[pos].Albedo);
    float Luminance_DemodulatedColor = ConvertRGBToLuminance(DemodulatedColor);
    float sqLuminance_DemodulatedColor = Luminance_DemodulatedColor*Luminance_DemodulatedColor;

    // ひとつ前のフレームが妥当である場合，すなわちひとつ前のフレームにおいてカメラからその交差点が見え，かつサンプル数が3以上ある場合: 前のフレームにおける情報を利用
    if(CurrentSVGFBuffer[pos].PosLastFrameI.x != -1 && PrevSVGFBuffer[prevPos].sampleNum >= 3) {
        CurrentSVGFBuffer[pos].sampleNum = PrevSVGFBuffer[prevPos].sampleNum + sample;
        constexpr uint32_t threshold = 5;
        float CurrentWeight = 1.0f / threshold; // Exponential Moving Average
        if (CurrentSVGFBuffer[pos].sampleNum < threshold) // Cumulative Moving Average
            CurrentWeight = 1.0f / (float)CurrentSVGFBuffer[pos].sampleNum;

        CurrentSVGFBuffer[pos].MomentOne = (1.0f - CurrentWeight) * PrevSVGFBuffer[prevPos].MomentOne +
                                           CurrentWeight * Luminance_DemodulatedColor;
        CurrentSVGFBuffer[pos].MomentTwo = (1.0f - CurrentWeight) * PrevSVGFBuffer[prevPos].MomentTwo +
                                           CurrentWeight * sqLuminance_DemodulatedColor;
    }
        // ひとつ前のフレームの情報が利用できない場合: 画像における空間的処理を行う
    else {
        CurrentSVGFBuffer[pos].Variance = 0.0f;
    }
    // V[X] = E[X^2] - E[X]*E[X]
    CurrentSVGFBuffer[pos].Variance = CurrentSVGFBuffer[pos].MomentTwo - CurrentSVGFBuffer[pos].MomentOne*CurrentSVGFBuffer[pos].MomentOne;
}


#endif //CUDA_MYPT_TEST_CUH
