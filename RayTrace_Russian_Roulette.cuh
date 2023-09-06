#ifndef CUDA_MYPT_RAYTRACE_RUSSIAN_ROULETTE_CUH
#define CUDA_MYPT_RAYTRACE_RUSSIAN_ROULETTE_CUH

__device__ bool Russian_Roulette(Vec &Radiance, Xorshift *rand, Material mat, Vec &ThroughPut, const int depth, const int AlwaysTrueMinDepth, const int DecreaseThresholdDepth) {
    float probability_russian_roulette = max(mat.diffuse.x, max(mat.diffuse.y, mat.diffuse.z));
    if(probability_russian_roulette == 0) probability_russian_roulette = sINF;

    // ある一定の深さとなったら打ち切りを開始
    // 打ち切られたら自身の発光のみを返す
    const float rnd = random_rangeMinMax(rand->seed, 0, 1);

    // ある一定以上の深さのレイについてはロシアンルーレット通過確率を大きく下げる
    if(depth > DecreaseThresholdDepth)  {
        probability_russian_roulette /= (float)(depth*depth);
    }

    // ある一定以上の深さのレイについてはロシアンルーレットで打ち切りを行う
    if(depth > AlwaysTrueMinDepth) {
        if(probability_russian_roulette < rnd) {
            Radiance = Radiance + multiply(ThroughPut, mat.emission);
            return false;
        }
    }
    // 打ち切らない場合
    else {
        probability_russian_roulette = 1.0f;
    }

    // depthが0の時に限りその面での発光を追加する
    if(depth == 0) {
        Radiance = Radiance + multiply(ThroughPut, mat.emission);
    }

    ThroughPut = ThroughPut / probability_russian_roulette;
    return true;
}

#endif //CUDA_MYPT_RAYTRACE_RUSSIAN_ROULETTE_CUH
