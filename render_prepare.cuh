#ifndef CUDA_MYPT_RENDER_PREPARE_CUH
#define CUDA_MYPT_RENDER_PREPARE_CUH

#include "struct_Camera.cuh"
#include "ConstructBVH.cuh"
#include "FileIO/IO_load_WavefrontObj.cuh"
#include "NextEventEstimation.cuh"

// カメラの設定 (TODO: 直書きではなくてファイルから読み込みが出来るようにする)
const Vec camera_position_begin = {15.8153 ,0.908405 ,0.704717 };
const Vec camera_position_mid = {15.8153 ,0.908405 ,0.704717 };
const Vec camera_position_end = {15.8153 ,0.908405 ,0.704717 };
const float camera_rotate_x_begin = 103.426;
const float camera_rotate_x_mid = 103.426;
const float camera_rotate_x_end = 103.426;
const float camera_rotate_y_begin = 0;
const float camera_rotate_y_mid = 0;
const float camera_rotate_y_end = 0;
const float camera_rotate_z_begin = 95.9501;

const float camera_rotate_z_mid = 95.9501;
const float camera_rotate_z_end = 95.9501;
const float camera2screen_dist = 28;

/*
 * render関数を呼び出す前に以下の準備を行う
 * 1: オブジェクトデータ，テクスチャ，HDRを読み込み
 * 2: カメラの姿勢を計算
 * 3: BVHの構築を行う
 * 4: IORの情報を求める
 * 5: NextEventEstimationの前処理を行う
*/
__host__ void render_prepare(Camera* CAM) {
    // 1: オブジェクトデータ等の読み込み
    load_object();
    load_HDR(HOST_HDRData->pixels, HOST_HDRData->width, HOST_HDRData->height, HOST_HDRData->bpp);
    if (HOST_HDRData->pixels == nullptr) HOST_Exists_NoHDR = true;
    printf("Successfully loaded Objects and HDR\n");

    // 2: カメラの設定
    *CAM = Camera_SetUpCamera(camera_position_begin, camera2screen_dist, camera_rotate_x_begin, camera_rotate_y_begin, camera_rotate_z_begin);
    printf("Camera set up\n");

    // 3: BVHの構築
    construct_BVH();
    printf("Successfully constructed BVH\n");

    // 5: NextEventEstimationの前処理
    calculate_Lss();
    calculate_Lss_HDR();
}

int FrameMid = frame_num * 0.6;

__host__ void render_update(Camera* CAM, int FrameID) {
    if(FrameID < FrameMid) {
        float t = (float)FrameID/(float)FrameMid;
        Vec CameraPos = DoubleInterpolate1d(camera_position_begin, camera_position_mid, t);
        float CameraRotateX = DoubleInterpolate1d(camera_rotate_x_begin, camera_rotate_x_mid, t);
        float CameraRotateY = DoubleInterpolate1d(camera_rotate_y_begin, camera_rotate_y_mid, t);
        float CameraRotateZ = DoubleInterpolate1d(camera_rotate_z_begin, camera_rotate_z_mid, t);
        *CAM = Camera_SetUpCamera(CameraPos, camera2screen_dist, CameraRotateX, CameraRotateY, CameraRotateZ);
    }
    else {
        float t = (float)(FrameID - FrameMid)/(float)frame_num;
        Vec CameraPos = LinearInterpolate1d(camera_position_mid, camera_position_end, t);
        float CameraRotateX = LinearInterpolate1d(camera_rotate_x_mid, camera_rotate_x_end, t);
        float CameraRotateY = LinearInterpolate1d(camera_rotate_y_mid, camera_rotate_y_end, t);
        float CameraRotateZ = LinearInterpolate1d(camera_rotate_z_mid, camera_rotate_z_end, t);
        *CAM = Camera_SetUpCamera(CameraPos, camera2screen_dist, CameraRotateX, CameraRotateY, CameraRotateZ);
    }


}

#endif //CUDA_MYPT_RENDER_PREPARE_CUH
