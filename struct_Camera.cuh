#ifndef CUDA_MYPT_STRUCT_CAMERA_CUH
#define CUDA_MYPT_STRUCT_CAMERA_CUH

#include "struct_Vec.cuh"
#include "matrix.cuh"
#include "global_values.cuh"
#include "struct_Ray.cuh"
#include "struct_Triangle.cuh"
#include "struct_Intersection.cuh"
#include "ConstructBVH.cuh"

struct Camera {
    Vec CamOrg;
    Vec CamDir;
    float CamToScreenDist;
    Vec ScreenXDir;
    Vec ScreenYDir;
    float PixelSize;

    __device__ __host__ Camera(const Vec &CamDir, const Vec &CamOrg, const float &CamToScreenDist,
                               const Vec &ScreenXDir, const Vec &ScreenYDir, const float PixelSize) :
                               CamDir(CamDir), CamOrg(CamOrg), CamToScreenDist(CamToScreenDist),
                               ScreenXDir(ScreenXDir), ScreenYDir(ScreenYDir), PixelSize(PixelSize) {}
};

// 指定されたカメラの情報からカメラの姿勢を求める
__host__ Camera Camera_SetUpCamera(Vec CamOrg, float CamToScreenDist, float rotateX, float rotateY, float rotateZ) {
    const Vec CamUp = rotate(DEG2RAD * rotateX, DEG2RAD * rotateY, DEG2RAD * rotateZ, normalize({0.0, 1.0, 0.0}));
    const Vec CamDir = rotate(DEG2RAD * rotateX, DEG2RAD * rotateY, DEG2RAD * rotateZ, normalize({0.0, 0.0, -1.0}));
    const Vec ScreenXDir = cross(CamDir, CamUp);
    const Vec ScreenYDir = CamUp;
    const float PixelSize = 36.0f/(float)image_sizeY;

    return {CamDir, CamOrg, CamToScreenDist, ScreenXDir, ScreenYDir, PixelSize};
}

// 画像上のピクセルインデックス{IndexX, IndexY}に対してのカメラからの1次レイを求める
__device__ inline Ray Generate1stRay(Camera* CAM, const int IndexX, const int IndexY) {
    Vec RayDir = CAM->CamDir * CAM->CamToScreenDist +
                 CAM->ScreenXDir * ((float)IndexX - image_sizeX_device/2.0f) * CAM->PixelSize +
                 CAM->ScreenYDir * (image_sizeY_device/2.0f - (float)IndexY) * CAM->PixelSize;
    RayDir = normalize(RayDir, "FirstRayGen");

    return {RayDir, CAM->CamOrg, 0};
}

// ある交差点がカメラから見たときにどのピクセル座標にあるのかを計算
// カメラから見えない場合は{-1, -1}を返す
__device__ void Camera_ConvertFromHitPointPosToPixelPos(int FaceID, Vec HitPointPos, Camera *CAM, const Triangle* Objects, const BVH_node_Device* BVH_tree,
                                                        int2 &PosLastFrameI, float2 &PosLastFrameF) {
    // カメラからその交差点へのレイを求める
    Vec OrgToHitPointDir = normalize(HitPointPos - CAM->CamOrg, "OrgToHitPointDir");
    Ray OrgToScreenRay(OrgToHitPointDir, CAM->CamOrg, 0);
    // 交差判定
    Intersection ClosestIntersection;
    if(willIntersectWithTriangle_nonRecursive(OrgToScreenRay, &ClosestIntersection, ObjectsNum, Objects, BVH_tree)) {
        // その交差点が今回の題意の面である場合はピクセル座標を求める
        if(ClosestIntersection.face_id == FaceID) {
            /*
             * カメラのスクリーンを三角面2枚と見る
             * (1)交差判定が通った場合は交差点の座標からピクセル座標を求める
             * (2)通らなかった場合はカメラからは見えない（画面外にある）ので{-1, -1}を返す
             */
            // スクリーンの4隅の座標．DL: 左下 | DR: 右下 | UL: 左上 | UR: 右上
            Vec ScreenULPos = CAM->CamOrg + CAM->CamDir * CAM->CamToScreenDist +
                              CAM->ScreenXDir * ((float)-image_sizeX_device/2.0f) * CAM->PixelSize +
                              CAM->ScreenYDir * ((float)image_sizeY_device/2.0f) * CAM->PixelSize;
            Vec ScreenURPos = ScreenULPos + CAM->ScreenXDir * CAM->PixelSize * (float)image_sizeX_device;
            Vec ScreenDLPos = ScreenULPos - CAM->ScreenYDir * CAM->PixelSize * (float)image_sizeY_device;
            Vec ScreenDRPos = ScreenDLPos + CAM->ScreenXDir * CAM->PixelSize * (float)image_sizeX_device;
            // スクリーンをなす2つの三角形 DL: 左下 | UR: 右上
            // ピクセル座標をnormalのx. y成分とする
            // 交差判定時に交差面での法線ベクトルを線形補完する
            // それゆえ返される.normalはその交差点でのピクセル座標となる（TODO: コメント書けばこんなコードが許される訳ではない）
            Triangle ScreenDL, ScreenUR;
            ScreenDL.v0 = ScreenULPos, ScreenDL.v0_vn = {0, 0, 0};
            ScreenDL.v1 = ScreenDLPos, ScreenDL.v1_vn = {0, (float)image_sizeY_device, 0};
            ScreenDL.v2 = ScreenDRPos, ScreenDL.v2_vn = {(float)image_sizeX_device, (float)image_sizeY_device, 0};
            ScreenUR.v0 = ScreenDRPos, ScreenUR.v0_vn = {(float)image_sizeX_device, (float)image_sizeY_device, 0};
            ScreenUR.v1 = ScreenURPos, ScreenUR.v1_vn = {(float)image_sizeX_device, 0, 0};
            ScreenUR.v2 = ScreenULPos, ScreenUR.v2_vn = {0, 0, 0};
            // (1)
            Vec Position;
            if(ScreenDL.intersectTest_SVGFBuffer(OrgToScreenRay, Position) ||
               ScreenUR.intersectTest_SVGFBuffer(OrgToScreenRay, Position)) {
                int x = (int)Position.x;
                int y = (int)Position.y;
                if(x < 0 || x >= image_sizeX_device ||
                   y < 0 || y >= image_sizeY_device) {
                    PosLastFrameI = {-1,-1};
                    PosLastFrameF = {-1,-1};
                }
                PosLastFrameI = {(int)Position.x, (int)Position.y};
                PosLastFrameF = {Position.x, Position.y};
            }
            // (2)
            else {
                PosLastFrameI = {-1,-1};
                PosLastFrameF = {-1,-1};
            }

        }
        // そうでない場合はカメラからは見えないので{-1, -1}を返す
        else {
            PosLastFrameI = {-1,-1};
            PosLastFrameF = {-1,-1};
        }
    }
    PosLastFrameI = {-1,-1};
    PosLastFrameF = {-1,-1};
}

#endif //CUDA_MYPT_STRUCT_CAMERA_CUH
