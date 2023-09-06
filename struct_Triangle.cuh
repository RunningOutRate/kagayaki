#ifndef CUDA_MYPT_STRUCT_TRIANGLE_CUH
#define CUDA_MYPT_STRUCT_TRIANGLE_CUH

#include "struct_Vec.cuh"
#include "struct_Ray.cuh"
#include "struct_HitPoint.cuh"

struct Triangle{
    Vec v0;
    Vec v1;
    Vec v2;
    Vec v0_vn;
    Vec v1_vn;
    Vec v2_vn;
    Vec v0_vt;
    Vec v1_vt;
    Vec v2_vt;
    int material_id;
    bool is_SmoothNormal = false;
    int shape_id = -1;

    __host__ __device__ Triangle(const Vec &v0 = {0,0,0}, const Vec &v1 = {0,0,0}, const Vec &v2 = {0,0,0},
                                 const Vec &v0_vn = {0,0,0}, const Vec &v1_vn = {0,0,0}, const Vec &v2_vn = {0,0,0},
                                 const Vec &v0_vt = {0,0,0}, const Vec &v1_vt = {0,0,0}, const Vec &v2_vt = {0,0,0}
                                 , const int &material_id = 0, const bool &is_SmoothNormal = false, const int &shape_id = -1)
                                 : v0(v0), v1(v1), v2(v2),
                                 v0_vn(v0_vn), v1_vn(v1_vn), v2_vn(v2_vn),
                                 v0_vt(v0_vt), v1_vt(v1_vt), v2_vt(v2_vt), material_id(material_id) , is_SmoothNormal(is_SmoothNormal), shape_id(shape_id) {}

    __device__ inline bool intersectTest(const Ray &ray, HitPoint *hitpoint) const {
        float t, u, v;
        Vec normal_vec_at_hitpoint;
        // あまりにも細すぎる三角形や小さすぎる三角形など
        if(cross(v1-v0,v2-v0).length() == 0) {
            normal_vec_at_hitpoint = normalize(v0_vn + v1_vn + v2_vn, "normal_vec_at_hitpoint");
        }
        else {
            normal_vec_at_hitpoint = normalize(cross(v1-v0,v2-v0));
        }

        // 三角面がレイと平行の場合をはじく
        if(absdot(normal_vec_at_hitpoint, ray.dir) <= 0)
            return false;

        // 交差点の位置ベクトルは Ray.org + t * Ray.dir = v0 + u(v1-v0) + v(v2-v0)
        float term1 = (1.0f / dot(cross(ray.dir, (v2 - v0)), (v1 - v0)));
        t = term1 * dot(cross(ray.org - v0, v1 - v0), v2 - v0 );
        u = term1 * dot(cross(ray.dir, v2 - v0), ray.org - v0);
        v = term1 * dot(cross(ray.org - v0, v1 - v0), ray.dir );

        // その三角面とは交差していない
        // 三角面を含む平面について，レイと平面の交点は三角面の外側
        if(u < 0 || v < 0 || u+v > 1)
            return false;

        // 自己交差してる
        if (t < 1e-5)
            return false;

        // この時点で注目している三角面が自己交差ではない交差をしていることが確定する
        // 交点でのテクスチャuv座標
        Vec uv_tex_coordinate = u*v1_vt + v*v2_vt + (1-u-v)*v0_vt;

        if(uv_tex_coordinate.x < 0) uv_tex_coordinate.x += (float)((int)(-uv_tex_coordinate.x) + 1);
        else if(uv_tex_coordinate.x > 1) uv_tex_coordinate.x -= (float)((int)(uv_tex_coordinate.x));
        if(uv_tex_coordinate.y < 0) uv_tex_coordinate.y += (float)((int)(-uv_tex_coordinate.y) + 1);
        else if(uv_tex_coordinate.y > 1) uv_tex_coordinate.y -= (float)((int)(uv_tex_coordinate.y));

        // hitpointに各値を格納していく．
        hitpoint->u = uv_tex_coordinate.x;
        hitpoint->v = uv_tex_coordinate.y;
        hitpoint->distance = t;
        hitpoint->position = ray.org + hitpoint->distance * ray.dir;
        if(!is_SmoothNormal) hitpoint->normal = normal_vec_at_hitpoint;
        else hitpoint->normal = normalize(u*v1_vn + v*v2_vn + (1-u-v)*v0_vn, "intersection SmoothNormal");
        hitpoint->normal_NotSmoothed = normal_vec_at_hitpoint;

        return true;
    }

    // 三角面とレイの交差判定ではあるが，SVGFBufferを埋めるためだけの関数
    __device__ inline bool intersectTest_SVGFBuffer(const Ray &ray, Vec& PixelPos) const {
        float u, v, t;

        // 交差点の位置ベクトルは Ray.org + t * Ray.dir = v0 + u(v1-v0) + v(v2-v0)
        float term1 = (1.0f / dot(cross(ray.dir, (v2 - v0)), (v1 - v0)));
        t = term1 * dot(cross(ray.org - v0, v1 - v0), v2 - v0 );
        u = term1 * dot(cross(ray.dir, v2 - v0), ray.org - v0);
        v = term1 * dot(cross(ray.org - v0, v1 - v0), ray.dir );

        // 三角面を含む平面について，レイと平面の交点が三角面の外側にある場合
        if(u < -0.1 || v < -0.1 || u+v > 1.1) {
            return false;
        }

        // 自己交差は起きないが，反転を防ぐ
        if(t < 0) {
            return false;
        }

        // ピクセル座標を補完
        PixelPos = u*v1_vn + v*v2_vn + (1-u-v)*v0_vn;
        return true;
    }

    __host__ __device__ float area() const {
        return 0.5f * (cross(v1-v0, v2-v0)).length();
    }
};

#endif //CUDA_MYPT_STRUCT_TRIANGLE_CUH
