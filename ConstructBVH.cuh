#ifndef CUDA_MYPT_CONSTRUCTBVH_CUH
#define CUDA_MYPT_CONSTRUCTBVH_CUH

#include "struct_BVH_node.cuh"
#include "struct_Triangle.cuh"
#include "scene.cuh"
#include "struct_Intersection.cuh"
#include <map>
#include <tuple>

// オブジェクトデータからBVH_treeを初期化する
// BVH_treeの[0] ~ [Objects.size()-1]は一つの三角面のみを含むAABBとする
__host__ void init_BVH_tree () {
    for(int i = 0; i < HOST_ObjectsData.size(); i++) {
        Triangle face = HOST_ObjectsData[i];
        BVH_node_Host new_AABB;
        new_AABB.position_min = Vec (std::min(face.v0.x, std::min(face.v1.x, face.v2.x)),
                                     std::min(face.v0.y, std::min(face.v1.y, face.v2.y)),
                                     std::min(face.v0.z, std::min(face.v1.z, face.v2.z)));
        new_AABB.position_max = Vec (std::max(face.v0.x, std::max(face.v1.x, face.v2.x)),
                                     std::max(face.v0.y, std::max(face.v1.y, face.v2.y)),
                                     std::max(face.v0.z, std::max(face.v1.z, face.v2.z)));
        new_AABB.position_center = (new_AABB.position_min + new_AABB.position_max)/2.0;
        // 配列をコピーしないので不要
        // new_AABB.face_list.push_back(i);
        new_AABB.children = {-1,-1};
        HOST_BVH_tree.push_back(new_AABB);
    }
}

// 注目しているAABBを葉ノードへと変化させる
__host__ void make_leaf(BVH_node_Host &AABB) {
    AABB.children = std::pair{-1,-1};
}

// 2つのAABBを合成する(AABB1にAABB2を追加する)
__host__ BVH_node_Host merge_AABB (BVH_node_Host &AABB1, BVH_node_Host &AABB2) {
    BVH_node_Host new_AABB;
    new_AABB.position_min = Vec(std::min(AABB1.position_min.x, AABB2.position_min.x),
                                std::min(AABB1.position_min.y, AABB2.position_min.y),
                                std::min(AABB1.position_min.z, AABB2.position_min.z));
    new_AABB.position_max = Vec(std::max(AABB1.position_max.x, AABB2.position_max.x),
                                std::max(AABB1.position_max.y, AABB2.position_max.y),
                                std::max(AABB1.position_max.z, AABB2.position_max.z));
    new_AABB.position_center = (new_AABB.position_min + new_AABB.position_max)/2.0;
    // 配列をコピーするので遅い
    //new_AABB.face_list = AABB1.face_list;
    //for(int i = 0; i < AABB2.face_list.size(); i++) new_AABB.face_list.push_back(AABB2.face_list[i]);
    return new_AABB;
}

__host__ void make_root_AABB() {
    std::vector<int> face_list_root_AABB = {0};
    BVH_node_Host root_AABB = HOST_BVH_tree[0];
    for(int i = 1; i < HOST_BVH_tree.size(); i++) {
        root_AABB = merge_AABB(root_AABB, HOST_BVH_tree[i]);
        face_list_root_AABB.push_back(i);
    }
    HOST_BVH_tree.push_back(root_AABB);
    HOST_BVH_tree[HOST_BVH_tree.size()-1].face_list = face_list_root_AABB;
}

// コスト計算における定数
const float Taabb = 1.0;
const float Ttri = 1.0;
// 現在BVH_treeが[0]~[N]まで埋まっているとする(つまりBVH_tree.size() == N+1とする)
// この時，AABB_parentを分割してできたBVH_nodeをBVH_tree[N+1]と[N+2]に記録する
void Divide_AABB (BVH_node_Host &AABB_parent) {
    // AABB_parentに含まれるポリゴンたちを最善の方法で分割することを考える．

    // A(AABB_parent)
    float A_AABB_parent = AABB_parent.surface_area();

    // 各軸における最善のコストと切断後の2つのBVHを記録する
    std::vector<std::tuple<float,BVH_node_Host, BVH_node_Host>> best_cost_of_each_axis(3);
    // x,y,z軸に垂直な平面を用いてAABB_parentを分割する (1つめのfor axis: 0 → 2)
    for(int axis = 0; axis < 3; axis++) {
        // この軸における最善のコストと切断場所を記録する
        float best_cost_of_this_axis = Ttri*(float)AABB_parent.face_list.size() - 2*Taabb;
        int best_cutting_position = -1;

        // それぞれの軸に対応する座標のposition.centerでAABBを整列させる
        // 引数のAABBに含まれる，葉ノードのAABBを参照していく．
        // multimap <面iの葉ノードのposition.center.軸 , i>
        std::multimap<float, int> ordered_face_list;
        for(int & i : AABB_parent.face_list){
            float AABB_center_position;
            // 軸ごとに基準を変える
            if(axis==0) AABB_center_position = HOST_BVH_tree[i].position_center.x;
            else if(axis==1) AABB_center_position = HOST_BVH_tree[i].position_center.y;
            else AABB_center_position = HOST_BVH_tree[i].position_center.z;
            ordered_face_list.insert({AABB_center_position, i});
        }

        // 1つめにその軸における各切断時のTの値を記録し，2つめにその切断面でのS1, 3つめにS2を記録する
        std::vector<std::tuple<float, BVH_node_Host, BVH_node_Host>> cost(AABB_parent.face_list.size()-1);
        // 切断場所の左側におけるAABBをS1とし，右側をS2とする
        BVH_node_Host AABB_S1, AABB_S2;
        // 左から何番目を切断しているかを記録する．
        // 切断面の左にはcutting_position+1だけ面がある
        int cutting_position = 0;

        // 左から切断してみる．その場合の左側のコストを記録する．
        for (auto face : ordered_face_list) {

            auto index = face.second;
            // 最初は一番左端のAABBから始める
            if(cutting_position == 0) {
                AABB_S1 = HOST_BVH_tree[index];
            }
            else if(cutting_position != ordered_face_list.size()-1){
                AABB_S1 = merge_AABB(AABB_S1,HOST_BVH_tree[index]);
            }
            else break; // 右端まで行った
            // S1のコストを計算してcostに切断場所と一緒に保存する．
            // 追加コスト: ((S1の表面積)/(S_AABB_Parentの表面積))*(S1に含まれる面の数)*T_tri
            std::get<0>(cost[cutting_position]) += (AABB_S1.surface_area()/A_AABB_parent)*(float)(cutting_position+1)*Ttri;
            // S1を記録
            std::get<1>(cost[cutting_position]) = AABB_S1;

            cutting_position++;
        }

        // 一つ行き過ぎているので1つ左に戻る
        cutting_position--;

        // 次は右から切断してみる．左側からやる場合と基本的に同様である．
        // ここで切断しながら最善の切断場所とその時のS1とS2を記憶する．
        for(auto face = ordered_face_list.rbegin(); face != ordered_face_list .rend(); face++) {
            auto index = face->second;
            // 最初は一番右端のAABBから始める
            if(cutting_position==ordered_face_list.size()-2) {
                AABB_S2 = HOST_BVH_tree[index];
            }
            else if(cutting_position >= 0){
                AABB_S2 = merge_AABB(AABB_S2,HOST_BVH_tree[index]);
            }
            else break; // 左端まで行った
            // S2のコストを計算してcostに切断場所と一緒に保存する．
            // 追加コスト: ((S2の表面積)/(S_AABB_Parentの表面積))*(S2に含まれる面の数)*T_tri
            std::get<0>(cost[cutting_position]) += (AABB_S2.surface_area()/A_AABB_parent)*(float)(ordered_face_list.size()-cutting_position-1)*Ttri;
            // S2を記録
            std::get<2>(cost[cutting_position]) = AABB_S2;

            // 暫定の最善コストよりも良い切断方法がある場合は記録する
            if(best_cost_of_this_axis > std::get<0>(cost[cutting_position])) {
                best_cutting_position = cutting_position;
                best_cost_of_this_axis = std::get<0>(cost[cutting_position]);
            }

            cutting_position--;
        }

        // best_cutting_positionに基づいてface_listを生成する
        int count_adding_face_list = 0;
        if(best_cutting_position != -1) {
            for (auto face : ordered_face_list) {
                if(count_adding_face_list < best_cutting_position + 1) std::get<1>(cost[best_cutting_position]).face_list.push_back(face.second);
                else std::get<2>(cost[best_cutting_position]).face_list.push_back(face.second);
                count_adding_face_list++;
            }
        }
        // 切断しないということが最善の方法である場合もあるが，とりあえず各軸の最善個コストは記録する．
        if(best_cutting_position == -1) std::get<0>(best_cost_of_each_axis[axis]) =  Ttri*(float)AABB_parent.face_list.size();
        // その他ではS1とS2も記録する
        else best_cost_of_each_axis[axis] = {std::get<0>(cost[best_cutting_position]), std::get<1>(cost[best_cutting_position]), std::get<2>(cost[best_cutting_position])};
    }
    // 全ての軸について，最善の切断方法を確定する．
    float best_cost;
    int best_axis;
    if(std::get<0>(best_cost_of_each_axis[0]) >= std::get<0>(best_cost_of_each_axis[1]) && std::get<0>(best_cost_of_each_axis[0]) >= std::get<0>(best_cost_of_each_axis[2])){
        best_axis = 0;
        best_cost = std::get<0>(best_cost_of_each_axis[0]);
    }
    else if(std::get<0>(best_cost_of_each_axis[1]) >= std::get<0>(best_cost_of_each_axis[0]) && std::get<0>(best_cost_of_each_axis[1]) >= std::get<0>(best_cost_of_each_axis[2])) {
        best_axis = 1;
        best_cost = std::get<0>(best_cost_of_each_axis[1]);
    }
    else {
        best_axis = 2;
        best_cost = std::get<0>(best_cost_of_each_axis[2]);
    }
    // 切断しないということが最善の場合，AABB_parentを葉ノードにする
    if(best_cost == Ttri*(float)AABB_parent.face_list.size()) {
        make_leaf(AABB_parent);
    }
    // そうでない場合，切断する
    else {
        AABB_parent.children.first = (int)HOST_BVH_tree.size();
        AABB_parent.children.second = (int)HOST_BVH_tree.size() + 1;
        HOST_BVH_tree.push_back(std::get<1>(best_cost_of_each_axis[best_axis]));
        HOST_BVH_tree.push_back(std::get<2>(best_cost_of_each_axis[best_axis]));

        // 再帰的に分割していく(注目しているAABBの面リストが要素数1の場合は葉ノードにする)
        int tree_size = (int)HOST_BVH_tree.size();
        if(HOST_BVH_tree[tree_size-2].face_list.size() == 1) make_leaf(HOST_BVH_tree[tree_size-2]);
        else Divide_AABB(HOST_BVH_tree[tree_size-2]);
        if(HOST_BVH_tree[tree_size-1].face_list.size() == 1) make_leaf(HOST_BVH_tree[tree_size-1]);
        else Divide_AABB(HOST_BVH_tree[tree_size-1]);
    }
}

// BVH構築
__host__ void construct_BVH() {
    init_BVH_tree();
    make_root_AABB();
    Divide_AABB(HOST_BVH_tree[HOST_BVH_tree.size()-1]);
}

__device__ bool willIntersectWithAABB(const Ray &Ray, const BVH_node_Device &AABB) {
    float t_max = bINF;
    float t_min = -bINF;

    for (int i = 0; i < 3; i++) {
        float t1, t2, t_near, t_far;
        if(i == 0) {
            if(abs(Ray.dir.x) < sINF) { // x軸に垂直な平面上のレイを飛ばす場合
                if(AABB.position_min.x > Ray.org.x || AABB.position_max.x < Ray.org.x) return false;
                else continue;
            }
            t1 = (AABB.position_min.x - Ray.org.x) / Ray.dir.x;
            t2 = (AABB.position_max.x - Ray.org.x) / Ray.dir.x;
        }
        else if(i == 1) {
            if(abs(Ray.dir.y) < sINF) { // Y軸に垂直な平面上のレイを飛ばす場合
                if(AABB.position_min.y > Ray.org.y || AABB.position_max.y < Ray.org.y) return false;
                else continue;
            }
            t1 = (AABB.position_min.y - Ray.org.y)/Ray.dir.y;
            t2 = (AABB.position_max.y - Ray.org.y)/Ray.dir.y;
        }
        else {
            if(abs(Ray.dir.z) < sINF) { // Z軸に垂直な平面上のレイを飛ばす場合
                if(AABB.position_min.z > Ray.org.z || AABB.position_max.z < Ray.org.z) return false;
                else continue;
            }
            t1 = (AABB.position_min.z - Ray.org.z)/Ray.dir.z;
            t2 = (AABB.position_max.z - Ray.org.z)/Ray.dir.z;
        }

        t_near = min(t1, t2);
        t_far = max(t1, t2);
        t_max = min(t_max, t_far);
        t_min = max(t_min, t_near);

        // レイが外に出る時刻と侵入する時刻が逆転している => 交差していない
        if (t_min > t_max) return false;
    }
    return true;
}

#include "struct_stack.cuh"

__device__ bool willIntersectWithTriangle(const Ray &Ray, Intersection *nearest_intersection, int AABB_ID, const Triangle* Objects, const BVH_node_Device* BVH_tree) {

    // レイとAABBの交差判定を行う
    if(!willIntersectWithAABB(Ray, BVH_tree[AABB_ID])) {
        return false;
    }
    // そのAABBが葉ノードであった場合，そのAABBが含む全ての三角面と交差判定をおこなう(通常の線形探索とほとんど同じ)
    if(BVH_tree[AABB_ID].children.x == -1 && BVH_tree[AABB_ID].children.y == -1) {
        HitPoint hitpoint;
        int ObjectsNumOfThisAABB = BVH_tree[AABB_ID].face_list_size;
        for(int i = 0; i < ObjectsNumOfThisAABB; i++) {
            // オブジェクトID: iの三角面が交差している場合
            // nearest_hitpointに最近点の情報を記録する．
            if(Objects[BVH_tree[AABB_ID].face_list[i]].intersectTest(Ray, &hitpoint)) {
                // 新しく交差しました．その交差点がこれまでの最近点より近ければ最近点を更新
                if(nearest_intersection->hitpoint.distance > hitpoint.distance) {
                    nearest_intersection->hitpoint = hitpoint;
                    nearest_intersection->face_id = BVH_tree[AABB_ID].face_list[i];
                }
            }
        }
        return (nearest_intersection->face_id != -1);
    }
    // 中間ノードである場合，子ノードの両方について再帰的に調べる．
    else {
        bool result1 = willIntersectWithTriangle(Ray, nearest_intersection, BVH_tree[AABB_ID].children.x, Objects, BVH_tree);
        bool result2 = willIntersectWithTriangle(Ray, nearest_intersection, BVH_tree[AABB_ID].children.y, Objects, BVH_tree);
        if(result1 || result2) return true;
        else return false;
    }
}

__device__ bool willIntersectWithTriangle_nonRecursive(const Ray &Ray, Intersection *nearest_intersection, int AABB_ID, const Triangle* Objects, const BVH_node_Device* BVH_tree) {
    const int StackSize = 1024;
    const int TableSize = 5000;

    stack<int2, StackSize> stack;
    int table_index = 0;
    stack.push({~AABB_ID, table_index});
    stack.push({AABB_ID, table_index});
    table_index++;
    int table_willIntersect[TableSize] = {0}; // 0 -> unknown | 1 -> true | -1 -> false
    int2 table_childrenIndex[TableSize] = {int2{0,0}};

    while(!stack.empty()) {
        if(table_index >= TableSize) {
            printf("ERROR in BVH Intersection test: table_index is exceeding its limit\n");
            return false;
        }
        if(stack.full()) {
            printf("WARNING in BVH Intersection test: stack is at its size limit\n");
            return false;
        }
        int ID = stack.top().x;
        int table_index_now = stack.top().y;
        stack.pop();

        // 順伝播
        if(ID >= 0) {
            // レイとAABBの交差判定を行う
            if(!willIntersectWithAABB(Ray, BVH_tree[ID])) {
                table_willIntersect[table_index_now] = -1;
                continue;
            }
            // そのAABBが葉ノードであった場合，そのAABBが含む全ての三角面と交差判定をおこなう(通常の線形探索とほとんど同じ)
            if(BVH_tree[ID].children.x == -1 && BVH_tree[ID].children.y == -1) {
                HitPoint hitpoint;
                int ObjectsNumOfThisAABB = BVH_tree[ID].face_list_size;
                for(int i = 0; i < ObjectsNumOfThisAABB; i++) {
                    // 三角面が交差している場合nearest_hitpointに最近点の情報を記録する．
                    if(Objects[BVH_tree[ID].face_list[i]].intersectTest(Ray, &hitpoint)) {
                        // 新しく交差しました．その交差点がこれまでの最近点より近ければ最近点を更新
                        if(nearest_intersection->hitpoint.distance > hitpoint.distance) {
                            nearest_intersection->hitpoint = hitpoint;
                            nearest_intersection->face_id = BVH_tree[ID].face_list[i];
                        }
                    }
                }
                if(nearest_intersection->face_id != -1) {
                    table_willIntersect[table_index_now] = 1;
                }
                else {
                    table_willIntersect[table_index_now] = -1;
                }
            }
            // 中間ノードである場合，子ノードの両方について調べる必要がある
            else {
                table_childrenIndex[table_index_now] = {table_index, table_index + 1};
                stack.push({~BVH_tree[ID].children.x, table_index});
                stack.push({BVH_tree[ID].children.x, table_index});
                table_index++;
                stack.push({~BVH_tree[ID].children.y, table_index});
                stack.push({BVH_tree[ID].children.y, table_index});
                table_index++;
            }
        }
        // 逆伝播: IDの指すAABBが交差するかどうかを確定する
        else {
            int2 childrenIndex = table_childrenIndex[table_index_now];
            // 中間ノードの場合は子ノードの結果から交差の可否が分かる
            if(table_willIntersect[table_index_now] == 0) {
                if(table_willIntersect[childrenIndex.x] == 1 || table_willIntersect[childrenIndex.y] == 1) {
                    table_willIntersect[table_index_now] = 1;
                }
                else {
                    table_willIntersect[table_index_now] = -1;
                }
            }
        }
    }
    int topAABB = table_willIntersect[0];
    free(table_childrenIndex);
    free(table_willIntersect);
    return (topAABB == 1);
}

#endif //CUDA_MYPT_CONSTRUCTBVH_CUH
