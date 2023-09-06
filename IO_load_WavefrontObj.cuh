#ifndef CUDA_MYPT_IO_LOAD_WAVEFRONTOBJ_CUH
#define CUDA_MYPT_IO_LOAD_WAVEFRONTOBJ_CUH

#define TINYOBJLOADER_IMPLEMENTATION
#include "external_lib/tinyobjloader/tiny_obj_loader.h"
#include "scene.cuh"
#include "struct_Material.cuh"
#include "IO_load_images.cuh"

void load_object() {
    std::string filename = "OBJ.obj";
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "./"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto &attrib = reader.GetAttrib();
    auto &shapes = reader.GetShapes();
    auto &materials = reader.GetMaterials();

    // オブジェクトデータを読み込む(オブジェクト数だけループ)
    for (size_t s = 0; s < shapes.size(); s++) {
        //std::cout << shapes.size() << std::endl;
        // 面の数だけループ
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // スムーズシェーディングの適用面
            std::vector<unsigned int> is_smooth = shapes[s].mesh.smoothing_group_ids;
            // 頂点の座標
            std::vector<Vec> vertices_position(0);
            // 頂点の法線
            std::vector<Vec> vertices_normal(0);
            // 頂点のテクスチャ座標
            std::vector<Vec> vertices_texture(0);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {

                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];
                vertices_position.emplace_back(vx,vy,vz);

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3*size_t(idx.normal_index)+0];
                    tinyobj::real_t ny = attrib.normals[3*size_t(idx.normal_index)+1];
                    tinyobj::real_t nz = attrib.normals[3*size_t(idx.normal_index)+2];
                    vertices_normal.emplace_back(nx,ny,nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                    tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
                    vertices_texture.emplace_back(tx,ty,0);
                }
            }
            // マテリアルID: 0番は予約されている．(Blenderでマテリアル設定していない場合は0番が選ばれるようにする)
            int material_ID = shapes[s].mesh.material_ids[f];
            material_ID++;

            // sceneに追加
            HOST_ObjectsData.emplace_back(vertices_position[0], vertices_position[1], vertices_position[2],
                                       normalize(vertices_normal[0], "importing Obj"), normalize(vertices_normal[1]), normalize(vertices_normal[2]),
                                       vertices_texture[0], vertices_texture[1], vertices_texture[2],
                                       material_ID, is_smooth[f], s);

            index_offset += fv;
        }
    }

    std::map<std::string, int> TexNameToTexID;
    // マテリアルデータとテクスチャデータを読み込む(マテリアル数だけループ)
    for (size_t s = 0; s < materials.size(); s++) {
        Vec diffuse = Vec{materials[s].diffuse[0], materials[s].diffuse[1], materials[s].diffuse[2]};
        Vec specular = Vec{materials[s].specular[0], materials[s].specular[1], materials[s].specular[2]};
        Vec emission = Vec{materials[s].emission[0], materials[s].emission[1], materials[s].emission[2]};
        if(diffuse.x == 0 || diffuse.y == 0 || diffuse.z == 0) {
            diffuse = diffuse + Vec{1e-4f, 1e-4f, 1e-4f};
        }
        float roughness = materials[s].roughness;
        float IOR = materials[s].ior;
        float metallic = materials[s].metallic;
        int MATERIAL_TYPE = 0;

        std::string TEXTURE_NAME_COLOR = materials[s].diffuse_texname;
        // BaseColorテクスチャが存在するとき
        if(!TEXTURE_NAME_COLOR.empty()) {
            // 画像名だけを取り出す
            int tex_name_pos = (int)TEXTURE_NAME_COLOR.size() - 1;
            while(tex_name_pos != -1 && (TEXTURE_NAME_COLOR[tex_name_pos] != '\\' && TEXTURE_NAME_COLOR[tex_name_pos] != '/')) tex_name_pos--;
            TEXTURE_NAME_COLOR = TEXTURE_NAME_COLOR.substr(tex_name_pos + 1, TEXTURE_NAME_COLOR.size() - tex_name_pos);

            // 読み込んだテクスチャがこれまでに既に読み込まれていない場合
            if(!TexNameToTexID.count(TEXTURE_NAME_COLOR)) {
                TexNameToTexID.insert({TEXTURE_NAME_COLOR, (int)TexNameToTexID.size()});
            }
            HOST_MaterialsData.emplace_back(MATERIAL_TYPE, TexNameToTexID[TEXTURE_NAME_COLOR], diffuse, emission, 0.0, specular.length(), 0.0, metallic, roughness,
                                            0.0, 0.0, 0.0, 0.0, 0.0, IOR, 0.0, 0.0, 0.5);
            // 読み込んだテクスチャ名を出力
            std::cout << "Successfully loaded: " << TEXTURE_NAME_COLOR << " as Texture_" << TexNameToTexID[TEXTURE_NAME_COLOR] << std::endl;
        }
        // Diffuseテクスチャが存在しない時
        else {
            HOST_MaterialsData.emplace_back(MATERIAL_TYPE, -1, diffuse, emission, 0.0, specular.length(), 0.0, metallic, roughness,
                                         0.0,0.0,0.0, 1.0, 0.0,  IOR, 0.01, 0.01, 0.0);
        }
    }
    // Textureデータをテクスチャ配列にまとめる
    load_Textures(TexNameToTexID);
}

#endif //CUDA_MYPT_IO_LOAD_WAVEFRONTOBJ_CUH
