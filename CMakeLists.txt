cmake_minimum_required(VERSION 3.23)
project(cuda_mypt CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(cuda_mypt main.cu render_prepare.cuh render.cuh struct_Ray.cuh struct_Camera.cuh
        matrix.cuh struct_Vec.cuh global_values.cuh RayTrace.cuh struct_Xorshift.cuh
        struct_Triangle.cuh struct_HitPoint.cuh struct_Intersection.cuh host_to_device.cuh
        IO_save_png.cuh struct_BVH_node.cuh ConstructBVH.cuh struct_Material.cuh
        IO_load_WavefrontObj.cuh struct_Texture.cuh IO_load_images.cuh scene.cuh
        sampling.cuh struct_stack.cuh RayTrace_Russian_Roulette.cuh NextEventEstimation.cuh
        Algorithms.cuh ImageBasedLighting.cuh NextEventEstimation_FaceLight.cuh
        NextEventEstimation_HDR.cuh NextEventEstimation_PathTracing.cuh Disney_BRDF.cuh Disney_BRDF_sampling.cuh LambertBRDF_sampling.cuh struct_SVGFBuffers.cuh SVGF.cuh test.cuh)

set_target_properties(cuda_mypt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
