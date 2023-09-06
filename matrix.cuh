#ifndef CUDA_MYPT_MATRIX_CUH
#define CUDA_MYPT_MATRIX_CUH

#ifndef MYPT_MATRIX_H
#define MYPT_MATRIX_H

#include <vector>



std::vector<std::vector<float>> multiply_matrix(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B) {
    const int H1 = A.size(),W2 = B[0].size(),H2 = B.size();

    std::vector<std::vector<float>> Ans(H1,std::vector<float> (W2));

    for(int i = 0; i < H1; i++) {
        for(int k = 0; k < W2; k++) {
            for(int l = 0; l < H2; l++) {
                Ans[i][k] += A[i][l]*B[l][k];
                //std::cout << i << k << " " << i << l << " " << l << k << std::endl;
            }
        }
    }
    return Ans;
}

Vec rotate(const float x, const float y, const float z, Vec init) {
    std::vector<std::vector<float>> rotateX, rotateY, rotateZ(4,std::vector<float> (4));
    rotateX = {{1, 0, 0, 0}, {0, cos(x), -sin(x), 0}, {0, sin(x), cos(x), 0}, {0, 0, 0, 1}};
    rotateY = {{cos(y), 0, sin(y), 0}, {0, 1, 0, 0}, {-sin(y), 0, cos(y), 0}, {0,0,0,1}};
    rotateZ = {{cos(z), -sin(z), 0,0}, {sin(z), cos(z), 0, 0}, {0,0,1,0}, {0,0,0,1}};
    std::vector<std::vector<float>> before = {{init.x}, {init.y}, {init.z}, {1}};
    before = multiply_matrix(rotateX, before);
    before = multiply_matrix(rotateY, before);
    before = multiply_matrix(rotateZ, before);
    //cout(normalize({before[0][0], before[1][0], before[2][0]}));
    return normalize({before[0][0], before[1][0], before[2][0]}, "rotateMatrix");
}

#endif //MYPT_MATRIX_H


#endif //CUDA_MYPT_MATRIX_CUH

