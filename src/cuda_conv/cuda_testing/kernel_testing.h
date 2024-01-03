#ifndef SRC_KERNEL_TESTING_H_
#define SRC_KERNEL_TESTING_H_
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>


#include "../cuda_kernel.h"

class Kernel_testing : public Kernel
{
public:

    void testing_unroll(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* X, float* X_unroll);
    void testing_matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
                         dim3 blockSize = dim3(1));
    
};

#endif