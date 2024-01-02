#ifndef SRC_KERNEL_NONE_OPTIMIZE_H_
#define SRC_KERNEL_NONE_OPTIMIZE_H_
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>


#include "../cuda_kernel.h"

class Kernel_none_optimize : public Kernel
{
public:

    void none_optimize_unroll(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* X, float* X_unroll);
    void none_optimize_matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
                         dim3 blockSize = dim3(1));

    void conv_forward_gpu_full(const int n_samples, const int channel_in, const int height_in,const int width_in,
                                    int height_kernel, int width_kernel, const int channel_out,
                                    const float *input_data, const float *weight_data, float *output_data);

};

#endif