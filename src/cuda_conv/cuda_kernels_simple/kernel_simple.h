#ifndef SRC_KERNEL_SIMPLE_H_
#define SRC_KERNEL_SIMPLE_H_
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>


#include "../cuda_kernel.h"

class Kernel_simple : public Kernel
{
public:
    void conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,const float *bias_data,
                               const int num_samples, const int output_channel, const int input_channel,
                               const int height_in, const int width_in, const int kernel_height);
};

#endif