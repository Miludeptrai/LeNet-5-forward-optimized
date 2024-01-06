#ifndef SRC_KERNEL_OPTIMIZED_H_
#define SRC_KERNEL_OPTIMIZED_H_
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>


#include "../cuda_kernel.h"

class Kernel_optimized : public Kernel
{
public:
    void cuda_conv_forward( int n_samples,  int channel_in,  int height_in, int width_in,
                                    int height_kernel, int width_kernel,  int channel_out,
                                     float *input_data,  float *weight_data,float *bias_data, float *output_data);

};

#endif