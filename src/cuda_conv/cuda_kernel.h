#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "cuda_lib.h"


class Kernel
{
public:
    virtual char *concatStr(const char *s1, const char *s2);
    virtual void printDeviceInfo();
    virtual void conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,
                               const int num_samples, const int output_channel, const int input_channel,
                               const int height_in, const int width_in, const int kernel_height) =0 ;
};

#endif