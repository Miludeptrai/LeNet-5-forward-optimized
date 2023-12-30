#ifndef SRC_CONV_KERNEL_TESTING_H_
#define SRC_CONV_KERNEL_TESTING_H_
#pragma once

#include <vector>
#include "../../layer.h"
#include "kernel_testing.h"

#include "../cuda_conv_base.h"

class ConvKernel_testing : public ConvKernel
{
    public : 
    ConvKernel_testing(int channel_in, int height_in, int width_in, int channel_out,
               int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
               int pad_h = 0) : ConvKernel(channel_in, height_in, width_in,channel_out,height_kernel,width_kernel,stride,pad_w,pad_h)
    {
    }
    void forward(const Matrix &bottom);
};

#endif // SRC_LAYER_CONV_KERNEL_H_
