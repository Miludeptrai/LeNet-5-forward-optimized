#ifndef SRC_CONV_KERNEL_TESTING_H_
#define SRC_CONV_KERNEL_TESTING_H_

#include <vector>
#include "../layer.h"
#include "kernel_testing.h"

#include "../cuda_conv/cuda_conv_base.h"

class ConvKernel_testing : public ConvKernel
{
    ConvKernel_testing(int channel_in, int height_in, int width_in, int channel_out,
               int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
               int pad_h = 0) : dim_in(channel_in * height_in * width_in),
                                channel_in(channel_in), height_in(height_in), width_in(width_in),
                                channel_out(channel_out), height_kernel(height_kernel),
                                width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h)
    {
        init();
    }
    void forward(const Matrix &bottom);
};

#endif // SRC_LAYER_CONV_KERNEL_H_
