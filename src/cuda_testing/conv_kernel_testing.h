#ifndef SRC_CONV_KERNEL_TESTING_H_
#define SRC_CONV_KERNEL_TESTING_H_

#include <vector>
#include "../layer.h"
#include "kernel_testing.h"

#include "../cuda_conv/cuda_conv_base.h"

class ConvKernel_testing : public ConvKernel
{
    void forward(const Matrix &bottom);
};

#endif // SRC_LAYER_CONV_KERNEL_H_
