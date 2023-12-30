#include "conv_kernel_simple.h"
#include <math.h>
#include <iostream>



//####################################################################################################
void ConvKernel_simple::forward(const Matrix &bottom)
{
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    float *input_data = (float *)bottom.data();
    float *output_data = (float *)top.data();
    float *weight_data = (float *)weight.data();

    const int num_samples = n_sample;
    const int input_channel = channel_in;
    const int output_channel = channel_out;
    const int kernel_height = height_kernel; // Assuming width_kernel is also K

    Kernel_simple kernel;
    std::cout << "Convolution - GPU:" << std::endl;

    // Launch marker kernel to aid with student function timing
    // gpuInterface.insert_pre_barrier_kernel();

    // Start layer timer
    GpuTimer timer;
    timer.Start();
    kernel.conv_forward_gpu_full(output_data, input_data, weight_data,
                                 num_samples, output_channel, input_channel,
                                 height_in, width_in, kernel_height);

    // Stop layer timer
    timer.Stop();
    float duration_layer = timer.Elapsed();

    // Launch barrier kernel to aid with timing with nsight-compute
    // gpuInterface.insert_post_barrier_kernel();

    std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
}