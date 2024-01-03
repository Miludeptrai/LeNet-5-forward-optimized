#include "conv_kernel_simple.h"
#include <math.h>
#include <iostream>



//####################################################################################################
void ConvKernel_simple::forward(const Matrix &bottom)
{
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    float *input_data = (float *)bottom.data();
    float *output_data = (float *)malloc(height_out * width_out * channel_out *n_sample * sizeof(float));//(float *)top.data();
    float *weight_data = (float *)weight.data();
    float *bias_data = (float *)bias.data();

    const int num_samples = n_sample;
    const int input_channel = channel_in;
    const int output_channel = channel_out;
    const int kernel_height = height_kernel; // Assuming width_kernel is also K


    printf("%ld %ld \n",bottom.cols(),bottom.rows());
    printf("%d \n",height_out * width_out * channel_out);
    
    GpuTimer timer;
    std::cout << "Convolution - CPU:" << std::endl;
    timer.Start();
    data_cols.resize(n_sample);
    for (int i = 0; i < n_sample; i ++) {
        // im2col
        Matrix data_col;
        im2col(bottom.col(i), data_col);
        //data_cols[i] = data_col;
        // conv by product
        Matrix result = data_col * weight;  // result: (hw_out, channel_out)
        //result.rowwise() += bias.transpose();
        top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
    }
    
    timer.Stop();
    float duration_layer = timer.Elapsed();
    std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
    
    Kernel_simple kernel;
    std::cout << "Convolution - GPU:" << std::endl;
    timer.Start();

    // Launch marker kernel to aid with student function timing
    // gpuInterface.insert_pre_barrier_kernel();

    // Start layer timer
    kernel.conv_forward_gpu_full(output_data, input_data, weight_data,bias_data,
                                 num_samples, output_channel, input_channel,
                                 height_in, width_in, kernel_height);

    // Stop layer timer
    timer.Stop();
     duration_layer = timer.Elapsed();

    
    printError((float *)top.data(),output_data,height_out * width_out * channel_out *n_sample,1);

    // Launch barrier kernel to aid with timing with nsight-compute
    // gpuInterface.insert_post_barrier_kernel();

    std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
}