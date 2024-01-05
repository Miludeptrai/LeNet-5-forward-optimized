#include "conv_kernel_simple_improved.h"
#include <math.h>
#include <iostream>



//####################################################################################################
void ConvKernel_simple_improved::forward(const Matrix &bottom)
{
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    float *input_data = (float *)bottom.data();
    float *output_data = (float *)malloc(height_out * width_out * channel_out *n_sample * sizeof(float));//(float *)top.data();
    float *weight_data = (float *)weight.data();
    float *bias_data = (float *)bias.data();


    printf("%ld %ld \n",bottom.cols(),bottom.rows());
    printf("%d \n",height_out * width_out * channel_out);
    
    GpuTimer timer;
    std::cout << "Convolution - CPU:" << std::endl;
    timer.Start();
    data_cols.resize(n_sample);
    // for (int i = 0; i < n_sample; i ++) {
    //     // im2col
    //     Matrix data_col;
    //     im2col(bottom.col(i), data_col);
    //     //data_cols[i] = data_col;
    //     // conv by product
    //     Matrix result = data_col * weight;  // result: (hw_out, channel_out)
    //     result.rowwise() += bias.transpose();
    //     top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
    // }
    
    timer.Stop();
    float duration_layer = timer.Elapsed();
    std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
    
    Kernel_simple_improved kernel;
    std::cout << "Convolution - GPU:" << std::endl;
    timer.Start();

    // Launch marker kernel to aid with student function timing
    // gpuInterface.insert_pre_barrier_kernel();

    // Start layer timer
    kernel.cuda_conv_forward(n_sample,channel_in,  height_in, width_in,
                                    height_kernel, width_kernel, channel_out,
                                    input_data, weight_data,bias_data, output_data);

    // Stop layer timer
    timer.Stop();
     duration_layer = timer.Elapsed();

    
    printError((float *)top.data(),output_data,height_out * width_out * channel_out *n_sample,1);

    // Launch barrier kernel to aid with timing with nsight-compute
    // gpuInterface.insert_post_barrier_kernel();

    std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
}