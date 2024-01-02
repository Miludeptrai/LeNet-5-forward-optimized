#include "conv_kernel_testing.h"
#include <math.h>
#include <iostream>



//####################################################################################################
void ConvKernel_testing::forward(const Matrix &bottom)
{
    
    printf("%ld %ld \n",bottom.cols(),bottom.rows());
    printf("%d \n",height_out * width_out * channel_out);
    int n_sample = bottom.cols();
    top.resize(height_out * width_out * channel_out, n_sample);
    // float *input_data = (float *)bottom.data();
    // float *output_data = (float *)malloc(height_out * width_out * channel_out * n_sample * sizeof(float));//(float *)top.data();
    // float *weight_data = (float *)weight.data();
    // const int num_samples = n_sample;
    // const int input_channel = channel_in;
    // const int output_channel = channel_out;
    // const int kernel_height = height_kernel; // Assuming width_kernel is also K


    // Kernel kernel;
    // std::cout << "Convolution - GPU:" << std::endl;

    // // Launch marker kernel to aid with student function timing
    // // gpuInterface.insert_pre_barrier_kernel();

    // // Start layer timer
    // GpuTimer timer;
    // timer.Start();
    // kernel.conv_forward_gpu_full(output_data, input_data, weight_data,
    //                              num_samples, output_channel, input_channel,
    //                              height_in, width_in, kernel_height);

    // // Stop layer timer
    // timer.Stop();
    // float duration_layer = timer.Elapsed();

    // // Launch barrier kernel to aid with timing with nsight-compute
    // // gpuInterface.insert_post_barrier_kernel();

    // std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;


    ////////////////////////// test unroll pass 
    // printf("Start testing \n");
    
       GpuTimer timer;
       timer.Start();
    data_cols.resize(n_sample);
    for (int i = 0; i < n_sample; i ++) {
      
    //   float *input_data = (float *)bottom.col(i).data();
    //   float *output_data = (float *)malloc(height_out * width_out * height_kernel * width_kernel * channel_in * sizeof(float));

      Kernel_testing kernel_testing;
    //   std::cout << "Convolution - GPU:" << std::endl;

    //   // Launch marker kernel to aid with student function timing
    //   // gpuInterface.insert_pre_barrier_kernel();

    //   // Start layer timer
    Matrix data_col;
    data_col.resize(height_out*width_out, height_kernel*width_kernel * channel_in); 
      kernel_testing.testing_unroll(channel_in, height_in, width_in, height_kernel, 
                             width_kernel,  height_out,  width_out, 
                             (float *)bottom.col(i).data(), (float *)data_col.data());

    //   // Stop layer timer
    //   timer.Stop();
    //   float duration_layer = timer.Elapsed();

    //   // Launch barrier kernel to aid with timing with nsight-compute
    //   // gpuInterface.insert_post_barrier_kernel();

    //   std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;





    //   // im2col
    //   Matrix data_col;
    //   im2col(bottom.col(i), data_col);
    //   //data_cols[i] = data_col;
    //   // conv by product
    //   /////////////////////////////////////// test multiplication 

      
    // Kernel_testing kernel_testing;
         dim3 blockSize(32, 32);
    //     float *input_data1 = (float *)data_col.data();
    //     float *input_data2 = (float *)weight.data();
    //    float *output_data = (float *)malloc(height_out * width_out * channel_out * sizeof(float));
    // timer.Start();
    
    // printf("Start kernal \n");
    Matrix result;
    result.resize(height_out*width_out, channel_out); 
    
       kernel_testing.testing_matrix_multiplication(data_col.data(), weight.data(), result.data(), height_out * width_out, height_kernel * width_kernel * channel_in, channel_out,blockSize);
      
    // timer.Stop();
    // float duration_layer = timer.Elapsed();
    // std::cout << "\t - Layer Time: " << duration_layer << " ms" << std::endl;
      
    // timer.Start();
    //   Matrix result = data_col * weight;  // result: (hw_out, channel_out)
    // printError((float *)result.data(),output_data,height_out * width_out * channel_out,1);
    //   result.rowwise() += bias.transpose();
    //   top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
      
    // timer.Stop();
    // duration_layer = timer.Elapsed();
    
    // std::cout << "\t - CPU Layer Time: " << duration_layer << " ms" << std::endl;
    
    }
    
    timer.Stop();
    float duration_layer = timer.Elapsed();
    
     std::cout << "\t - GPU Layer Time: " << duration_layer << " ms" << std::endl;
    

}
