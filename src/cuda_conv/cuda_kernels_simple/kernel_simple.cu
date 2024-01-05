#include "kernel_simple.h"
#define TILE_WIDTH 32


__global__ void conv_forward_kernel_1(int channel_in,int height_in, int width_in,int height_kernel, 
                            int width_kernel, int height_out, int width_out, int channel_out,
                            float *input_data,  float *weight_data,float *bias_data, float *output_data)
{
    //int batch_idx = blockIdx.z;
    int out_channel_ith = blockIdx.y;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1 ;

    int row_idx = blockIdx.x / width_grid * TILE_WIDTH + threadIdx.y;
    int col_idx = blockIdx.x % width_grid * TILE_WIDTH + threadIdx.x;
    
    float accumulator =  bias_data[out_channel_ith];

    if (row_idx < height_out && col_idx < width_out)
    {
        for (int in_channel_ith = 0; in_channel_ith < channel_in; in_channel_ith++)
        {
            for (int w_row = 0; w_row < height_kernel; w_row++)
            {
                for (int w_col = 0; w_col < width_kernel; w_col++)
                {
                    accumulator += input_data[//(batch_idx * (channel_in * height_in * width_in)) +
                                         (in_channel_ith * (height_in * width_in)) +
                                         ((row_idx + w_row) * width_in) +
                                         col_idx + w_col] *
                                   weight_data[(out_channel_ith * (channel_in * height_kernel * width_kernel)) +
                                          (in_channel_ith * (height_kernel * width_kernel)) +
                                          (w_row * width_kernel) +
                                          w_col];
                }
            }
        }
        output_data[//(batch_idx * (channel_out * height_out * width_out)) +
               (out_channel_ith * (height_out * width_out)) +
               (row_idx * width_out) +
               col_idx] = accumulator;
    }
}

                                     
__host__ void Kernel_simple::cuda_conv_forward(int n_samples,  int channel_in,  int height_in, int width_in,
                                    int height_kernel, int width_kernel,  int channel_out,
                                     float *input_data,  float *weight_data,float *bias_data, float *output_data)
{
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - width_kernel + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight,*device_bias;
    CHECK(cudaMalloc((void **)&device_input, n_samples * channel_in * height_in * width_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_output, n_samples * channel_out * height_out * width_out * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_bias, channel_out * sizeof(float)));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_input, input_data, n_samples * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_bias, bias_data, channel_out * sizeof(float), cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    int height_grid = (height_out - 1) / TILE_WIDTH + 1 ;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(Z, channel_out,1);

    // Launch the kernel
    
    for (int i = 0; i < n_samples; i ++) {
        conv_forward_kernel_1<<<num_blocks_in_grid, num_threads_per_block>>>( channel_in, height_in,  width_in, height_kernel, 
                             width_kernel,  height_out,  width_out,  channel_out,
                            device_input + i*channel_in * height_in * width_in,  device_weight,device_bias, device_output + i*channel_out * height_out * width_out);
    }
    //CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

    // Copy the output back to host
    CHECK(cudaMemcpy(output_data, device_output, n_samples * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_weight));
}
