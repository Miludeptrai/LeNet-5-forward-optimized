#include "kernel_simple.h"
#define TILE_WIDTH 32


__global__ void conv_forward_kernel(int channel_in,int height_in, int width_in,int height_kernel, 
                            int width_kernel, int height_out, int width_out, int channel_out,
                            float *input_data,  float *weight_data,float *bias_data, float *output_data)
{
    //int batch_idx = blockIdx.z;
    int output_feature_idx = blockIdx.y;
    int row_idx = blockIdx.x / width_out * TILE_WIDTH + threadIdx.y;
    int col_idx = blockIdx.x % width_out * TILE_WIDTH + threadIdx.x;
    
    float accumulator =  bias_data[output_feature_idx];
    int start_weight = output_feature_idx * (channel_in * height_kernel * width_kernel);

    //int start_batch = batch_idx * (channel_in * height_in * width_in);

    if (row_idx < height_out && col_idx < width_out)
    {
        for (int channel_in_idx = 0; channel_in_idx < channel_in; channel_in_idx++)
        {
            int start_channel =  channel_in_idx * (height_in * width_in);
            int start_kernel = channel_in_idx * (height_kernel * width_kernel);
            for (int kernel_row = 0; kernel_row < height_kernel; kernel_row++)
            {
                int c_row = (row_idx + kernel_row)*width_in;
                int k_row = kernel_row*width_kernel;
                for (int kernel_col = 0; kernel_col < width_kernel; kernel_col++)
                {
                    int input_col = col_idx + kernel_col;
                    accumulator += input_data[//start_batch +
                                         start_channel + c_row +  input_col] *
                                   weight_data[ start_weight +  start_kernel +k_row + kernel_col];
                }
            }
        }
        output_data[//(batch_idx * (channel_out * height_out * width_out)) +
               (output_feature_idx * (height_out * width_out)) +
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
    int height_grid = (height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int width_grid = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(Z, channel_out,1);
    
    // Launch the kernel
    
    for (int i = 0; i < n_samples; i ++) {
        conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>( channel_in, height_in,  width_in, height_kernel, 
                             width_kernel,  height_out,  width_out,  channel_out,
                            device_input + i*channel_in * height_in * width_in,  device_weight,device_bias, device_output + i*channel_out * height_out * width_out);
    }
    CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

    // Copy the output back to host
    CHECK(cudaMemcpy(output_data, device_output, n_samples * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_weight));
}
