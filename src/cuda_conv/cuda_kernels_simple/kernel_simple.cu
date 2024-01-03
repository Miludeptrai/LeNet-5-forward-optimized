#include "kernel_simple.h"
#define TILE_WIDTH 32


__global__ void conv_forward_kernel(float *output, const float *input, const float *kernel,const float *bias
                                    const int num_samples, const int output_channel, const int input_channel,
                                    const int height, const int width, const int kernel_size)
{
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;

    int batch_idx = blockIdx.x;
    int output_feature_idx = blockIdx.y;
    int row_idx = blockIdx.z / width_out * TILE_WIDTH + threadIdx.y;
    int col_idx = blockIdx.z % width_out * TILE_WIDTH + threadIdx.x;
    
    float accumulator =  bias[output_feature_idx];

    if (row_idx < height_out && col_idx < width_out)
    {
        for (int input_channel_idx = 0; input_channel_idx < input_channel; input_channel_idx++)
        {
            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++)
            {
                for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++)
                {
                    int input_row = row_idx + kernel_row;
                    int input_col = col_idx + kernel_col;
                    accumulator += input[(batch_idx * (input_channel * height * width)) +
                                         (input_channel_idx * (height * width)) +
                                         (input_row * width) +
                                         input_col] *
                                   kernel[(output_feature_idx * (input_channel * kernel_size * kernel_size)) +
                                          (input_channel_idx * (kernel_size * kernel_size)) +
                                          (kernel_row * kernel_size) +
                                          kernel_col];
                }
            }
        }
        output[(batch_idx * (output_channel * height_out * width_out)) +
               (output_feature_idx * (height_out * width_out)) +
               (row_idx * width_out) +
               col_idx] = accumulator;
    }
}

__host__ void Kernel_simple::conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,const float *bias_data,
                                            const int num_samples, const int output_channel, const int input_channel,
                                            const int height_in, const int width_in, const int kernel_height)
{
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight,*device_bias;
    CHECK(cudaMalloc((void **)&device_input, num_samples * input_channel * height_in * width_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_output, num_samples * output_channel * height_out * width_out * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_bias, output_channel * sizeof(float)));
    CHECK(cudaMemcpy(device_bias, bias_data, output_channel * sizeof(float), cudaMemcpyHostToDevice));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weight, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_bias, bias_data, output_channel * sizeof(float), cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    int height_grid = (height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int width_grid = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(num_samples, output_channel, Z);

    // Launch the kernel
    conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_weight,device_bias, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);
    CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

    // Copy the output back to host
    CHECK(cudaMemcpy(output_data, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_weight));
}
