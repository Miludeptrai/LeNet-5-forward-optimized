#include "kernel_simple.h"
#define TILE_WIDTH 32


__global__ void conv_forward_kernel_1(int channel_in,int height_in, int width_in,int height_kernel, 
                            int width_kernel, int height_out, int width_out, int channel_out,
                            float *input_data,  float *weight_data,float *bias_data, float *output_data)
{
    int batch_idx = blockIdx.z;
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
                    accumulator += input_data[(batch_idx * (channel_in * height_in * width_in)) +
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
        output_data[(batch_idx * (channel_out * height_out * width_out)) +
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
    float *device_weight,*device_bias;
    CHECK(cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_bias, channel_out * sizeof(float)));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_bias, bias_data, channel_out * sizeof(float), cudaMemcpyHostToDevice));
    
    //what is this? yes we have n_samples images, but in some case, we cannot load them all into GPU mem,
    //we seperate them to each batch, an here I set it 32 
    int batch_size = 32;
    // setting cuda streams
    int nStreams = 4;
    float **device_input = new float*[nStreams], **device_output = new float*[nStreams];
    cudaStream_t streams[nStreams];
    
    // Set the kernel dimensions and call the kernel
    int height_grid = (height_out - 1) / TILE_WIDTH + 1;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(Z, channel_out,batch_size);

    for (int i = 0; i < nStreams; i++){
		CHECK(cudaStreamCreate(&streams[i]));    
        //Each stream use its GPU mem, and no new GPU location
        CHECK(cudaMalloc((void **)&device_input[i], batch_size * channel_in * height_in * width_in * sizeof(float)));
        CHECK(cudaMalloc((void **)&device_output[i], batch_size * channel_out * height_out * width_out * sizeof(float)));
    }
    // loop through each sample
    for (int stream = 0; stream < nStreams; stream++){
        for (int i = stream * batch_size; i < n_samples; i+=nStreams*batch_size) {
            //There is a problem. Most of time, the final batch dont have enough image, will it cause error?
            //The answer is no, because there are still some images from batch before the last
            int start_in = i * channel_in * height_in * width_in;
            int start_out = i * channel_out * height_out * width_out;
            
            //copy the data to correct stream mem 
            CHECK(cudaMemcpyAsync(device_input[stream], input_data + start_in, min(batch_size,n_samples-i) * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice, streams[stream]));

            conv_forward_kernel_1<<<num_blocks_in_grid, num_threads_per_block>>>( channel_in, height_in,  width_in, height_kernel, 
                                width_kernel,  height_out,  width_out,  channel_out,
                                device_input[stream],  device_weight,device_bias, device_output[stream]);
            CHECK(cudaMemcpyAsync(output_data + start_out, device_output[stream], min(batch_size,n_samples-i) * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost, streams[stream]));
        }
    }
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
    // Free device memory
    for (int i = 0; i < nStreams; i++){
        CHECK(cudaStreamSynchronize(streams[i]));
        cudaStreamDestroy(streams[i]);
        //delete each stream GPU mem 
        CHECK(cudaFree(device_input[i]));
        CHECK(cudaFree(device_output[i]));
    }
    
    CHECK(cudaFree(device_bias));
    CHECK(cudaFree(device_weight));
}
