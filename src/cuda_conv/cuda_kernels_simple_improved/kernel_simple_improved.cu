#include "kernel_simple_improved.h"
#define TILE_WIDTH 32

#define MAX_CONSTANT_SIZE 8192 

__constant__ float dc_weight[MAX_CONSTANT_SIZE];

__global__ void conv_forward_kernel_2(int channel_in,int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, int channel_out,
                            float *input_data,  float *weight_data,float *bias_data, float *output_data)
{
    //int batch_idx = blockIdx.z;
    int out_channel_ith = blockIdx.y;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1 ;
    int width_tiled = width_kernel+ TILE_WIDTH -1;

    //remember that gridSize.x is used to perform height_out*witdh_out pixel, but in one demension
    //we need to see it in 2D instead of 1D
    //and we caculate where the current block point to
    int block_start_y = blockIdx.x / width_grid * TILE_WIDTH;
    int block_start_x = blockIdx.x % width_grid * TILE_WIDTH;

    //where this thread point to in 
    int row_idx = block_start_y + threadIdx.y;
    int col_idx = block_start_x + threadIdx.x;
    
    //this s_m size : (TILE_WIDTH + height_kernel) * (TILE_WIDTH + width_kernel) + height_kernel * width_kernel
    extern __shared__ float s_m[];
    float * temp_input = (float*)&s_m[0];
    float * temp_kernel = (float*)&s_m[(TILE_WIDTH + height_kernel) * (TILE_WIDTH + width_kernel)];

    //local 
    int r = threadIdx.y;
    int c = threadIdx.x;


    float accumulator =  bias_data[out_channel_ith];

    //loop each channel 
    int i,j;
    for (int in_channel_ith = 0; in_channel_ith < channel_in; in_channel_ith++){
        //read kernal for its channel 

        // for ( i = r ;i<height_kernel; i+= TILE_WIDTH){
        //     for ( j = c ; j < width_kernel; j+= TILE_WIDTH){
        //         temp_kernel[i*width_kernel + j] = weight_data[out_channel_ith*(channel_in*width_kernel*height_kernel) +
        //                                                     in_channel_ith*(width_kernel*height_kernel) + i*width_kernel + j];
        //     }
        // }

        //load data to shared mem 
        for ( i = r ;i<height_kernel+ TILE_WIDTH -1; i+= TILE_WIDTH){
            for ( j = c ; j < width_tiled ; j+= TILE_WIDTH){
                if(block_start_y  + i < height_in && block_start_x + j < width_in){
                    temp_input[i*width_tiled + j] = input_data[//batch_idx * (channel_in*width_in*height_in)
                                                            in_channel_ith*(width_in*height_in) + 
                                                            (block_start_y  + i)*width_in + block_start_x + j];
                }
            }
        }
        __syncthreads();
        //calculate 
        for ( i = 0 ;i<height_kernel; i++){
            for ( j = 0 ; j < width_kernel; j++){
                if (row_idx < height_out && col_idx < width_out) {
                    accumulator += temp_input[(i+r)*width_tiled + j+c] * weight_data[out_channel_ith*(channel_in*width_kernel*height_kernel) +
                                                                        in_channel_ith*(width_kernel*height_kernel) + i*width_kernel + j]; //temp_kernel[i*width_kernel + j];
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (row_idx < height_out && col_idx < width_out)
    {
        output_data[//(batch_idx * (channel_out * height_out * width_out)) +
               (out_channel_ith * (height_out * width_out)) +
               (row_idx * width_out) +
               col_idx] = accumulator;
    }
}

                                     
__host__ void Kernel_simple_improved::cuda_conv_forward(int n_samples,  int channel_in,  int height_in, int width_in,
                                    int height_kernel, int width_kernel,  int channel_out,
                                     float *input_data,  float *weight_data,float *bias_data, float *output_data)
{
    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - width_kernel + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight,*device_bias;
    CHECK(cudaMalloc((void **)&device_input, n_samples * channel_in * height_in * width_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_output, n_samples * channel_out * height_out * width_out * sizeof(float)));
    if (channel_out * channel_in * height_kernel * width_kernel < MAX_CONSTANT_SIZE){
        printf("Using constant!\n");
        CHECK(cudaMemcpyToSymbol(dc_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
        device_weight = dc_weight;
        printf("pointer is %p %p\n",device_weight, dc_weight);
    }else{
        CHECK(cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
        CHECK(cudaMemcpy(device_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice));
    }
    CHECK(cudaMalloc((void **)&device_bias, channel_out * sizeof(float)));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_input, input_data, n_samples * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    
    CHECK(cudaMemcpy(device_bias, bias_data, channel_out * sizeof(float), cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    int height_grid = (height_out - 1) / TILE_WIDTH + 1;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(Z, channel_out,1);

    // Launch the kernel
    int share_mem_size = ((TILE_WIDTH + height_kernel) * (TILE_WIDTH + width_kernel) + height_kernel * width_kernel) * sizeof(float);
    for (int i = 0; i < n_samples; i ++) {
        conv_forward_kernel_2<<<num_blocks_in_grid, num_threads_per_block,share_mem_size>>>( channel_in, height_in,  width_in, height_kernel, 
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
