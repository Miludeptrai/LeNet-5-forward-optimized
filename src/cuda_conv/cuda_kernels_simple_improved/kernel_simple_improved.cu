#include "kernel_simple_improved.h"
#define TILE_WIDTH 32

#define MAX_CONSTANT_SIZE 8192 

__constant__ float dc_weight[MAX_CONSTANT_SIZE];

__global__ void conv_forward_kernel_2(int channel_in,int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, int channel_out,
                            float *input_data,  float *weight_data,float *bias_data, float *output_data)
{
    int batch_idx = blockIdx.z;
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

    //Here we decided to choose use constant or shared mem for weight
    float * temp_kernel ;
    int weight_lenght = channel_out * channel_in * height_kernel * width_kernel ; 
    if ( weight_lenght <= MAX_CONSTANT_SIZE) {
        temp_kernel= dc_weight + out_channel_ith*(channel_in*width_kernel*height_kernel);//
    }else{
        temp_kernel = (float*)&s_m[(TILE_WIDTH + height_kernel) * (TILE_WIDTH + width_kernel)];
    }

    //local 
    int r = threadIdx.y;
    int c = threadIdx.x;


    float accumulator =  bias_data[out_channel_ith];

    //loop each channel 
    //if use constant skip_weight = in_channel_ith
    //and if not, the thread loads only each channel layer kernel, we need to set it 0
    int i,j, skip_weight;
    for (int in_channel_ith = 0; in_channel_ith < channel_in; in_channel_ith++){
        //read kernal for its channel 

        if ( weight_lenght <= MAX_CONSTANT_SIZE) {
            skip_weight = in_channel_ith;
        }else{
            for ( i = r ;i<height_kernel; i+= TILE_WIDTH){
                for ( j = c ; j < width_kernel; j+= TILE_WIDTH){
                    temp_kernel[i*width_kernel + j] = weight_data[
                                                                out_channel_ith*(channel_in*width_kernel*height_kernel) +
                                                                in_channel_ith*(width_kernel*height_kernel) + i*width_kernel + j];
                }
            }
            skip_weight = 0;
        }

        //load data to shared mem 
        for ( i = r ;i<height_kernel+ TILE_WIDTH -1; i+= TILE_WIDTH){
            for ( j = c ; j < width_tiled ; j+= TILE_WIDTH){
                if(block_start_y  + i < height_in && block_start_x + j < width_in){
                    temp_input[i*width_tiled + j] = input_data[batch_idx * (channel_in*width_in*height_in) +
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
                    accumulator += temp_input[(i+r)*width_tiled + j+c] * temp_kernel[
                                                                        skip_weight*(width_kernel*height_kernel) + i*width_kernel + j]; //temp_kernel[i*width_kernel + j];
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
    if (row_idx < height_out && col_idx < width_out)
    {
        output_data[(batch_idx * (channel_out * height_out * width_out)) +
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
    //float *device_input, *device_output;
    float *device_weight,*device_bias;

    CHECK(cudaMalloc((void **)&device_bias, channel_out * sizeof(float)));
    CHECK(cudaMemcpy(device_bias, bias_data, channel_out * sizeof(float), cudaMemcpyHostToDevice));

    //As we can see, we can use constant to store weight, but what if the size is too long? 
    //so, just check, and in our kernel, we must check again and decide which loctation we will use
    if (channel_out * channel_in * height_kernel * width_kernel <= MAX_CONSTANT_SIZE){
        printf("Using constant!\n");
        CHECK(cudaMemcpyToSymbol(dc_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
    }else{
        CHECK(cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
        CHECK(cudaMemcpy(device_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice));
    }


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

    int share_mem_size = ((TILE_WIDTH + height_kernel) * (TILE_WIDTH + width_kernel) + height_kernel * width_kernel) * sizeof(float);

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

            conv_forward_kernel_2<<<num_blocks_in_grid, num_threads_per_block,share_mem_size>>>( channel_in, height_in,  width_in, height_kernel, 
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
    if (channel_out * channel_in * height_kernel * width_kernel > MAX_CONSTANT_SIZE){
        CHECK(cudaFree(device_weight));
    }
    //
}
