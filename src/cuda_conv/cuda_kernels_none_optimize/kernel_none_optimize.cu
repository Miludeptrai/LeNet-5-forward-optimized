#include "kernel_none_optimize.h"
#define TILE_WIDTH 32



__global__ void unroll_kernel_1(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* X, float* X_unroll)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x; //
    int height_unroll = height_out * width_out; //2
    if(t < channel_in*height_unroll)
    {   
        //output is a vector size : imagearea x (kernalarea * channel_in)

        //which chanel are we using?
        int c = t / height_unroll; //
        //Which row are we using?
        int row_unroll = t % height_unroll;//

        //start position 
        int row_out = row_unroll / width_out;//
        int col_out = row_unroll % width_out;//

        //channel start position 
        int start_c = c*(width_in*height_in);//

        //how many rows of the channel before this?
        int w_base =  c * width_kernel * height_kernel; //
        for (int p = 0; p < height_kernel; p++){ 
            int f_row = ( row_out + p)*width_in; // 
            for(int q = 0; q < width_kernel; q++){
                int f_col =  col_out + q; //
                int col_unroll =w_base + p * width_kernel + q; //+

                //Attention, in spite of each channel (vector) store data in row-major, 
                //But our output is a matrix, so we need to perform storing in col-major
                //I hate this =.= 
                X_unroll[col_unroll*height_unroll + row_unroll] = X[start_c + f_row + f_col];
            }
        }
    }
    //We dont use multi here, because, we cant sync whole grid 

}
__global__ void unroll_kernel_2(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* input_data, float* unroll_matrix)
{
int t = blockIdx.x * blockDim.x + threadIdx.x; //
    int height_unroll = height_out * width_out; //2
    int hw_kernel = width_kernel * height_kernel;
    int width_unroll = hw_kernel * channel_in;
    if(t < width_unroll)
    {   
        //output is a vector size : imagearea x (kernalarea * channel_in)

        //which chanel are we using?
        int c = t / hw_kernel; //
        //ith of each filter?
        int ith = t % hw_kernel;//
        
        int a0 = c*(width_in*height_in);
        int *i_row = new int;
        int *i_col = new int;
        for (int i =0 ;i <height_unroll ; i++){
            *i_row = i/width_out + ith/width_kernel;
            *i_col = i%width_out + ith%width_kernel;
            unroll_matrix[ t*height_unroll+ i] = input_data[a0 + (*i_row)*width_in + *i_col];
        }
    }
}


__global__ void multi_weight_add_bias_kernel_1(float* unroll_matrix, float *weight_data, float* output_data,float* bias_data,
                                                int height_unroll, int width_unroll,int channel_out)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x; 
	int r = blockIdx.y * blockDim.y + threadIdx.y; 
	if (r < height_unroll && c < channel_out) {
    float sum = 0 ;
    for (int i = 0; i < width_unroll ; i++) { 
      sum += unroll_matrix[i*height_unroll + r] * weight_data[c*width_unroll + i];
    }
    output_data[c*height_unroll + r] = sum + bias_data[c]; 

  } 
}




__host__ void Kernel_none_optimize::cuda_conv_forward( int n_samples,  int channel_in,  int height_in, int width_in,
                                    int height_kernel, int width_kernel,  int channel_out,
                                     float *input_data, float *weight_data,float *bias_data, float *output_data){

    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - width_kernel + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight,*device_bias, *device_unroll_matrix; 
    CHECK(cudaMalloc((void **)&device_input, n_samples * channel_in * height_in * width_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_output, n_samples * channel_out * height_out * width_out * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_bias, channel_out * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_unroll_matrix, height_out * width_out * channel_in * height_kernel * width_kernel * sizeof(float)));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_input, input_data, n_samples * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_bias, bias_data, channel_out * sizeof(float), cudaMemcpyHostToDevice));

    // // Set the kernel dimensions and call the kernel
    // int height_grid = (height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // int width_grid = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // int Z = height_grid * width_grid;
    // dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 num_blocks_in_grid(n_samples, output_channel, Z);
    dim3 blockSize_unroll(1024);
    dim3 gridSize_unroll((height_out * width_out  * channel_in-1)/1024 + 1 );

    dim3 blockSize_multi(32, 32);
    dim3 gridSize_multi(( channel_out-1)/blockSize_multi.y + 1,(height_out * width_out-1)/blockSize_multi.x + 1,1);

    //printf("block : 1024, grid : %d\n",(height_out * width_out  * max(channel_in,channel_out)-1)/1024 + 1 );    

    for (int i = 0; i < n_samples; i ++) {
        unroll_kernel_2<<<gridSize_unroll, blockSize_unroll>>>
                            (channel_in,  height_in,  width_in,  height_kernel, 
                             width_kernel,  height_out,  width_out, 
                            device_input + i*channel_in * height_in * width_in,  device_unroll_matrix);
                            
        multi_weight_add_bias_kernel_1<<<gridSize_multi,blockSize_multi>>>
                            (device_unroll_matrix,device_weight,device_output + i*channel_out * height_out * width_out,device_bias
                            ,height_out * width_out, height_kernel * width_kernel * channel_in, channel_out);
    }

    // Launch the kernel
    //conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_weight, n_samples, output_channel, channel_in, height_in, width_in, kernel_height);
    //CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

    // Copy the output back to host
    CHECK(cudaMemcpy(output_data, device_output, n_samples * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_weight));
    CHECK(cudaFree(device_unroll_matrix));

}