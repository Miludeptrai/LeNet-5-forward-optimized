#include "kernel_none_optimize.h"
#define TILE_WIDTH 32



__global__ void unroll_kernel_1(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* X, float* X_unroll)
{
    int batch_idx = blockIdx.z;

    int t = blockIdx.x * blockDim.x + threadIdx.x;//
    int height_unroll = height_out * width_out; //2
    int width_unroll = height_kernel * width_kernel * channel_in; //2
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
        int start_c =batch_idx*channel_in*height_in*width_in + c*(width_in*height_in);//

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
                X_unroll[batch_idx*width_unroll*height_unroll + col_unroll*height_unroll + row_unroll] = X[start_c + f_row + f_col];
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
    
    int batch_idx = blockIdx.z;
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
        
        int a0 =batch_idx*channel_in*height_in*width_in + c*(width_in*height_in);
        for (int i =0 ;i <height_unroll ; i++){
            int i_row = i/width_out + ith/width_kernel;
            int i_col = i%width_out + ith%width_kernel;
            unroll_matrix[batch_idx*width_unroll*height_unroll + t*height_unroll+ i] = input_data[a0 + i_row*width_in + i_col];
        }
    }
}

__global__ void unroll_kernel_3(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* input_data, float* unroll_matrix)
{
    int batch_idx = blockIdx.z;

    int t = blockIdx.x * blockDim.x + threadIdx.x; //
    int height_unroll = height_out * width_out; //2
    int hw_kernel = width_kernel * height_kernel;
    int width_unroll = hw_kernel * channel_in;
    int hw_in = height_in * width_in;
    if(t <channel_in* hw_in)
    {   
        //output is a vector size : imagearea x (kernalarea * channel_in)

        //which chanel are we using?
        int c = t / hw_in; //
        //ith of each channel?
        int ith = t % hw_in;//

        //start position 
        int row_in = ith / width_in;//
        int col_in = ith % width_in;//

        int in_value = input_data[batch_idx*channel_in*height_in*width_in + c*(width_in*height_in) + row_in*width_in + col_in];

        for (int p=0;p<height_kernel;p++){
            for (int q=0;q<width_kernel;q++){
                if((row_in -p) < height_out && col_in - q < width_out && (row_in -p) >= 0 && col_in - q >= 0)
                    unroll_matrix[batch_idx*width_unroll*height_unroll + (c * width_kernel * height_kernel + p*width_kernel + q)*height_unroll+ (row_in -p) *width_out + col_in - q] = in_value;
            }
        }
    }
}


__global__ void multi_weight_add_bias_kernel_1(float* unroll_matrix, float *weight_data, float* output_data,float* bias_data,
                                                int height_unroll, int width_unroll,int channel_out)
{
    
    int batch_idx = blockIdx.z;
	int c = blockIdx.x * blockDim.x + threadIdx.x; 
	int r = blockIdx.y * blockDim.y + threadIdx.y; 
	if (r < height_unroll && c < channel_out) {
    float sum = 0 ;
    for (int i = 0; i < width_unroll ; i++) { 
      sum += unroll_matrix[batch_idx*width_unroll*height_unroll + i*height_unroll + r] * weight_data[c*width_unroll + i];
    }
    output_data[batch_idx*height_unroll*channel_out + c*height_unroll + r] = sum + bias_data[c]; 

  } 
}




__host__ void Kernel_none_optimize::cuda_conv_forward( int n_samples,  int channel_in,  int height_in, int width_in,
                                    int height_kernel, int width_kernel,  int channel_out,
                                     float *input_data, float *weight_data,float *bias_data, float *output_data){

    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - width_kernel + 1;

    // Allocate device memory
    float *device_weight,*device_bias, *device_unroll_matrix; 
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

    // dim3 blockSize_unroll(1024);
    // dim3 gridSize_unroll((height_out * width_out  * channel_in-1)/1024 + 1 ,1,batch_size);
    dim3 blockSize_unroll(1024);
    dim3 gridSize_unroll((height_in * width_in  * channel_in-1)/1024 + 1 ,1,batch_size);

    dim3 blockSize_multi(32, 32);
    dim3 gridSize_multi(( channel_out-1)/blockSize_multi.y + 1,(height_out * width_out-1)/blockSize_multi.x + 1,batch_size);


    for (int i = 0; i < nStreams; i++){
		CHECK(cudaStreamCreate(&streams[i]));    
        //Each stream use its GPU mem, and no new GPU location
        CHECK(cudaMalloc((void **)&device_input[i], batch_size * channel_in * height_in * width_in * sizeof(float)));
        CHECK(cudaMalloc((void **)&device_output[i], batch_size * channel_out * height_out * width_out * sizeof(float)));
    }
    
    CHECK(cudaMalloc((void **)&device_unroll_matrix, batch_size * height_out * width_out * channel_in * height_kernel * width_kernel * sizeof(float)));

    // loop through each sample
    for (int stream = 0; stream < nStreams; stream++){
        for (int i = stream * batch_size; i < n_samples; i+=nStreams*batch_size) {
            //There is a problem. Most of time, the final batch dont have enough image, will it cause error?
            //The answer is no, because there are still some images from batch before the last
            int start_in = i * channel_in * height_in * width_in;
            int start_out = i * channel_out * height_out * width_out;
            
            //copy the data to correct stream mem 
            CHECK(cudaMemcpyAsync(device_input[stream], input_data + start_in, min(batch_size,n_samples-i) * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice, streams[stream]));

            unroll_kernel_1<<<gridSize_unroll, blockSize_unroll, 0, streams[stream]>>>
                            (channel_in,  height_in,  width_in,  height_kernel, 
                             width_kernel,  height_out,  width_out, 
                            device_input[stream],  device_unroll_matrix);
                            
            multi_weight_add_bias_kernel_1<<<gridSize_multi,blockSize_multi, 0, streams[stream]>>>
                                (device_unroll_matrix,device_weight,device_output[stream],device_bias
                                ,height_out * width_out, height_kernel * width_kernel * channel_in, channel_out);
                    
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
    CHECK(cudaFree(device_unroll_matrix));

}