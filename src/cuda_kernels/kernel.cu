#include "kernel.h"
#define TILE_WIDTH 16

char *Kernel::concatStr(const char *s1, const char *s2)
{
    char *result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void Kernel::printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);
    printf("****************************\n");
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *kernel,
                                    const int num_samples, const int output_channel, const int input_channel,
                                    const int height, const int width, const int kernel_size)
{
    const int height_out = height - kernel_size + 1;
    const int width_out = width - kernel_size + 1;

    int batch_idx = blockIdx.x;
    int output_feature_idx = blockIdx.y;
    int row_idx = blockIdx.z / width_out * TILE_WIDTH + threadIdx.y;
    int col_idx = blockIdx.z % width_out * TILE_WIDTH + threadIdx.x;

    float accumulator = 0.0f;

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


__global__ void unroll_kernel(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* X, float* X_unroll)
{
    // int t = blockIdx.x * blockDim.x + threadIdx.x; //1
    // int width_unroll = height_out * width_out; //2*2 
    // if(t < channel_in*width_unroll)
    // {
    //     int c = t / width_unroll; //0 
    //     int col_unroll = t % width_unroll;//1
    //     int row_out = col_unroll / width_out;//0
    //     int col_out = col_unroll % width_out;//1
    //     int a0 = c*(width_in*height_in);//0
    //     int w_base = c * width_kernel * height_kernel; //0
    //     for (int p = 0; p < height_kernel; p++){ 
    //         int a1 = ( row_out + p)*width_in; // 0
    //         for(int q = 0; q < width_kernel; q++){
    //             int a2 =  col_out + q; //1
    //             int row_unroll = w_base + p * width_kernel + q; 
    //             X_unroll[row_unroll*width_unroll + col_unroll] = X[a0 + a1 + a2];
    //         }
    //     }
    // }
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
        int a0 = c*(width_in*height_in);//

        //how many rows of the channel before this?
        int w_base =  c * width_kernel * height_kernel; //
        for (int p = 0; p < height_kernel; p++){ 
            int a1 = ( row_out + p)*width_in; // 
            for(int q = 0; q < width_kernel; q++){
                int a2 =  col_out + q; //
                int col_unroll =w_base + p * width_kernel + q; //+

                //Attention, in spite of each channel (vector) store data in row-major, 
                //But our output is a matrix, so we need to perform storing in col-major
                //I hate this =.= 
                X_unroll[col_unroll*height_unroll + row_unroll] = X[a0 + a1 + a2];
            }
        }
    }
}
__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	int c = blockIdx.x * blockDim.x + threadIdx.x; 
	int r = blockIdx.y * blockDim.y + threadIdx.y; 
	float sum = 0 ;
	for(int b = 0 ; b < (n-1)/TILE_WIDTH + 1 ; b++){
        if (r<m && b*TILE_WIDTH+threadIdx.x<n)
            s_A[threadIdx.y][threadIdx.x] = A[r*n + b*TILE_WIDTH+threadIdx.x];
        else
            s_A[threadIdx.y][threadIdx.x] = 0;
        if (b*TILE_WIDTH+threadIdx.y<n && c < k)
            s_B[threadIdx.y][threadIdx.x] = B[(b*TILE_WIDTH + threadIdx.y)*k + c];
        else
            s_B[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();
		if (r<m && c<k){
            for(int j = 0; j < TILE_WIDTH; ++j)
                sum += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
            __syncthreads();
        }
	}
    if (r<m && c<k)
        C[r * k + c] = sum; 
}

__host__ void Kernel::conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,
                                            const int num_samples, const int output_channel, const int input_channel,
                                            const int height_in, const int width_in, const int kernel_height)
{
    const int height_out = height_in - kernel_height + 1;
    const int width_out = width_in - kernel_height + 1;

    // Allocate device memory
    //this->printDeviceInfo();
    float *device_input, *device_output, *device_weight;
    CHECK(cudaMalloc((void **)&device_input, num_samples * input_channel * height_in * width_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_output, num_samples * output_channel * height_out * width_out * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_weight, output_channel * input_channel * kernel_height * kernel_height * sizeof(float)));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_input, input_data, num_samples * input_channel * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weight, weight_data, output_channel * input_channel * kernel_height * kernel_height * sizeof(float), cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    int height_grid = (height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int width_grid = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Z = height_grid * width_grid;
    dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 num_blocks_in_grid(num_samples, output_channel, Z);

    // Launch the kernel

    conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_weight, num_samples, output_channel, input_channel, height_in, width_in, kernel_height);
    CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

    // Copy the output back to host
    CHECK(cudaMemcpy(output_data, device_output, num_samples * output_channel * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_weight));
}

__host__ void Kernel::testing_unroll(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out, int width_out, 
                            float* X, float* X_unroll)
{
    // Allocate device memory
    //this->printDeviceInfo();
    float *device_input, *device_output;
    CHECK(cudaMalloc((void **)&device_input, channel_in * height_in * width_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_output, height_out * width_out * height_kernel * width_kernel * channel_in * sizeof(float)));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_input, X, channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));

    // Set the kernel dimensions and call the kernel
    dim3 num_threads_per_block(1024);
    dim3 num_blocks_in_grid((height_out * width_out  * channel_in-1)/1024 + 1 );

    // Launch the kernel
    unroll_kernel<<<num_blocks_in_grid, num_threads_per_block>>>( channel_in,  height_in,  width_in,  height_kernel, 
                             width_kernel,  height_out,  width_out, 
                            device_input,  device_output);
    CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

    // Copy the output back to host
    CHECK(cudaMemcpy(X_unroll, device_output, height_out * width_out * height_kernel * width_kernel * channel_in * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
}


__host__ void Kernel::testing_matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
                         dim3 blockSize = dim3(1))
{
    // Allocate device memory
    //this->printDeviceInfo();
    float* d_A, * d_B, * d_C;
    CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
    CHECK(cudaMalloc(&d_B, n * k * sizeof(float)));
    CHECK(cudaMalloc(&d_C, m * k * sizeof(float)));

    // TODO: Copy data to device memories
    CHECK(cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 gridSize((k-1)/blockSize.y + 1,(m-1)/blockSize.x + 1,1); // TODO: Compute gridSize
    
    if (kernelType == 1)
        matrix_multiplication_kernel1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
    else if (kernelType == 2)
        matrix_multiplication_kernel2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

    // TODO: Copy result from device memory
    CHECK(cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));

    // TODO: Free device memories
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}