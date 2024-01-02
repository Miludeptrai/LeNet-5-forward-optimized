#include "kernel_none_optimize.h"
#define TILE_WIDTH 32




// __global__ void unroll_kernel(int channel_in, int height_in, int width_in, int height_kernel, 
//                             int width_kernel, int height_out, int width_out, 
//                             float* X, float* X_unroll)
// {
//     // int t = blockIdx.x * blockDim.x + threadIdx.x; //1
//     // int width_unroll = height_out * width_out; //2*2 
//     // if(t < channel_in*width_unroll)
//     // {
//     //     int c = t / width_unroll; //0 
//     //     int col_unroll = t % width_unroll;//1
//     //     int row_out = col_unroll / width_out;//0
//     //     int col_out = col_unroll % width_out;//1
//     //     int a0 = c*(width_in*height_in);//0
//     //     int w_base = c * width_kernel * height_kernel; //0
//     //     for (int p = 0; p < height_kernel; p++){ 
//     //         int a1 = ( row_out + p)*width_in; // 0
//     //         for(int q = 0; q < width_kernel; q++){
//     //             int a2 =  col_out + q; //1
//     //             int row_unroll = w_base + p * width_kernel + q; 
//     //             X_unroll[row_unroll*width_unroll + col_unroll] = X[a0 + a1 + a2];
//     //         }
//     //     }
//     // }
//     int t = blockIdx.x * blockDim.x + threadIdx.x; //
//     int height_unroll = height_out * width_out; //2
//     if(t < channel_in*height_unroll)
//     {   
//         //output is a vector size : imagearea x (kernalarea * channel_in)

//         //which chanel are we using?
//         int c = t / height_unroll; //
//         //Which row are we using?
//         int row_unroll = t % height_unroll;//

//         //start position 
//         int row_out = row_unroll / width_out;//
//         int col_out = row_unroll % width_out;//

//         //channel start position 
//         int a0 = c*(width_in*height_in);//

//         //how many rows of the channel before this?
//         int w_base =  c * width_kernel * height_kernel; //
//         for (int p = 0; p < height_kernel; p++){ 
//             int a1 = ( row_out + p)*width_in; // 
//             for(int q = 0; q < width_kernel; q++){
//                 int a2 =  col_out + q; //
//                 int col_unroll =w_base + p * width_kernel + q; //+

//                 //Attention, in spite of each channel (vector) store data in row-major, 
//                 //But our output is a matrix, so we need to perform storing in col-major
//                 //I hate this =.= 
//                 X_unroll[col_unroll*height_unroll + row_unroll] = X[a0 + a1 + a2];
//             }
//         }
//     }
// }
// __global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
// {
// 	int c = blockIdx.x * blockDim.x + threadIdx.x; 
// 	int r = blockIdx.y * blockDim.y + threadIdx.y; 
// 	if (r < m && c < k) {
//     float sum = 0 ;
//     for (int i = 0; i < n ; i++) { 
//       sum += A[i*m + r] * B[c*n + i];
//     }
//     C[c*m + r] = sum ; 

//   } 
//   	// __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
// 	// __shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
// 	// int c = blockIdx.x * blockDim.x + threadIdx.x; 
// 	// int r = blockIdx.y * blockDim.y + threadIdx.y; 
// 	// float sum = 0 ;
// 	// for(int b = 0 ; b < (n-1)/TILE_WIDTH + 1 ; b++){
//     //     if (r<m && b*TILE_WIDTH+threadIdx.x<n)
//     //         s_A[threadIdx.y][threadIdx.x] = A[r + (b*TILE_WIDTH+threadIdx.x)*m];
//     //     else
//     //         s_A[threadIdx.y][threadIdx.x] = 0;
//     //     if (b*TILE_WIDTH+threadIdx.y<n && c < k)
//     //         s_B[threadIdx.y][threadIdx.x] = B[(b*TILE_WIDTH + threadIdx.y) + c*n];
//     //     else
//     //         s_B[threadIdx.y][threadIdx.x] = 0;
//     //     __syncthreads();
        
// 	// 	if (r<m && c<k){
//     //         for(int j = 0; j < TILE_WIDTH; ++j)
//     //             sum += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
//     //     }
        
//     //         __syncthreads();
// 	// }
//     // if (r<m && c<k)
//     //     C[c * m + r] = sum; 
// }

// __host__ void Kernel_none_optimize::none_optimize_unroll(int channel_in, int height_in, int width_in, int height_kernel, 
//                             int width_kernel, int height_out, int width_out, 
//                             float* X, float* X_unroll)
// {
//     // Allocate device memory
//     //this->printDeviceInfo();
//     float *device_input, *device_output;
//     CHECK(cudaMalloc((void **)&device_input, channel_in * height_in * width_in * sizeof(float)));
//     CHECK(cudaMalloc((void **)&device_output, height_out * width_out * height_kernel * width_kernel * channel_in * sizeof(float)));

//     // Copy input and mask data to device
//     CHECK(cudaMemcpy(device_input, X, channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));

//     // Set the kernel dimensions and call the kernel
//     dim3 num_threads_per_block(1024);
//     dim3 num_blocks_in_grid((height_out * width_out  * channel_in-1)/1024 + 1 );

//     // Launch the kernel
//     unroll_kernel<<<num_blocks_in_grid, num_threads_per_block>>>( channel_in,  height_in,  width_in,  height_kernel, 
//                              width_kernel,  height_out,  width_out, 
//                             device_input,  device_output);
//     CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

//     // Copy the output back to host
//     CHECK(cudaMemcpy(X_unroll, device_output, height_out * width_out * height_kernel * width_kernel * channel_in * sizeof(float), cudaMemcpyDeviceToHost));

//     // Free device memory
//     CHECK(cudaFree(device_input));
//     CHECK(cudaFree(device_output));
// }


// __host__ void Kernel_none_optimize::none_optimize_matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
//                          dim3 blockSize )
// {
//     // Allocate device memory
//     //this->printDeviceInfo();
//     float* d_A, * d_B, * d_C;
//     CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
//     CHECK(cudaMalloc(&d_B, n * k * sizeof(float)));
//     CHECK(cudaMalloc(&d_C, m * k * sizeof(float)));

//     // TODO: Copy data to device memories
//     CHECK(cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice));
//     CHECK(cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice));
    
//     dim3 gridSize((k-1)/blockSize.y + 1,(m-1)/blockSize.x + 1,1); // TODO: Compute gridSize
    
//         matrix_multiplication_kernel2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

//     // TODO: Copy result from device memory
//     CHECK(cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost));

//     // TODO: Free device memories
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }



__global__ void conv_forward_kernel(int channel_in, int height_in, int width_in, int height_kernel, 
                            int width_kernel, int height_out,int width_out,int channel_out,
                            const float *input_data, float*unroll_matrix, const float *weight_data, float *output_data) //float* X, float* X_unroll 
{
    int t = blockIdx.x * blockDim.x + threadIdx.x; //
    int height_unroll = height_out * width_out; //2
    int width_unroll = height_kernel * width_kernel * channel_in;
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
                //X_unroll[col_unroll*height_unroll + row_unroll] = X[a0 + a1 + a2];
                unroll_matrix[col_unroll*height_unroll + row_unroll] = input_data[a0 + a1 + a2];
            }
        }
    }
    __syncthreads();
    if (t < height_unroll * channel_out) {
        int c = t % channel_out ; 
        int r = t / channel_out;
        //if (r < m && c < k) {
            float sum = 0 ;
            for (int i = 0; i < width_unroll ; i++) { 
                sum += unroll_matrix[i*height_unroll + r] * weight_data[c*width_unroll + i];
            }
            output_data[c*height_unroll + r] = sum ; 
        //}
    }
}

__host__ void Kernel_none_optimize::conv_forward_gpu_full( int n_samples,  int channel_in,  int height_in, int width_in,
                                    int height_kernel, int width_kernel,  int channel_out,
                                     float *input_data, float *weight_data, float *output_data){

    const int height_out = height_in - height_kernel + 1;
    const int width_out = width_in - width_kernel + 1;

    // Allocate device memory
    float *device_input, *device_output, *device_weight, *device_unroll_matrix;
    CHECK(cudaMalloc((void **)&device_input, n_samples * channel_in * height_in * width_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_output, n_samples * channel_out * height_out * width_out * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_weight, channel_out * channel_in * height_kernel * width_kernel * sizeof(float)));
    CHECK(cudaMalloc((void **)&device_unroll_matrix, height_out * width_out * channel_in * height_kernel * width_kernel * sizeof(float)));

    // Copy input and mask data to device
    CHECK(cudaMemcpy(device_input, input_data, n_samples * channel_in * height_in * width_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(device_weight, weight_data, channel_out * channel_in * height_kernel * width_kernel * sizeof(float), cudaMemcpyHostToDevice));

    // // Set the kernel dimensions and call the kernel
    // int height_grid = (height_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // int width_grid = (width_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // int Z = height_grid * width_grid;
    // dim3 num_threads_per_block(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 num_blocks_in_grid(n_samples, output_channel, Z);
    
    dim3 num_threads_per_block(1024);
    dim3 num_blocks_in_grid((height_out * width_out  * max(channel_in,channel_out)-1)/1024 + 1 );

    for (int i = 0; i < n_samples; i ++) {
        conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(channel_in, height_in, width_in, height_kernel, 
                            width_kernel, height_out, width_out, channel_out,
                            device_input + i*channel_in * height_in * width_in, device_unroll_matrix, 
                            device_weight + i*channel_in * height_in * width_in, device_output + i*channel_in * height_in * width_in);
    }

    // Launch the kernel
    //conv_forward_kernel<<<num_blocks_in_grid, num_threads_per_block>>>(device_output, device_input, device_weight, n_samples, output_channel, channel_in, height_in, width_in, kernel_height);
    CHECK(cudaDeviceSynchronize()); // Ensure that the GPU has completed the computation

    // Copy the output back to host
    CHECK(cudaMemcpy(output_data, device_output, n_samples * channel_out * height_out * width_out * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK(cudaFree(device_input));
    CHECK(cudaFree(device_output));
    CHECK(cudaFree(device_weight));
    CHECK(cudaFree(device_unroll_matrix));

}