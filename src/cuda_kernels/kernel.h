#ifndef CONV_GPU_H
#define CONV_GPU_H
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                \
    {                                                              \
        const cudaError_t error = call;                            \
        if (error != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error,       \
                    cudaGetErrorString(error));                    \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

class Kernel
{
public:
    char *concatStr(const char *s1, const char *s2);
    void printDeviceInfo();
    void conv_forward_gpu_full(float *output_data, const float *input_data, const float *weight_data,
                               const int num_samples, const int output_channel, const int input_channel,
                               const int height_in, const int width_in, const int kernel_height);
};

#endif