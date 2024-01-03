#ifndef SRC_CUDA_LIB_H_
#define SRC_CUDA_LIB_H_
#pragma once

#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

//each time compiler see some thing include lib, multiple time 
// It will define any funtion in that lib multiple time 
//use  inline to make it complie one 

inline float computeError(float * a1, float * a2, int n)
{
	float err = 0;
    bool flag = true;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)(a1[i]*1000) - (int)(a2[i]*1000));
        if (abs((int)(a1[i]*1000) - (int)(a2[i]*1000)) > 0.1 && flag){
            printf("first error %d at i=%d, values : %f, %f\n",abs((int)a1[i] - (int)a2[i]),i,a1[i],a2[i]);
            flag = false;
        }
	}
	err /= (n);
	return err;
}

inline void printError(float * deviceResult, float * hostResult, int width, int height)
{
    printf("printerror is called\n");
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
	printf("Sample :\n%f %f\n%f %f\n%f %f\n%f %f\n%f %f\n", deviceResult[0],hostResult[0],deviceResult[1],hostResult[1],deviceResult[2],hostResult[2],deviceResult[3],hostResult[3],deviceResult[4],hostResult[4]);
}

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


#endif 