#ifndef SRC_CUDA_CONV_BASE_H_
#define SRC_CUDA_CONV_BASE_H_
#pragma once

#include <vector>
#include "../layer.h"
#include "cuda_lib.h"

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
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
	printf("Sample :\n%f %f\n%f %f\n%f %f\n%f %f\n%f %f\n", deviceResult[0],hostResult[0],deviceResult[1],hostResult[1],deviceResult[2],hostResult[2],deviceResult[3],hostResult[3],deviceResult[4],hostResult[4]);
}



class ConvKernel : public Layer
{
protected:
    const int dim_in;
    int dim_out;

    int channel_in;
    int height_in;
    int width_in;
    int channel_out;
    int height_kernel;
    int width_kernel;
    int stride;
    int pad_h;
    int pad_w;

    int height_out;
    int width_out;

    Matrix weight;      // weight param, size=channel_in*h_kernel*w_kernel*channel_out
    Vector bias;        // bias param, size = channel_out
    Matrix grad_weight; // gradient w.r.t weight
    Vector grad_bias;   // gradient w.r.t bias

    std::vector<Matrix> data_cols;

    virtual void init();

public:
    ConvKernel(int channel_in, int height_in, int width_in, int channel_out,
               int height_kernel, int width_kernel, int stride = 1, int pad_w = 0,
               int pad_h = 0) : dim_in(channel_in * height_in * width_in),
                                channel_in(channel_in), height_in(height_in), width_in(width_in),
                                channel_out(channel_out), height_kernel(height_kernel),
                                width_kernel(width_kernel), stride(stride), pad_w(pad_w), pad_h(pad_h)
    {
        init();
    }

    virtual void forward(const Matrix &bottom) = 0;
    virtual void backward(const Matrix &bottom, const Matrix &grad_top);
    virtual void update(Optimizer &opt);
    virtual void im2col(const Vector &image, Matrix &data_col);
    virtual void col2im(const Matrix &data_col, Vector &image);
    virtual int output_dim() { return dim_out; }
    virtual std::vector<float> get_parameters() const;
    virtual std::vector<float> get_derivatives() const;
    virtual void set_parameters(const std::vector<float> &param);
};

#endif 
