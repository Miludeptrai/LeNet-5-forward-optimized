/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
 /*
 step by step to add kernal
 file leNet5.h : include conv_kernel.h; example : #include "../cuda_testing/conv_kernel_testing.h"
 file src/CMAKE : add auxsource; example : aux_source_directory(./cuda_testing DIR_LIB_SRCS)
 file .h from kernal : rewrite define; example : SRC_CONV_KERNEL_TESTING_H_
 file .cc .cu .h : re include 

 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "../layer.h"
#include "../layer/conv.h"
#include "../cuda_conv/cuda_testing/conv_kernel_testing.h"
//#include "../cuda_kernels_simple/conv_kernel_simple.h"

#include "../layer/fully_connected.h"
#include "../layer/ave_pooling.h"
#include "../layer/max_pooling.h"
#include "../layer/relu.h"
#include "../layer/sigmoid.h"
#include "../layer/softmax.h"
#include "../loss.h"
#include "../loss/mse_loss.h"
#include "../loss/cross_entropy_loss.h"
#include "../mnist.h"
#include "../network.h"
#include "../optimizer.h"
#include "../optimizer/sgd.h"

Network LeNet5_CPU();
//Network LeNet5_SIMPLE();
Network LeNet5_CUDA_TESTING();
//Network LeNet5_CUDA_NONE_OPTIMIZE();
// Network LeNet5_CUDA_OPTIMIZED();