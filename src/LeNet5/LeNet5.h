/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "../layer.h"
#include "../layer/conv.h"
#include "../cuda_testing/kernel_testing.h"

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
Network LeNet5_CUDA_TESTING();
//Network LeNet5_CUDA_NONE_OPTIMIZE();
// Network LeNet5_CUDA_OPTIMIZED();