/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"


#include "src/LeNet5/LeNet5.h"

int main() {
  // data
  MNIST dataset("../data/fashion/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist dim_in: " << 112*112 << std::endl;
  // dnn
  Network dnn = LeNet5_CUDA_OPTIMIZED_QUAD();
  dataset.test_data.resize(112*112, 10000/16);
  dnn.forward(dataset.test_data);
  return 0;
}