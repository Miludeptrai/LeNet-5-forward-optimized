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


int main1(Network Net) {
  // data
  MNIST dataset("../data/fashion/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist dim_in: " << dim_in << std::endl;
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn = Net;

  // train & test
  SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 5;
  const int batch_size = 128;
  int epoch = 0;
  shuffle_data(dataset.train_data, dataset.train_labels);
  int start_idx = 0;

  int ith_batch = start_idx / batch_size;
  Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                std::min(batch_size, n_train - start_idx));
  dnn.forward(x_batch);
  return 0;
}
int main2(Network Net) {
  // data
  MNIST dataset("../data/fashion/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist dim_in: " << dim_in << std::endl;
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn = Net;
  dnn.forward(dataset.test_data);
  return 0;
}

int main3(Network Net) {
  // data
  MNIST dataset("../data/fashion/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist dim_in: " << dim_in << std::endl;
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn = Net;
  dataset.test_data.resize(56*56, 10000/4);
  dnn.forward(dataset.test_data);
  return 0;
}
int main4(Network Net) {
  // data
  MNIST dataset("../data/fashion/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist dim_in: " << dim_in << std::endl;
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
  // dnn
  Network dnn = Net;
  dataset.test_data.resize(112*112, 10000/16);
  dnn.forward(dataset.test_data);
  return 0;
}
int main(int argc, char** argv) {
  // data
  main1(LeNet5_CUDA_SIMPLE());
}