# LeNet-5-forward-optimized
**LeNet-5-forward-optimized** is an invidual project. The goal is deploy successfully using CUDA instead of CPU in forward step.

## Usage
Project is based on [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) is a C++ demo of deep neural networks.
I use [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and [FASHION](https://github.com/zalandoresearch/fashion-mnist) dataset.

```shell
mkdir build
cd build
cmake ..
make
```

Run `./demo`.

Result: 

