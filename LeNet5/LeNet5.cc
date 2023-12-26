#include "LeNet5.h"

Network LeNet5_CPU(){
    Network dnn;
    Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new Conv(6, 12, 12, 16, 5, 5, 1, 0, 0);
    Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer* fc3 = new FullyConnected(pool2->output_dim(), 120);
    Layer* fc4 = new FullyConnected(120, 84);
    Layer* fc5 = new FullyConnected(84, 10);
    Layer* relu1 = new ReLU;
    Layer* relu2 = new ReLU;
    Layer* relu3 = new ReLU;
    Layer* relu4 = new ReLU;
    Layer* softmax = new Softmax;
    dnn.add_layer(conv1);
    dnn.add_layer(relu1);
    dnn.add_layer(pool1);
    dnn.add_layer(conv2);
    dnn.add_layer(relu2);
    dnn.add_layer(pool2);
    dnn.add_layer(fc3);
    dnn.add_layer(relu3);
    dnn.add_layer(fc4);
    dnn.add_layer(relu4);
    dnn.add_layer(fc5);
    dnn.add_layer(softmax);
    // loss
    Loss* loss = new CrossEntropy;
    dnn.add_loss(loss);


    return dnn;
}


