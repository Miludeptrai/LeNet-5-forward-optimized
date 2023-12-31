#include "LeNet5.h"

Network LeNet5_CPU(){
    Network dnn;
    Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new Conv(6, 12, 12, 16, 5, 5, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 8, 8, 2, 2, 2);
    Layer* conv3 = new Conv(16, 4, 4, 120, 4, 4, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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

Network LeNet5_CUDA_SIMPLE(){
    Network dnn;
    Layer* conv1 = new ConvKernel_simple(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new ConvKernel_simple(6, 12, 12, 16, 5, 5, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 8, 8, 2, 2, 2);
    Layer* conv3 = new ConvKernel_simple(16, 4, 4, 120, 4, 4, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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

Network LeNet5_CUDA_SIMPLE_IMPROVED(){
    Network dnn;
    Layer* conv1 = new ConvKernel_simple_improved(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new ConvKernel_simple_improved(6, 12, 12, 16, 5, 5, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 8, 8, 2, 2, 2);
    Layer* conv3 = new ConvKernel_simple_improved(16, 4, 4, 120, 4, 4, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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

Network LeNet5_CUDA_NONE_OPTIMIZE(){
    Network dnn;
    Layer* conv1 = new ConvKernel_none_optimize(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new ConvKernel_none_optimize(6, 12, 12, 16, 5, 5, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 8, 8, 2, 2, 2);
    Layer* conv3 = new ConvKernel_none_optimize(16, 4, 4, 120, 4, 4, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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

Network LeNet5_CUDA_OPTIMIZED(){
    Network dnn;
    Layer* conv1 = new ConvKernel_optimized(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new ConvKernel_optimized(6, 12, 12, 16, 5, 5, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 8, 8, 2, 2, 2);
    Layer* conv3 = new ConvKernel_optimized(16, 4, 4, 120, 4, 4, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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

Network LeNet5_CUDA_TESTING(){
    Network dnn;
    Layer* conv1 = new ConvKernel_testing(1, 28, 28, 6, 5, 5, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new ConvKernel_testing(6, 12, 12, 16, 5, 5, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 8, 8, 2, 2, 2);
    Layer* conv3 = new ConvKernel_testing(16, 4, 4, 120, 4, 4, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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

Network LeNet5_CUDA_NONE_OPTIMIZE_DOU(){
    Network dnn;
    Layer* conv1 = new ConvKernel_none_optimize(1, 56, 56, 6, 9, 9, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 48, 48, 2, 2, 2);
    Layer* conv2 = new ConvKernel_none_optimize(6, 24, 24, 16, 9, 9, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 16, 16, 2, 2, 2);
    Layer* conv3 = new ConvKernel_none_optimize(16, 8, 8, 120, 8, 8, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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
Network LeNet5_CUDA_OPTIMIZED_DOU(){
    Network dnn;
    Layer* conv1 = new ConvKernel_optimized(1, 56, 56, 6, 9, 9, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 48, 48, 2, 2, 2);
    Layer* conv2 = new ConvKernel_optimized(6, 24, 24, 16, 9, 9, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 16, 16, 2, 2, 2);
    Layer* conv3 = new ConvKernel_optimized(16, 8, 8, 120, 8, 8, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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
Network LeNet5_CUDA_SIMPLE_DOU(){
    Network dnn;
    Layer* conv1 = new ConvKernel_simple(1, 56, 56, 6, 9, 9, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 48, 48, 2, 2, 2);
    Layer* conv2 = new ConvKernel_simple(6, 24, 24, 16, 9, 9, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 16, 16, 2, 2, 2);
    Layer* conv3 = new ConvKernel_simple(16, 8, 8, 120, 8, 8, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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
Network LeNet5_CUDA_SIMPLE_IMPROVED_DOU(){
    Network dnn;
    Layer* conv1 = new ConvKernel_simple_improved(1, 56, 56, 6, 9, 9, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 48, 48, 2, 2, 2);
    Layer* conv2 = new ConvKernel_simple_improved(6, 24, 24, 16, 9, 9, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 16, 16, 2, 2, 2);
    Layer* conv3 = new ConvKernel_simple_improved(16, 8, 8, 120, 8, 8, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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






Network LeNet5_CUDA_NONE_OPTIMIZE_QUAD(){
    Network dnn;
    Layer* conv1 = new ConvKernel_none_optimize(1, 112, 112, 6, 17, 17, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 96, 96, 2, 2, 2);
    Layer* conv2 = new ConvKernel_none_optimize(6, 48, 48, 16, 17, 17, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 32, 32, 2, 2, 2);
    Layer* conv3 = new ConvKernel_none_optimize(16, 16, 16, 120, 16, 16, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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
Network LeNet5_CUDA_OPTIMIZED_QUAD(){
    Network dnn;
    Layer* conv1 = new ConvKernel_optimized(1, 112, 112, 6, 17, 17, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 96, 96, 2, 2, 2);
    Layer* conv2 = new ConvKernel_optimized(6, 48, 48, 16, 17, 17, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 32, 32, 2, 2, 2);
    Layer* conv3 = new ConvKernel_optimized(16, 16, 16, 120, 16, 16, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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
Network LeNet5_CUDA_SIMPLE_QUAD(){
    Network dnn;
    Layer* conv1 = new ConvKernel_simple(1, 112, 112, 6, 17, 17, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 96, 96, 2, 2, 2);
    Layer* conv2 = new ConvKernel_simple(6, 48, 48, 16, 17, 17, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 32, 32, 2, 2, 2);
    Layer* conv3 = new ConvKernel_simple(16, 16, 16, 120, 16, 16, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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
Network LeNet5_CUDA_SIMPLE_IMPROVED_QUAD(){
    Network dnn;
    Layer* conv1 = new ConvKernel_simple_improved(1, 112, 112, 6, 17, 17, 1, 0, 0);
    Layer* pool1 = new AvePooling(6, 96, 96, 2, 2, 2);
    Layer* conv2 = new ConvKernel_simple_improved(6, 48, 48, 16, 17, 17, 1, 0, 0);
    Layer* pool2 = new AvePooling(16, 32, 32, 2, 2, 2);
    Layer* conv3 = new ConvKernel_simple_improved(16, 16, 16, 120, 16, 16, 1, 0, 0);
    Layer* fc4 = new FullyConnected(conv3->output_dim(), 84);
    //Layer* fc4 = new FullyConnected(120, 84);
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
    dnn.add_layer(conv3);
    //dnn.add_layer(fc3);
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