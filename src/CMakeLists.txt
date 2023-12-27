# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
enable_language(CUDA)

set(CUDA_ARCHITECTURES "75")
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS} -fmad=false -arch=sm_75
    )

set(CUDA_SOURCES
    LeNet5/LeNet5.cc
    LeNet5/LeNet5.h
    cuda_kernels/kernel.cu
    cuda_kernels/kernel.h
    layer
)
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)
aux_source_directory(./LeNet5 DIR_LIB_SRCS)
aux_source_directory(./cuda_kernels DIR_LIB_SRCS)
# 生成链接库
add_library (MiniDNNLib ${DIR_LIB_SRCS})

set_target_properties(MiniDNNLib PROPERTIES CUDA_ARCHITECTURES "75")
