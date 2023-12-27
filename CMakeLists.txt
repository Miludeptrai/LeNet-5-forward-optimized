cmake_minimum_required (VERSION 3.0)
project (myproject LANGUAGES CUDA C CXX)
enable_language(CUDA)

set(CUDA_ARCHITECTURES "75")
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS} -fmad=false -arch=sm_75
    )

set(CUDA_SOURCES
    /src/LeNet5/LeNet5.cc
    /src/LeNet5/LeNet5.h
    /src/cuda_kernels/kernel.cu
    /src/cuda_kernels/kernel.h
    /src/layer
)

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 

add_executable(demo ${DIR_SRCS})
# 添加链接库
target_link_libraries(demo MiniDNNLib)

