cmake_minimum_required (VERSION 3.0)
project (myproject LANGUAGES CUDA C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_language(CUDA)

set(CUDA_ARCHITECTURES "75")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false")

set(CUDA_SOURCES
    /src/LeNet5/LeNet5.cc
    /src/LeNet5/LeNet5.h
    /src/cuda_conv
    /src/layer
)

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 

add_executable(x_test/demo ${DIR_SRCS})
# 添加链接库
target_link_libraries(x_test/demo MiniDNNLib)

