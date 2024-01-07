cmake_minimum_required (VERSION 3.0)
project (myproject LANGUAGES CUDA C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_language(CUDA)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

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

#add_executable(demo ${DIR_SRCS})
# 添加链接库
#target_link_libraries(demo MiniDNNLib)

file(GLOB SOURCE_FILES *.cc *.cpp *.c)

foreach(SOURCE_PATH ${SOURCE_FILES})

    get_filename_component(EXECUTABLE_NAME ${SOURCE_PATH} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_PATH})
    target_link_libraries(${EXECUTABLE_NAME} MiniDNNLib)

endforeach(SOURCE_PATH ${SOURCE_FILES})

#add_executable(simple_1_28_28 ${DIR_SRCS})
#target_link_libraries(simple_1_28_28 MiniDNNLib)
#add_executable(simple_625_112_112 ${DIR_SRCS})
#target_link_libraries(simple_625_112_112 MiniDNNLib)
#add_executable(simple_2500_56_56 ${DIR_SRCS})
#target_link_libraries(simple_2500_56_56 MiniDNNLib)
#add_executable(simple_10000_28_28 ${DIR_SRCS})
#target_link_libraries(simple_10000_28_28 MiniDNNLib)

