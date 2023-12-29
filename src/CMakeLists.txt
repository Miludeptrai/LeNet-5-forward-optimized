# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_LIB_SRCS 变量
aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)
aux_source_directory(./LeNet5 DIR_LIB_SRCS)
aux_source_directory(./cuda_testing DIR_LIB_SRCS)
aux_source_directory(./cuda_kernels_none_optimize DIR_LIB_SRCS)
aux_source_directory(./cuda_kernels_optimized DIR_LIB_SRCS)
# 生成链接库
add_library (MiniDNNLib ${DIR_LIB_SRCS})

set_target_properties(MiniDNNLib PROPERTIES CUDA_ARCHITECTURES "75")
