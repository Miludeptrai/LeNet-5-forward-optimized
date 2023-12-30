#ifndef SRC_KERNEL_H_
#define SRC_KERNEL_H_
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "cuda_lib.h"


class Kernel
{
public:
    virtual char *concatStr(const char *s1, const char *s2);
    virtual void printDeviceInfo();
};

#endif