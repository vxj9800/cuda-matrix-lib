// Include CUDA libraries
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Macro definitions
#define confirm(cond,err)               \
try{                                    \
    if (cond)                           \
    {                                   \
        std::cout << err << endl;       \
        throw 1;                        \
    }                                   \
}                                       \
catch(int n){}

// Include cu_matrix files
#include "cu_error_list.cu"
#include "error_check.cu"
#include "cu_matrix_class.cu"
#include "cu_matrix_functions.cu"