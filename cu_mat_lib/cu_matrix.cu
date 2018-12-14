// Include CUDA libraries
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// Macro definitions
#define look_for_errors try{

#define report_errors }catch(int n){}

#define confirm(cond,err)                   \
if(!(cond))                                 \
{                                           \
    std::cout << "\a" << err << endl;       \
    throw 1;                                \
}

// Include cu_matrix files
#include "block_dim.cu"
#include "cu_error_list.cu"
#include "error_check.cu"
#include "./class/cu_matrix_class.cu"