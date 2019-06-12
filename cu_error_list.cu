#include "imp_includes.hcu"

// cuBLAS API errors
const char *cublasGetErrorString(cublasStatus_t err)
{
    switch (err) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
          return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
          return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
          return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
          return "CUBLAS_STATUS_ARCH_MISMATCH";
        // case CUBLAS_STATUS_MAPPING_err:
        //   return "CUBLAS_STATUS_MAPPING_err";
        case CUBLAS_STATUS_EXECUTION_FAILED:
          return "CUBLAS_STATUS_EXECUTION_FAILED";
        // case CUBLAS_STATUS_INTERNAL_err:
        //   return "CUBLAS_STATUS_INTERNAL_err";
        case CUBLAS_STATUS_NOT_SUPPORTED:
          return "CUBLAS_STATUS_NOT_SUPPORTED";
        // case CUBLAS_STATUS_LICENSE_err:
        //   return "CUBLAS_STATUS_LICENSE_err";
      }
      return "<unknown>";
}

  // cuSOLVER API errors
const char *cusolverGetErrorString(cusolverStatus_t err)
{
    switch (err) {
      case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
      case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
      case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
      case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    //   case CUSOLVER_STATUS_MAPPING_err:
    //     return "CUSOLVER_STATUS_MAPPING_err";
      case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    //   case CUSOLVER_STATUS_INTERNAL_err:
    //     return "CUSOLVER_STATUS_INTERNAL_err";
      case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
      case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
      case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
      case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
    }
    return "<unknown>";
}

  // cuRAND API errors
const char *curandGetErrorString(curandStatus_t err)
{
    switch (err) {
      case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
      case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
      case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
    //   case CURAND_STATUS_TYPE_err:
    //     return "CURAND_STATUS_TYPE_err";
      case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
      case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
      case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
      case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
      case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
      case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
    //   case CURAND_STATUS_INTERNAL_err:
    //     return "CURAND_STATUS_INTERNAL_err";
    }
    return "<unknown>";
}

// cuFFT API errors
// static const char *cufftGetErrorString(cufftResult err)
// {
//     switch (err) {
//       case CUFFT_INVALID_PLAN:
//         return "CUFFT_INVALID_PLAN";
//       case CUFFT_ALLOC_FAILED:
//         return "CUFFT_ALLOC_FAILED";
//       case CUFFT_INVALID_TYPE:
//         return "CUFFT_INVALID_TYPE";
//       case CUFFT_INVALID_VALUE:
//         return "CUFFT_INVALID_VALUE";
//       case CUFFT_INTERNAL_err:
//         return "CUFFT_INTERNAL_err";
//       case CUFFT_EXEC_FAILED:
//         return "CUFFT_EXEC_FAILED";
//       case CUFFT_SETUP_FAILED:
//         return "CUFFT_SETUP_FAILED";
//       case CUFFT_INVALID_SIZE:
//         return "CUFFT_INVALID_SIZE";
//       case CUFFT_UNALIGNED_DATA:
//         return "CUFFT_UNALIGNED_DATA";
//       case CUFFT_INCOMPLETE_PARAMETER_LIST:
//         return "CUFFT_INCOMPLETE_PARAMETER_LIST";
//       case CUFFT_INVALID_DEVICE:
//         return "CUFFT_INVALID_DEVICE";
//       case CUFFT_PARSE_err:
//         return "CUFFT_PARSE_err";
//       case CUFFT_NO_WORKSPACE:
//         return "CUFFT_NO_WORKSPACE";
//       case CUFFT_NOT_IMPLEMENTED:
//         return "CUFFT_NOT_IMPLEMENTED";
//       case CUFFT_LICENSE_err:
//         return "CUFFT_LICENSE_err";
//       case CUFFT_NOT_SUPPORTED:
//         return "CUFFT_NOT_SUPPORTED";
//     }
//     return "<unknown>";
// }

  // cuSPARSE API errors
// static const char *cusparseGetErrorString(cusparseStatus_t err)
// {
//     switch (err) {
//       case CUSPARSE_STATUS_NOT_INITIALIZED:
//         return "CUSPARSE_STATUS_NOT_INITIALIZED";
//       case CUSPARSE_STATUS_ALLOC_FAILED:
//         return "CUSPARSE_STATUS_ALLOC_FAILED";
//       case CUSPARSE_STATUS_INVALID_VALUE:
//         return "CUSPARSE_STATUS_INVALID_VALUE";
//       case CUSPARSE_STATUS_ARCH_MISMATCH:
//         return "CUSPARSE_STATUS_ARCH_MISMATCH";
//       case CUSPARSE_STATUS_MAPPING_err:
//         return "CUSPARSE_STATUS_MAPPING_err";
//       case CUSPARSE_STATUS_EXECUTION_FAILED:
//         return "CUSPARSE_STATUS_EXECUTION_FAILED";
//       case CUSPARSE_STATUS_INTERNAL_err:
//         return "CUSPARSE_STATUS_INTERNAL_err";
//       case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
//         return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
//     }
//     return "<unknown>";
// }
