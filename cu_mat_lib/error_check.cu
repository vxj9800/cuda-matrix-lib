static void HandleError( cudaError_t err,const char *file,int line )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        throw 1;
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// static void CUBLAS_CALL(cublasStatus_t error)
// {
//     switch (error)
//     {
//         case CUBLAS_STATUS_SUCCESS:
//         	return;

//         case CUBLAS_STATUS_NOT_INITIALIZED:
//             std::cout << "CUBLAS_STATUS_NOT_INITIALIZED";
//             return;

//         case CUBLAS_STATUS_INVALID_VALUE:
//             std::cout << "CUBLAS_STATUS_INVALID_VALUE";
//             return;

//         case CUBLAS_STATUS_ARCH_MISMATCH:
//             std::cout << "CUBLAS_STATUS_ARCH_MISMATCH";
//             return;

//         case CUBLAS_STATUS_INTERNAL_ERROR:
//             std::cout << "CUBLAS_STATUS_INTERNAL_ERROR";
//             return;
//     }
//    return;
// }

// static void CUSOLVER_CALL(cusolverStatus_t error)
// {
//     switch (error)
//     {
//         case CUSOLVER_STATUS_SUCCESS:
//         	return;

//         case CUSOLVER_STATUS_NOT_INITIALIZED:
//             std::cout << "CUSOLVER_STATUS_NOT_INITIALIZED";
//             return;

//         case CUSOLVER_STATUS_INVALID_VALUE:
//             std::cout << "CUSOLVER_STATUS_INVALID_VALUE";
//             return;

//         case CUSOLVER_STATUS_ARCH_MISMATCH:
//             std::cout << "CUSOLVER_STATUS_ARCH_MISMATCH";
//             return;

//         case CUSOLVER_STATUS_INTERNAL_ERROR:
//             std::cout << "CUSOLVER_STATUS_INTERNAL_ERROR";
//             return;
//     }
//    return;
// }