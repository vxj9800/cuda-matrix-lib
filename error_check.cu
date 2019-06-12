#include "imp_includes.hcu"

//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void HandleError( cudaError_t err,const char *file,int line )
{
    confirm(err == cudaSuccess,cudaGetErrorString( err )<<" in "<<file<<" at "<<line<<".");
}

void HandleError( cublasStatus_t err,const char *file,int line )
{
    confirm(err == CUBLAS_STATUS_SUCCESS,cublasGetErrorString( err )<<" in "<<file<<" at "<<line<<".");
}

void HandleError( cusolverStatus_t err,const char *file,int line )
{
    confirm(err == CUSOLVER_STATUS_SUCCESS,cusolverGetErrorString( err )<<" in "<<file<<" at "<<line<<".");
}

void HandleError( curandStatus_t err,const char *file,int line )
{
    confirm(err == CURAND_STATUS_SUCCESS,curandGetErrorString( err )<<" in "<<file<<" at "<<line<<".");
}

// static void HandleError( cufftResult err,const char *file,int line )
// {
//     confirm(err == CUFFT_SUCCESS,cufftGetErrorString( err )<<" in "<<file<<" at "<<line<<".");
// }

// static void HandleError( cusparseStatus_t err,const char *file,int line )
// {
//     confirm(err == CUSPARSE_STATUS_SUCCESS,cusparseGetErrorString( err )<<" in "<<file<<" at "<<line<<".");
// }
