static void HandleError( cudaError_t err,const char *file,int line )
{
    if (err != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
        throw 1;
    }
}

static void HandleError(cublasStatus_t err,const char *file,int line )
{
    switch (err)
    {
        case CUBLAS_STATUS_SUCCESS:
        	return;

        case CUBLAS_STATUS_NOT_INITIALIZED:
            std::cout << "CUBLAS_STATUS_NOT_INITIALIZEDn in " << file << " at line " << line << "." << endl;
            throw 1;

        case CUBLAS_STATUS_INVALID_VALUE:
            std::cout << "CUBLAS_STATUS_INVALID_VALUE" << file << " at line " << line << "." << endl;
            throw 1;

        case CUBLAS_STATUS_ARCH_MISMATCH:
            std::cout << "CUBLAS_STATUS_ARCH_MISMATCH" << file << " at line " << line << "." << endl;
            throw 1;

        case CUBLAS_STATUS_INTERNAL_ERROR:
            std::cout << "CUBLAS_STATUS_INTERNAL_ERROR" << file << " at line " << line << "." << endl;
            throw 1;
    }
   return;
}

static void HandleError(cusolverStatus_t err,const char *file,int line )
{
    switch (err)
    {
        case CUSOLVER_STATUS_SUCCESS:
        	return;

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            std::cout << "CUSOLVER_STATUS_NOT_INITIALIZED" << file << " at line " << line << "." << endl;
            throw 1;

        case CUSOLVER_STATUS_INVALID_VALUE:
            std::cout << "CUSOLVER_STATUS_INVALID_VALUE" << file << " at line " << line << "." << endl;
            throw 1;

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            std::cout << "CUSOLVER_STATUS_ARCH_MISMATCH" << file << " at line " << line << "." << endl;
            throw 1;

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            std::cout << "CUSOLVER_STATUS_INTERNAL_ERROR" << file << " at line " << line << "." << endl;
            throw 1;
    }
   return;
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))