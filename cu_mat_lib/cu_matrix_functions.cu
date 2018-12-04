cu_matrix randn(size_t r = 1, size_t c = 1)
{
    cu_matrix a(r,c);
    curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long) clock());
	curandGenerateNormalDouble(prng,a.p,r*c,0.0,1.0); //The number of values requested has to be multiple of 2.
    curandDestroyGenerator(prng);
    return a;
}

cu_matrix mld(const cu_matrix a, const cu_matrix b) // Adapted from CUSOLVER_Library.pdf QR examples
{
    if (a.rows != b.rows)
    {
		std::cout << "Error using mld()." << endl;
        std::cout << "Matrix dimensions must agree." << endl;
		throw 1;
    }
    cu_matrix A = a, B = b; // Copy current matrix to a new matrix for calculations.
    double *d_tau = NULL;
    double *d_work = NULL, alf = 1.0;
    int *devInfo = NULL, lwork = 0, info_gpu = 0;
    cusolverDnHandle_t cusolver_handle = NULL;
    cublasHandle_t cublas_handle = NULL;

    // step 1: create cusolver/cublas handle
    HANDLE_ERROR( cusolverDnCreate(&cusolver_handle) );
    HANDLE_ERROR( cublasCreate(&cublas_handle) );

    // step 2: allocate required extra memory on GPU.
    HANDLE_ERROR( cudaMalloc((void**)&d_tau,sizeof(double)*A.cols) );
    HANDLE_ERROR( cudaMalloc((void**)&devInfo,sizeof(int)) );

    // step 3: query working space of geqrf and ormqr
    HANDLE_ERROR( cusolverDnDgeqrf_bufferSize(cusolver_handle,A.rows,A.cols,A.p,A.rows,&lwork) );
    HANDLE_ERROR( cudaMalloc((void**)&d_work, sizeof(double)*lwork) );

    // step 4: compute QR factorization
    HANDLE_ERROR( cusolverDnDgeqrf(cusolver_handle,A.rows,A.cols,A.p,A.rows,d_tau,d_work,lwork,devInfo) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    // check if QR is good or not
    HANDLE_ERROR( cudaMemcpy(&info_gpu, devInfo, sizeof(int),cudaMemcpyDeviceToHost) );
    if (info_gpu != 0) { cout << "QR decomposition failed." << endl; throw 1; }

    // step 5: compute Q^T*B (CUSOLVER documentation has typos. Follow LAPACK documentation.)
    HANDLE_ERROR( cusolverDnDormqr(cusolver_handle,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,B.rows,B.cols,A.cols,A.p,A.rows,d_tau,B.p,B.rows,d_work,lwork,devInfo) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    // check if QR is good or not
    HANDLE_ERROR( cudaMemcpy(&info_gpu, devInfo, sizeof(int),cudaMemcpyDeviceToHost) );
    if (info_gpu != 0) { cout << "QR decomposition failed." << endl; throw 1; }

    // step 6: compute x = R \ (Q^T*B)
    HANDLE_ERROR( cublasDtrsm(cublas_handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,A.cols,B.cols,&alf,A.p,A.rows,B.p,A.cols) );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // Free resources
    HANDLE_ERROR( cudaFree(d_tau) );
    HANDLE_ERROR( cudaFree(devInfo) );
    HANDLE_ERROR( cudaFree(d_work) );

    HANDLE_ERROR( cublasDestroy(cublas_handle) );
    HANDLE_ERROR( cusolverDnDestroy(cusolver_handle) );

    return B;
}