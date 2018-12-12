cu_mat randn(size_t r = 1, size_t c = 1)
{
    cu_mat a(r,c);
    curandGenerator_t prng;
	HANDLE_ERROR( curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW) );
	HANDLE_ERROR( curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long) clock()) );
	HANDLE_ERROR( curandGenerateNormalDouble(prng,a.p,r*c,0.0,1.0) ); //The number of values requested has to be multiple of 2.
    HANDLE_ERROR( curandDestroyGenerator(prng) );
    return a;
}

__global__ void eye_mat(double* p, const int r, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        p[idx*r+idx] = 1.0;
    }
}

cu_mat eye(size_t n)
{
    cu_mat temp(n,n);
    size_t n_threads = block_dim(n);
    eye_mat<<<n/n_threads,n_threads>>>(temp.p,n,n);
    return temp;
}

cu_mat eye(size_t r, size_t c)
{
    cu_mat temp(r,c);
    size_t n_ele = min(r,c);
    size_t n_threads = block_dim(n_ele);
    eye_mat<<<n_ele/n_threads,n_threads>>>(temp.p,r,n_ele);
    return temp;
}

cu_mat mld(const cu_mat a, const cu_mat b) // Adapted from CUSOLVER_Library.pdf QR examples
{
    confirm(a.n_rows == b.n_rows,"Error: 'mld()' operation cannot be performed. Matrix dimensions must agree.")

    cu_mat A = a, B = b; // Copy current matrix to a new matrix for calculations.
    double *d_tau = NULL;
    double *d_work = NULL, alf = 1.0;
    int *devInfo = NULL, lwork = 0, info_gpu = 0;
    cusolverDnHandle_t cusolver_handle = NULL;
    cublasHandle_t cublas_handle = NULL;

    // step 1: create cusolver/cublas handle
    HANDLE_ERROR( cusolverDnCreate(&cusolver_handle) );
    HANDLE_ERROR( cublasCreate(&cublas_handle) );

    // step 2: allocate required extra memory on GPU.
    HANDLE_ERROR( cudaMalloc((void**)&d_tau,sizeof(double)*A.n_cols) );
    HANDLE_ERROR( cudaMalloc((void**)&devInfo,sizeof(int)) );

    // step 3: query working space of geqrf and ormqr
    HANDLE_ERROR( cusolverDnDgeqrf_bufferSize(cusolver_handle,A.n_rows,A.n_cols,A.p,A.n_rows,&lwork) );
    HANDLE_ERROR( cudaMalloc((void**)&d_work, sizeof(double)*lwork) );

    // step 4: compute QR factorization
    HANDLE_ERROR( cusolverDnDgeqrf(cusolver_handle,A.n_rows,A.n_cols,A.p,A.n_rows,d_tau,d_work,lwork,devInfo) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    // check if QR is good or not
    HANDLE_ERROR( cudaMemcpy(&info_gpu, devInfo, sizeof(int),cudaMemcpyDeviceToHost) );
    confirm(info_gpu == 0,"Error: 'mld()' operation cannot be performed. QR decomposition failed.");

    // step 5: compute Q^T*B (CUSOLVER documentation has typos. Follow LAPACK documentation.)
    HANDLE_ERROR( cusolverDnDormqr(cusolver_handle,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,B.n_rows,B.n_cols,A.n_cols,A.p,A.n_rows,d_tau,B.p,B.n_rows,d_work,lwork,devInfo) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    // check if QR is good or not
    HANDLE_ERROR( cudaMemcpy(&info_gpu, devInfo, sizeof(int),cudaMemcpyDeviceToHost) );
    confirm(info_gpu == 0,"Error: 'mld()' operation cannot be performed. QR decomposition failed.");

    // step 6: compute x = R \ (Q^T*B)
    HANDLE_ERROR( cublasDtrsm(cublas_handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,A.n_cols,B.n_cols,&alf,A.p,A.n_rows,B.p,A.n_cols) );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // Free resources
    HANDLE_ERROR( cudaFree(d_tau) );
    HANDLE_ERROR( cudaFree(devInfo) );
    HANDLE_ERROR( cudaFree(d_work) );

    HANDLE_ERROR( cublasDestroy(cublas_handle) );
    HANDLE_ERROR( cusolverDnDestroy(cusolver_handle) );

    return B;
}