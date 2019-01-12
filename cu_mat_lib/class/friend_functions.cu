#ifndef _CU_MATRIX_CLASS_FRIEND_FUNCTIONS_INCLUDED_
#define _CU_MATRIX_CLASS_FRIEND_FUNCTIONS_INCLUDED_

/**************************************   Matrix with random numbers   ***********************************************/
cu_mat randn(const size_t r, const size_t c)
{
    size_t r_new = r, c_new = c;
    if ((r%2!=0)&&(c%2!=0))
    {
        r_new = r+1; c_new = c+1;
    }
    cu_mat a(r_new,c_new);
    curandGenerator_t prng;
	HANDLE_ERROR( curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW) );
	HANDLE_ERROR( curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long) clock()) );
	HANDLE_ERROR( curandGenerateNormalDouble(prng,a.p,a.n_rows*a.n_cols,0.0,1.0) ); //The number of values requested has to be multiple of 2.
    HANDLE_ERROR( curandDestroyGenerator(prng) );
    return a(1,r,1,c);
}
cu_mat randn(const size_t n=1){return randn(n,n);}
/***************************************************************************************************************************/


/****************************************   Identity matrix   *******************************************/
__global__ void eye_mat(double* p, const int r, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        p[idx*r+idx] = 1.0;
    }
}
cu_mat eye(const size_t r, const size_t c)
{
    cu_mat temp(r,c);
    size_t n_ele = min(r,c);
    size_t n_threads = block_dim(n_ele);
    eye_mat<<<n_ele/n_threads,n_threads>>>(temp.p,r,n_ele);
    return temp;
}
cu_mat eye(const size_t n){return eye(n,n);}
/***************************************************************************************************************************/


/*****************************************   Matrix left divide   *****************************************/
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
/***************************************************************************************************************************/


/*****************************************   Matrix with all values 1   *****************************************/
cu_mat ones(const size_t r, const size_t c)
{
    return cu_mat(r,c,1);
}
cu_mat ones(const size_t n){return ones(n,n);}
/***************************************************************************************************************************/

/*****************************************   Matrix with all values 0   *****************************************/
cu_mat zeros(const size_t r, const size_t c)
{
    return cu_mat(r,c);
}
cu_mat zeros(const size_t n){return zeros(n,n);}
/***************************************************************************************************************************/


/***************************************   Transpose current matrix   *****************************************/
__global__ void mat_trans(double* a, double* b, size_t rows, size_t cols, double n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t r = idx%rows, c = idx/rows;
    if (idx<n_ele)
    a[c+r*cols] = b[idx];
}
cu_mat trans(const cu_mat a)
{
    cu_mat tmp(a.n_cols,a.n_rows);
    size_t n_ele = a.n_rows*a.n_cols;
    size_t n_threads = block_dim(n_ele);
    mat_trans<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,a.n_rows,a.n_cols,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/***************************************   Horizontal concatenation of two matrices   *****************************************/
cu_mat horzcat(const cu_mat a, const cu_mat b)
{
    confirm(a.n_rows==b.n_rows,"Error: Dimensions of arrays being horizontally concatenated are not consistent.");
    cu_mat tmp(a.n_rows,a.n_cols+b.n_cols);
    HANDLE_ERROR( cudaMemcpy(tmp.p,a.p,a.n_rows*a.n_cols*sizeof(double),cudaMemcpyDeviceToDevice) );
    size_t n_ele = b.n_rows*b.n_cols, n_threads = block_dim(n_ele);
    copymat<<<n_ele/n_threads,n_threads>>>(tmp.p,b.p,a.n_cols*tmp.n_rows,tmp.n_rows,0,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***************************************************************************************************************************/


/***************************************   Vertical concatenation of two matrices   *****************************************/
    // cu_mat temp(r_end-r_begin+1,c_end-c_begin+1);
    // size_t bias = (c_begin-1)*n_rows+r_begin-1;
    // size_t main_rows_bias = n_rows-temp.n_rows;
    // size_t n_ele = temp.n_rows*temp.n_cols;
    // size_t n_threads = block_dim(n_ele);
    // copymat<<<n_ele/n_threads,n_threads>>>(p,temp.p,bias,temp.n_rows,main_rows_bias,n_ele);
    // HANDLE_ERROR( cudaPeekAtLastError() );
cu_mat vertcat(const cu_mat a, const cu_mat b)
{
    confirm(a.n_cols==b.n_cols,"Error: Dimensions of arrays being vertically concatenated are not consistent.");
    cu_mat tmp(a.n_rows+b.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    copymat<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,0,a.n_rows,tmp.n_rows-a.n_rows,n_ele);
    n_ele = b.n_rows*b.n_cols; n_threads = block_dim(n_ele);
    copymat<<<n_ele/n_threads,n_threads>>>(tmp.p,b.p,a.n_rows,b.n_rows,tmp.n_rows-b.n_rows,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***************************************************************************************************************************/

#endif