/**************************************   Access single element of the matrix   *******************************************/
cu_mat cu_mat::operator()(const size_t r, const size_t c)
{
    confirm((r<=n_rows)&&(c<=n_cols),"Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
    cu_mat temp(1,1);
    HANDLE_ERROR( cudaMemcpy(temp.p,p+(c-1)*n_rows+r-1,sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
    return temp;
}
/***********************************************************************************************************************/


/**************************************   Access sub-matrix   *******************************************/
__global__ void submat(double* current, double* tmp, size_t bias, size_t tmp_rows, size_t main_rows_bias, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    tmp[idx] = current[bias+idx+idx/tmp_rows*main_rows_bias];
}
cu_mat cu_mat::operator()(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end)
{
    confirm((r_end<=n_rows)&&(c_end<=n_cols),"Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
    cu_mat temp(r_end-r_begin+1,c_end-c_begin+1);
    size_t bias = (c_begin-1)*n_rows+r_begin-1;
    size_t main_rows_bias = n_rows-temp.n_rows;
    size_t n_ele = temp.n_rows*temp.n_cols;
    size_t n_threads = block_dim(n_ele);
    submat<<<n_ele/n_threads,n_threads>>>(p,temp.p,bias,temp.n_rows,main_rows_bias,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return temp;
}
/***********************************************************************************************************************/


/***************************************   Assignment operator   **************************************/
cu_mat& cu_mat::operator=(const cu_mat b)
{
    if ((n_rows!=b.n_rows) || (n_cols!=b.n_cols))
    {
        HANDLE_ERROR( cudaFree(p) );
        n_rows = b.n_rows; n_cols = b.n_cols;
        HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) ); // Allocate memory on GPU.
    }
    HANDLE_ERROR( cudaMemcpy(p,b.p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy array from GPU to GPU
    return *this;
}
/***********************************************************************************************************************/


/***************************************   Matrix multiplication   **************************************/
cu_mat cu_mat::operator*(const cu_mat b)
{
    confirm(n_cols == b.n_rows,"Error : Matrix multiplication is not possible. Inner matrix dimensions must agree.");
    cu_mat c(n_rows,b.n_cols);
    double alf = 1.0, bet = 0;
    cublasHandle_t handle;
	HANDLE_ERROR( cublasCreate(&handle) );
	HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
    HANDLE_ERROR( cublasDestroy(handle) );
    return c;
}
/***********************************************************************************************************************/


/***************************************   Matrix addition   ****************************************/
__global__ void addition(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    c[idx] = a[idx] + b[idx];
}
cu_mat cu_mat::operator+(const cu_mat b)                             // Matrix addition operator
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = n_rows*n_cols;
    size_t n_threads = block_dim(n_ele);
    addition<<<n_ele/n_threads,n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return c;
}
/**********************************************************************************************************************/


/***************************************   Matrix negation   ****************************************/
__global__ void negate_mat(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    c[idx] = a[idx] - b[idx];
}
cu_mat cu_mat::operator-(const cu_mat b)                             // Matrix negation operator
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix negation is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = n_rows*n_cols;
    size_t n_threads = block_dim(n_ele);
    negate_mat<<<n_ele/n_threads,n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return c;
}
/**********************************************************************************************************************/


/***************************************   Matrix power   **************************************/
cu_mat cu_mat::operator^(const unsigned int n)
{
    confirm(n_rows == n_cols,"Error: Matrix has to be square for matrix power(^) to be executed.")
    confirm(n>0,"Error: So far, only non-zero natural numbers are supported for powers.")
    // if (n == 0)
    // {
    //     return eye(n_rows,n_cols);
    // }
    // else if (n == 1)
    if (n==1)
    {
        return *this;
    }
    else
    {
        cu_mat tmp = *this;
        for(int i = 1; i<n; ++i)
        {
            tmp = tmp*(*this);
        }
        return tmp;
    }
}
/***********************************************************************************************************************/