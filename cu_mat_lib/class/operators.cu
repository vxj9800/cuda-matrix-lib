/**************************************   Access single element of the matrix   *******************************************/
cu_mat cu_mat::operator()(const size_t r, const size_t c)
{
    confirm((r<=n_rows)&&(c<=n_cols),"Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
    cu_mat temp(1,1);
    HANDLE_ERROR( cudaMemcpy(temp.p,p+(c-1)*n_rows+r-1,sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
    return temp;
}
/***********************************************************************************************************************/


/**************************************   Access sub-matrix(Working)   *******************************************/
cu_mat cu_mat::operator()(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end)
{
    confirm((r<=n_rows)&&(c<=n_cols),"Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
    cu_mat temp(1,1);
    HANDLE_ERROR( cudaMemcpy(temp.p,p+(c-1)*n_rows+r-1,sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
    return temp;
}
/***********************************************************************************************************************/


/***************************************   Matrix multiplication   **************************************/
cu_mat cu_mat::operator*(const cu_mat b)
{
    confirm(n_cols == b.n_rows,"Error : Matrix multiplication is not possible. Inner matrix dimensions must agree.");
    cu_mat c(n_rows,b.n_cols);
    double alf = 1.0, bet = 0;
    cublasHandle_t handle;
	cublasCreate(&handle);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows);
    cublasDestroy(handle);
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
    return c;
}
/**********************************************************************************************************************/