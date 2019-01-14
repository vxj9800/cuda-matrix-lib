#ifndef _CU_MATRIX_CLASS_OPERATORS_INCLUDED_
#define _CU_MATRIX_CLASS_OPERATORS_INCLUDED_

/**************************************   Access single element of the matrix   *******************************************/
cu_mat cu_mat::operator()(const size_t r, const size_t c)
{
    confirm((r<=n_rows)&&(c<=n_cols),"Error: Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".");
    cu_mat temp(1,1);
    HANDLE_ERROR( cudaMemcpy(temp.p,p+(c-1)*n_rows+r-1,sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
    return temp;
}
/***********************************************************************************************************************/


/**************************************   Access sub-matrix   *******************************************/
__global__ void submat(double* dest, double* src, size_t bias, size_t dest_rows, size_t main_rows_bias, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    dest[idx] = src[bias+idx+idx/dest_rows*main_rows_bias];
}
cu_mat cu_mat::operator()(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end)
{
    confirm((r_end<=n_rows)&&(c_end<=n_cols),"Error: Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
    cu_mat temp(r_end-r_begin+1,c_end-c_begin+1);
    size_t bias = (c_begin-1)*n_rows+r_begin-1;
    size_t main_rows_bias = n_rows-temp.n_rows;
    size_t n_ele = temp.n_rows*temp.n_cols;
    size_t n_threads = block_dim(n_ele);
    submat<<<n_ele/n_threads,n_threads>>>(temp.p,p,bias,temp.n_rows,main_rows_bias,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return temp;
}
/***********************************************************************************************************************/


/***************************************   Assignment operator to copy 'cu_mat'   **************************************/
cu_mat& cu_mat::operator=(const cu_mat b)
{
    if ((n_rows*n_cols)!=(b.n_rows*b.n_cols))
    {
        HANDLE_ERROR( cudaFree(p) );
        HANDLE_ERROR( cudaMalloc((void**)&p, b.n_rows*b.n_cols*sizeof(double)) ); // Allocate memory on GPU.
    }
    n_rows = b.n_rows; n_cols = b.n_cols;
    HANDLE_ERROR( cudaMemcpy(p,b.p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy array from GPU to GPU
    return *this;
}
/***********************************************************************************************************************/


/***************************************   Matrix multiplication   **************************************/
__global__ void const_mat_mult(double *dest, double *src, double *n, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    dest[idx] = (*n)*src[idx];
}
cu_mat cu_mat::operator*(const cu_mat b)
{
    if ((n_rows)*(n_cols)==1)
    {
        cu_mat c(b.n_rows,b.n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_ele/n_threads,n_threads>>>(c.p,b.p,p,n_ele);
        return c;
    }
    else if ((b.n_rows)*(b.n_cols)==1)
    {
        cu_mat c(n_rows,n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_ele/n_threads,n_threads>>>(c.p,p,b.p,n_ele);
        return c;
    }
    else
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
#include "friend_functions.cu"
cu_mat cu_mat::operator^(const unsigned int n)
{
    confirm(n_rows == n_cols,"Error: Matrix has to be square for matrix power(^) to be executed.")
    // confirm(n>=0,"Error: So far, only natural numbers are supported for powers.")
    if (n == 0)
    {
        return eye(n_rows,n_cols);
    }
    else if (n == 1)
    // if (n==1)
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


/***************************************   Type conversion from cu_mat to double   **************************************/
cu_mat::operator double()
{
    confirm((n_rows==1) && (n_cols==1), "Error: Type conversion is only possible in the case of 1x1 matrix.");
    double val;
    // Copy data from GPU to CPU.
    HANDLE_ERROR( cudaMemcpy(&val,p,sizeof(double),cudaMemcpyDeviceToHost) );
    return val;
}

#endif