#ifndef _CU_MATRIX_CLASS_OPERATORS_INCLUDED_
#define _CU_MATRIX_CLASS_OPERATORS_INCLUDED_

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
__global__ void submat(double* dest, double* src, size_t bias, size_t dest_rows, size_t main_rows_bias, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    dest[idx] = src[bias+idx+idx/dest_rows*main_rows_bias];
}
cu_mat cu_mat::operator()(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end)
{
    confirm((r_end<=n_rows)&&(c_end<=n_cols),"Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
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


/***************************************   Assignment operator to copy single 'double' value   **************************************/
cu_mat& cu_mat::operator=(const double b)
{
    if ((n_rows*n_cols)!=1)
    {
        HANDLE_ERROR( cudaFree(p) );
        HANDLE_ERROR( cudaMalloc((void**)&p, sizeof(double)) ); // Allocate memory on GPU.
    }
    n_rows = 1; n_cols = 1;
    HANDLE_ERROR( cudaMemcpy(p,&b,n_rows*n_cols*sizeof(double),cudaMemcpyHostToDevice) ); // Copy array from GPU to GPU
    return *this;
}
/***********************************************************************************************************************/


/***************************************   Assignment operator to copy 'double' initializer list   **************************************/
cu_mat& cu_mat::operator=(const initializer_list<initializer_list<double>> b)
{
    for(int i = 0; i<b.size(); ++i)
    {
        confirm((b.begin()+i)->size()==(b.begin()->size()),"Error: Object initialization failed. Number of elements in each row must be same.");
    }

    double *m = new double[b.size()*(b.begin()->size())]();    // Allocate space on CPU memory.
    confirm(m,"Error: Memory allocation failed while initializing the object."); // Check proper allocation.                              

    for(int i = 0; i<b.size(); ++i)
    {
        for(int j = 0; j<(b.begin()->size()); ++j)
        {
            m[j*b.size()+i] = *((b.begin()+i)->begin()+j);
        }
    }
    if ((n_rows*n_cols)!=b.size()*(b.begin()->size()))
    {
        HANDLE_ERROR( cudaFree(p) );
        HANDLE_ERROR( cudaMalloc((void**)&p, b.size()*(b.begin()->size())*sizeof(double)) ); // Allocate memory on GPU.
    }
    n_rows = b.size(); n_cols = (b.begin()->size());
    HANDLE_ERROR( cudaMemcpy(p,m,n_rows*n_cols*sizeof(double),cudaMemcpyHostToDevice) );
    return *this;
}
/***********************************************************************************************************************/


/***************************************   Assignment operator to copy 'cu_mat' initializer list   **************************************/
cu_mat& cu_mat::operator=(const initializer_list<initializer_list<cu_mat>> mat)
{
    // Calculate total number of columns
    size_t mat_rows = 0, mat_cols = 0;
    for(int i = 0; i<mat.begin()->size(); ++i)
        mat_cols += ((mat.begin())->begin()+i)->n_cols;

    // Check equal number of rows for horizontal concatenation and calculate total number of rows.
    for(int i = 0; i<mat.size(); ++i)
    {
        size_t cols = ((mat.begin()+i)->begin())->n_cols;
        for(int j = 0; j<(mat.begin()+i)->size()-1; ++j)
        {
            confirm(((mat.begin()+i)->begin()+j)->n_rows==((mat.begin()+i)->begin()+j+1)->n_rows,"Error: Dimensions of arrays being horizontally concatenated are not consistent.");
            cols += ((mat.begin()+i)->begin()+j+1)->n_cols;
        }
        confirm(cols == mat_cols,"Error: Dimensions of arrays being vertically concatenated are not consistent.")
        mat_rows += ((mat.begin()+i)->begin())->n_rows;
    }

    if ((n_rows*n_cols)!=mat_rows*mat_cols)
    {
        HANDLE_ERROR( cudaFree(p) );
        HANDLE_ERROR( cudaMalloc((void**)&p, mat_rows*mat_cols*sizeof(double)) ); // Allocate memory on GPU.
    }
    n_rows = mat_rows; n_cols = mat_rows;
    size_t bias, src_rows, src_cols;
    size_t main_rows_bias, n_ele, n_threads;
    size_t r_sum = 0, c_sum = 0;
    for(int i = 0; i<mat.size(); ++i){
        for(int j = 0; j<(mat.begin()+i)->size(); ++j){
            bias = c_sum*n_rows+r_sum; src_rows = ((mat.begin()+i)->begin()+j)->n_rows; src_cols = ((mat.begin()+i)->begin()+j)->n_cols;
            main_rows_bias = n_rows-src_rows; n_ele = src_rows*src_cols; n_threads = block_dim(n_ele); c_sum += src_cols;
            copymat<<<n_ele/n_threads,n_threads>>>(p,((mat.begin()+i)->begin()+j)->p,bias,src_rows,main_rows_bias,n_ele);
            HANDLE_ERROR( cudaPeekAtLastError() );
        }
        r_sum += src_rows; c_sum = 0;
    }
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