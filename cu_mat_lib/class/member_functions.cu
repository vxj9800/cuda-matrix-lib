#ifndef _CU_MATRIX_CLASS_MEMBER_FUNCTIONS_INCLUDED_
#define _CU_MATRIX_CLASS_MEMBER_FUNCTIONS_INCLUDED_

/************************************   Element wise division   ***********************************************/
__global__ void elem_div(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    c[idx] = a[idx] / b[idx];
}
cu_mat cu_mat::div(cu_mat b)
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = n_rows*n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_div<<<n_ele/n_threads,n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return c;
}
/***********************************************************************************************************************/


/************************************   Element wise multiplication   ***********************************************/
__global__ void elem_mult(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    c[idx] = a[idx] * b[idx];
}
cu_mat cu_mat::mult(cu_mat b)
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = n_rows*n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_mult<<<n_ele/n_threads,n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return c;
}
/***********************************************************************************************************************/


/************************************   Element wise power   ***********************************************/
__global__ void elem_power(double* dest, double* src, double n, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    dest[idx] = pow(src[idx],n);
}
cu_mat cu_mat::pow(double n)
{
    cu_mat c(n_rows,n_cols);
    size_t n_ele = n_rows*n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_power<<<n_ele/n_threads,n_threads>>>(c.p,p,n,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return c;
}
/***********************************************************************************************************************/


/************************************   Replace an element with a 'cu_mat' value   ***********************************************/
void cu_mat::replace(const size_t r, const size_t c, const cu_mat n)
{
    confirm((n.n_rows==1) && (n.n_cols==1),"Error: Value being replaced with has to be scalar.");
    size_t bias = c*n_rows+r, src_rows = 1, src_cols = 1;
    size_t main_rows_bias = n_rows-src_rows, n_ele = src_rows*src_cols, n_threads = block_dim(n_ele);
    copymat<<<n_ele/n_threads,n_threads>>>(p,n.p,bias,src_rows,main_rows_bias,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
}
/***********************************************************************************************************************/


/************************************   Replace submatrix with a 'cu_mat' matrix   ***********************************************/
void cu_mat::replace(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end, const cu_mat n)
{
    confirm((r_end<=n_rows) && (c_end<=n_cols),"Error: Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".");
    confirm((n.n_rows==r_end-r_begin+1) && (n.n_cols==c_end-c_begin+1),"Error: Unable to replace the data due to size mismatch.");
    size_t bias = (c_begin-1)*n_rows+r_begin-1, src_rows = n.n_rows, src_cols = n.n_cols;
    size_t main_rows_bias = n_rows-src_rows, n_ele = src_rows*src_cols, n_threads = block_dim(n_ele);
    copymat<<<n_ele/n_threads,n_threads>>>(p,n.p,bias,src_rows,main_rows_bias,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
}
/***********************************************************************************************************************/


/************************************   Print matrix data   ***********************************************/
void cu_mat::get()
{
    double *m = new double[n_rows*n_cols]();    // Allocate space on CPU memory.
    confirm(m,"Error: Memory allocation failed in 'get()'.") // Check proper allocation.

    // Copy data from GPU to CPU.
    HANDLE_ERROR( cudaMemcpy(m,p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToHost) );
    for(int i = 0; i<n_rows; ++i)
    {
        for(int j = 0; j<n_cols; ++j)
        {
            cout<<" "<<m[j*n_rows+i];
        }
        cout<<endl;
    }
    cout<<endl;
    delete[] m;
}
/***********************************************************************************************************************/


/************************************   Print matrix to a file   ***********************************************/
void cu_mat::print(ofstream &print)
{
    double *m = new double[n_rows*n_cols]();    // Allocate space on CPU memory.
    confirm(m,"Error: Memory allocation failed in 'get()'.") // Check proper allocation.

    // Copy data from GPU to CPU.
    HANDLE_ERROR( cudaMemcpy(m,p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToHost) );

    // Print the matrix
    print << scientific << setprecision(8);
    for(int i = 0; i<n_rows; ++i)
    {
        print << " ";
        for(int j = 0; j<n_cols; ++j)
        {
            print << " " << m[j*n_rows+i] << " ";
        }
        print << endl;
    }
    delete[] m;
}
/***********************************************************************************************************************/


/***************************************   Get number of rows   *****************************************/
size_t cu_mat::rows(){return n_rows;}
/***********************************************************************************************************************/


/***************************************   Get number of columns   *****************************************/
size_t cu_mat::cols(){return n_cols;}
/***********************************************************************************************************************/

#endif