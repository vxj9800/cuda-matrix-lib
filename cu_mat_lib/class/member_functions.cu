#ifndef _CU_MATRIX_CLASS_MEMBER_FUNCTIONS_INCLUDED_
#define _CU_MATRIX_CLASS_MEMBER_FUNCTIONS_INCLUDED_

/************************************   Element wise multiplication   ***********************************************/
__global__ void mat_multiplication(double* a, double* b, double* c, size_t n_ele)
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
    addition<<<n_ele/n_threads,n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return c;
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


/***************************************   Get number of rows   *****************************************/
size_t cu_mat::rows(){return n_rows;}
/***********************************************************************************************************************/


/***************************************   Get number of columns   *****************************************/
size_t cu_mat::cols(){return n_cols;}
/***********************************************************************************************************************/

#endif