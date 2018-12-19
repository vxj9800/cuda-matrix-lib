#ifndef _CU_MATRIX_CLASS_MEMBER_FUNCTIONS_INCLUDED_
#define _CU_MATRIX_CLASS_MEMBER_FUNCTIONS_INCLUDED_

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