/**************************************   Single argument constructor   *******************************************/
cu_mat::cu_mat(const initializer_list<initializer_list<double>> mat) : n_rows(mat.size()), n_cols(mat.begin()->size())
// ' -> ' Means:  pointer to an object -> member function. Essentially accessing a member function with the help of a pointer to that object.
{
    // Define number of rows from the array input. Define number of columns from first row of array input
    // Check if the number of elements in each row are same.
    for(int i = 0; i<n_rows; ++i)
    {
        confirm((mat.begin()+i)->size()==n_cols,"Error: Object initialization failed. Number of elements in each row must be same.");
    }

    // Copy input initializer-list to a new 2D array while making it column major.
    double *m = new double[n_rows*n_cols]();    // Allocate space on CPU memory.
    confirm(m,"Error: Memory allocation failed while initializing the object."); // Check proper allocation.                              

    for(int i = 0; i<n_rows; ++i)
    {
        for(int j = 0; j<n_cols; ++j)
        {
            m[j*n_rows+i] = *((mat.begin()+i)->begin()+j);
        }
    }

    HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) ); // Allocate memory on GPU.
    HANDLE_ERROR( cudaMemcpy(p,m,n_rows*n_cols*sizeof(double),cudaMemcpyHostToDevice) ); // Copy array from CPU to GPU
    delete[] m;
}
/***********************************************************************************************************************/


/************************************   Copy constructor   ***********************************************/
cu_mat::cu_mat(const cu_mat &to_b_copied) : n_rows(to_b_copied.n_rows), n_cols(to_b_copied.n_cols)
{
    HANDLE_ERROR( cudaMalloc((void**)&p,n_rows*n_cols*sizeof(double)) ); // Allocate memory on GPU.
    HANDLE_ERROR( cudaMemcpy(p,to_b_copied.p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy array from CPU to GPU
}
/***********************************************************************************************************************/


/************************************   Two argument constructor   ***********************************************/
cu_mat::cu_mat(const size_t r, const size_t c) : n_rows(r), n_cols(c)
{
    HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) );
    HANDLE_ERROR( cudaMemset(p,0,n_rows*n_cols*sizeof(double)) );
}
/***********************************************************************************************************************/