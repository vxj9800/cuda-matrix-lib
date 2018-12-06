class cu_mat
{
    protected:
    size_t n_rows,n_cols;
    double *p=NULL;
    cu_mat(){}              // Inaccessible default constructor

    public:
        /***** Constructors *****/
        cu_mat(const initializer_list<initializer_list<double>> mat);                    // Single argument constructor
        cu_mat(const cu_mat &to_b_copied);                                            // Copy constructor
        cu_mat(size_t r, size_t c);                                                      // Two argument constructor

        /***** Operators *****/
        cu_mat operator()(size_t r, size_t c);                                           // Matrix element access
        cu_mat operator()(size_t r_begin, size_t r_end, size_t c_begin, size_t c_end);   // Sub-matrix access
        cu_mat operator*(const cu_mat b);                                             // Matrix multiplication operator
        cu_mat operator+(const cu_mat b);                                             // Matrix addition operator

        /***** Member functions *****/
        void get();                                                                         // Print data
        size_t rows();                                                                      // Get number of rows
        size_t cols();                                                                      // Get number of columns

        /***** Supported external functions *****/
        friend cu_mat randn(size_t r, size_t c);                                         // Generate a matrix with normalized random numbers
        friend cu_mat mld(const cu_mat a, const cu_mat b);                         // Matrix left divide operator

        /***** Destructor *****/
        ~cu_mat()                                                                        // Destructor to free the memory
        {
            // cout << "Destructor called." << endl;
            HANDLE_ERROR( cudaFree(p) );
        }
};



/***********************************************************************************************************************/
cu_mat::cu_mat(const cu_mat &to_b_copied) : n_rows(to_b_copied.n_rows), n_cols(to_b_copied.n_cols)     // Copy constructor
{
    HANDLE_ERROR( cudaMalloc((void**)&p,n_rows*n_cols*sizeof(double)) ); // Allocate memory on GPU.
    HANDLE_ERROR( cudaMemcpy(p,to_b_copied.p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy array from CPU to GPU
}
/***********************************************************************************************************************/


/***********************************************************************************************************************/
cu_mat::cu_mat(const initializer_list<initializer_list<double>> mat) : n_rows(mat.size()), n_cols(mat.begin()->size())    // Single argument constructor
// ' -> ' Means:  pointer to an object -> member function. Essentially accessing a member function with the help of a pointer to that object.
{
    // Define number of rows from the array input. Define number of columns from first row of array input
    // Check if the number of elements in each row are same.
    for(int i = 0; i<n_rows; ++i)
    {
        confirm((mat.begin()+i)->size()==n_cols,"Error: Object initialization failed. Number of elements in each row must be same.");
    }

    // Copy input array to a new matrix while making it column major.
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


/***********************************************************************************************************************/
cu_mat::cu_mat(size_t r, size_t c) : n_rows(r), n_cols(c)                                      // Two argument constructor
{
    HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) );
}
/***********************************************************************************************************************/


/***********************************************************************************************************************/
cu_mat cu_mat::operator()(size_t r, size_t c)                                           // Matrix element access
{
    confirm((r<=n_rows)&&(c<=n_cols),"Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
    cu_mat temp(1,1);
    HANDLE_ERROR( cudaMemcpy(temp.p,p+(c-1)*n_rows+r-1,sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
    return temp;
}
/***********************************************************************************************************************/


/***********************************************************************************************************************/
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


/**********************************************************************************************************************/
// cu_mat cu_mat::operator+(const cu_mat b);                             // Matrix addition operator
// {
//     confirm((n_rows == b.n_rows) && (n_cols != b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");

// }
/**********************************************************************************************************************/


/***********************************************************************************************************************/
void cu_mat::get()   // Print data
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


/***********************************************************************************************************************/
size_t cu_mat::rows(){return n_rows;}                                                      // Get number of rows
/***********************************************************************************************************************/


/***********************************************************************************************************************/
size_t cu_mat::cols(){return n_cols;}                                                      // Get number of columns
/***********************************************************************************************************************/