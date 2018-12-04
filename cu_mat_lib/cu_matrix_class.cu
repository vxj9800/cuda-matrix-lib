class cu_matrix
{
    protected:
    size_t rows,cols;
    double *p=NULL;
    cu_matrix(){}              // Inaccessible default constructor

    public:
        /***** Constructors *****/
        cu_matrix(const initializer_list<initializer_list<double>> mat);    // Single argument constructor
        cu_matrix(const cu_matrix &to_b_copied);                            // Copy constructor
        cu_matrix(size_t r, size_t c);                                      // Two argument constructor

        /***** Operators *****/
        cu_matrix operator*(const cu_matrix b);                             // Matrix multiplication operator

        /***** Member functions *****/
        void get();                                                         // Print data

        /***** Supported external functions *****/
        friend cu_matrix randn(size_t r, size_t c);                         // Generate a matrix with normalized random numbers
        friend cu_matrix mld(const cu_matrix a, const cu_matrix b);         // Matrix left divide operator

        /***** Destructor *****/
        ~cu_matrix()                                                        // Destructor to free the memory
        { 
            HANDLE_ERROR( cudaFree(p) );
        }
};



/***********************************************************************************************************************/
cu_matrix::cu_matrix(const cu_matrix &to_b_copied) : rows(to_b_copied.rows), cols(to_b_copied.cols)     // Copy constructor
{
    HANDLE_ERROR( cudaMalloc((void**)&p, rows*cols*sizeof(double)) ); // Allocate memory on GPU.
    HANDLE_ERROR( cudaMemcpy(p,to_b_copied.p,rows*cols*sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy array from CPU to GPU
}
/***********************************************************************************************************************/


/***********************************************************************************************************************/
cu_matrix::cu_matrix(const initializer_list<initializer_list<double>> mat) : rows(mat.size()), cols(mat.begin()->size())    // Single argument constructor
// ' -> ' Means:  pointer to an object -> member function. Essentially accessing a member function with the help of a pointer to that object.
{
    // Define number of rows from the array input. Define number of columns from first row of array input
    // Check if the number of elements in each row are same.
    for(int i = 0; i<rows; ++i)
    {
        if ((mat.begin()+i)->size()!=cols)
        {
            cout << "Error: Number of elements in each row must be same.";
            throw 1;
        }
    }

    // Copy input array to a new matrix while making it column major.
    double *m = new double[rows*cols]();    // Allocate space on CPU memory.
    if (!m)                                 // Check proper allocation.
    {
    cout << "Memory allocation failed\n";
    throw 1;
    }

    for(int i = 0; i<rows; ++i)
    {
        for(int j = 0; j<cols; ++j)
        {
            m[j*rows+i] = *((mat.begin()+i)->begin()+j);
        }
    }

    HANDLE_ERROR( cudaMalloc((void**)&p, rows*cols*sizeof(double)) ); // Allocate memory on GPU.
    HANDLE_ERROR( cudaMemcpy(p,m,rows*cols*sizeof(double),cudaMemcpyHostToDevice) ); // Copy array from CPU to GPU
    delete[] m;
}
/***********************************************************************************************************************/


/***********************************************************************************************************************/
cu_matrix::cu_matrix(size_t r, size_t c) : rows(r), cols(c)                                      // Two argument constructor
{
    HANDLE_ERROR( cudaMalloc((void**)&p, rows*cols*sizeof(double)) );
}
/***********************************************************************************************************************/


/***********************************************************************************************************************/
cu_matrix cu_matrix::operator*(const cu_matrix b)
{
    if (cols != b.rows)
    {
		std::cout << "Error : Inner matrix dimensions must agree.\n";
		throw 1;
    }
    cu_matrix c(rows,b.cols);
    double alf = 1.0, bet = 0;
    cublasHandle_t handle;
	cublasCreate(&handle);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,rows,b.cols,cols,&alf,p,rows,b.p,cols,&bet,c.p,rows);
    cublasDestroy(handle);
    return c;
}
/***********************************************************************************************************************/


/***********************************************************************************************************************/
void cu_matrix::get()   // Print data
{
    double *m = new double[rows*cols]();    // Allocate space on CPU memory.
    if (!m)                                 // Check proper allocation.
    {
    cout << "Memory allocation failed.\n";
    throw 1;
    }

    // Copy data from GPU to CPU.
    HANDLE_ERROR( cudaMemcpy(m,p,rows*cols*sizeof(double),cudaMemcpyDeviceToHost) );
    for(int i = 0; i<rows; ++i)
    {
        for(int j = 0; j<cols; ++j)
        {
            cout<<" "<<m[j*rows+i];
        }
        cout<<endl;
    }
    cout<<endl;
    delete[] m;
}
/***********************************************************************************************************************/