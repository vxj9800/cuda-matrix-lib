#include <cuda_runtime.h>
// #include <helper_cuda.h>
#include "error_check.cu"

class cu_matrix
{
    protected:
    size_t rows,cols;
    double *p=NULL;
    cu_matrix(){}              // Inaccessible default constructor

    public:    
    cu_matrix(const initializer_list<initializer_list<double>> mat);    // Single argument constructor
    void get();                                                         // Print data
    ~cu_matrix()                                                        // Destructor to free the memory
    { 
        HANDLE_ERROR( cudaFree(p) );
    }
};





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

    // Copy array from CPU to GPU
    HANDLE_ERROR( cudaMalloc((void**)&p, rows*cols*sizeof(double)) );
    HANDLE_ERROR( cudaMemcpy(p,m,rows*cols*sizeof(double),cudaMemcpyHostToDevice) );
    delete[] m;
}





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
    delete[] m;
}