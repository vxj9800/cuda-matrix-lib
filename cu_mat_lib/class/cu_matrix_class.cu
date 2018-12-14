class cu_mat
{
    protected:
    size_t n_rows,n_cols;
    double *p=NULL;
    cu_mat(){}              // Inaccessible default constructor

    public:
        /***** Constructors *****/
        cu_mat(const initializer_list<initializer_list<double>> mat);                    // Single argument constructor
        cu_mat(const cu_mat &to_b_copied);                                              // Copy constructor
        cu_mat(const size_t r, const size_t c);                                                      // Two argument constructor

        /***** Operators *****/
        cu_mat operator()(const size_t r, const size_t c);                                           // Matrix element access
        cu_mat operator()(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end);   // Sub-matrix access
        cu_mat operator*(const cu_mat b);                                             // Matrix multiplication operator
        cu_mat operator+(const cu_mat b);                                             // Matrix addition operator
        cu_mat operator-(const cu_mat b);  

        /***** Member functions *****/
        void get();                                                                         // Print data
        size_t rows();                                                                      // Get number of rows
        size_t cols();                                                                      // Get number of columns

        /***** Supported external functions *****/
        friend cu_mat randn(size_t r, size_t c);                                         // Generate a matrix with normalized random numbers
        friend cu_mat mld(const cu_mat a, const cu_mat b);                               // Matrix left divide operator
        friend cu_mat eye(size_t n);                                           // Generate a square identity matrix
        friend cu_mat eye(size_t r, size_t c);                                           // Generate a non-square identity matrix

        /***** Destructor *****/
        ~cu_mat()                                                                        // Destructor to free the memory
        {
            // cout << "Destructor called." << endl;
            HANDLE_ERROR( cudaFree(p) );
        }
};

#include "constructors.cu"
#include "operators.cu"
#include "member_functions.cu"
#include "friend_functions.cu"