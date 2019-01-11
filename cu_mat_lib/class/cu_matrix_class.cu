#ifndef _CU_MATRIX_CLASS_INCLUDED_
#define _CU_MATRIX_CLASS_INCLUDED_

class cu_mat
{
    protected:
    size_t n_rows=0, n_cols=0;
    double *p=NULL;
    cu_mat(){}              // Inaccessible default constructor
    cu_mat(const size_t r, const size_t c, const double n);  // Two argument constructor with initialization

    public:
        /***** Constructors *****/
        cu_mat(const initializer_list<initializer_list<double>> mat);                    // Single argument constructor with 'double' values
        cu_mat(const initializer_list<initializer_list<cu_mat>> mat);                    // Single argument constructor with 'cu_mat' values
        cu_mat(const double n);                                                          // Single value constructor
        cu_mat(const cu_mat &to_b_copied);                                              // Copy constructor

        /***** Operators *****/
        cu_mat operator()(const size_t r, const size_t c);                                           // Matrix element access
        cu_mat operator()(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end);   // Sub-matrix access
        cu_mat& operator=(const cu_mat b);                                            // Assignment operator to copy 'cu_mat'
        cu_mat& operator=(const double b);                                            // Assignment operator to copy single 'double' value
        cu_mat& operator=(const initializer_list<initializer_list<double>> b);        // Assignment operator to copy 'double' initializer list
        cu_mat& operator=(const initializer_list<initializer_list<cu_mat>> b);        // Assignment operator to copy 'cu_mat' initializer list
        cu_mat operator*(const cu_mat b);                                             // Matrix multiplication operator
        cu_mat operator+(const cu_mat b);                                             // Matrix addition operator
        cu_mat operator-(const cu_mat b);                                             // Matrix negattion operator
        cu_mat operator^(const unsigned int n);                                       // Matrix power operator
        operator double();                                     // Type conversion from cu_mat to double

        /***** Member functions *****/
        cu_mat mult(cu_mat b);                                                              // Element wise multiplication
        void replace(const size_t r, const size_t c, const double n);                       // Replace an element with a 'double' value
        void replace(const size_t r, const size_t c, const cu_mat mat);                     // Replace an element with a 'cu_mat' value
        void replace(const size_t r_begin, const size_t r_end, const size_t c_begin, const size_t c_end, const cu_mat mat);// Replace submatrix with a 'cu_mat' matrix
        void get();                                                                         // Print data
        size_t rows();                                                                      // Get number of rows
        size_t cols();                                                                      // Get number of columns

        /***** Supported external functions *****/
        friend cu_mat randn(const size_t r, const size_t c);                                         // Generate a matrix with normalized random numbers
        friend cu_mat mld(const cu_mat a, const cu_mat b);                                          // Matrix left divide operator
        friend cu_mat eye(const size_t r, const size_t c);                                           // Generate a non-square identity matrix
        friend cu_mat ones(const size_t r, const size_t c);                                          // Matrix with all values 1
        friend cu_mat zeros(const size_t r, const size_t c);                                        // Matrix with all values 0
        friend cu_mat trans(const cu_mat a);                                                        // Transpose of the matrix
        friend cu_mat horzcat(const cu_mat a, const cu_mat b);                                      // Horizontal concatenation of two matrices
        friend cu_mat vertcat(const cu_mat a, const cu_mat b);                                      // Vertical concatenation of two matrices

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

#endif