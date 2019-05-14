#ifndef _CU_MATRIX_CLASS_INCLUDED_
#define _CU_MATRIX_CLASS_INCLUDED_

class cu_mat
{
    protected:
    size_t n_rows=0, n_cols=0;
    bool del = 1;
    // double *p=NULL;
    cu_mat(){}              // Inaccessible default constructor
    cu_mat(const size_t &r, const size_t &c, const double &n);  // Two argument constructor with initialization
    void init(const size_t &r, const size_t &c); // Two argument memory allocation with initialization

    public:
        double *p=NULL;
        /***** Constructors *****/
        cu_mat(const initializer_list<initializer_list<double>> &mat);                              // Single argument constructor with 'double' values
        cu_mat(const initializer_list<initializer_list<cu_mat>> &mat);                              // Single argument constructor with 'cu_mat' values
        cu_mat(const double &n);                                                                    // Single value constructor
        cu_mat(const cu_mat &to_b_copied);                                                          // Copy constructor

        /***** Operators *****/ // Add an ultimate '()' operator.
        cu_mat operator()(const cu_mat rows, const cu_mat cols);                                    // Sub-matrix access with 'cu_mat'
        cu_mat operator()(const size_t &idx);                                                        // Matrix element access based on index
        cu_mat operator()(const size_t &r, const size_t &c);                                          // Matrix element access
        cu_mat operator()(const size_t &r_begin, const size_t &r_end, const size_t &c_begin, const size_t &c_end);   // Sub-matrix access
        cu_mat& operator=(const cu_mat &b);                                                          // Assignment operator to copy 'cu_mat'
        cu_mat operator*(const cu_mat &b);                                                           // Matrix multiplication operator
        cu_mat operator/(const cu_mat &b);                                                           // Matrix right divide operator
        cu_mat operator+(const cu_mat &b);                                                           // Matrix addition operator
        cu_mat operator-(const cu_mat &b);                                                           // Matrix negattion operator
        cu_mat operator^(const unsigned int &n);                                                     // Matrix power operator
        cu_mat operator>(const cu_mat &b);                                                           // Greather than operator
        cu_mat operator<(const cu_mat &b);                                                           // Smaller than operator
        cu_mat operator>=(const cu_mat &b);                                                          // Greather than operator or equal to
        cu_mat operator<=(const cu_mat &b);                                                          // Smaller than operator or equal to
        cu_mat operator!();                                                                         // NOT operator
        cu_mat operator==(const cu_mat &b);                                                          // Comparison equal to operator
        cu_mat operator!=(const cu_mat &b);                                                          // Comparison not equal to operator
        cu_mat operator&&(const cu_mat &b);                                                          // Logical 'AND' operator
        cu_mat operator||(const cu_mat &b);                                                          // Logical 'OR' operator
        explicit operator double();                                                                 // Type conversion from cu_mat to double

        /***** Member functions *****/ // Add an ultimate replace function
        cu_mat div(const cu_mat &b);                                                                       // Element wise division
        cu_mat mult(const cu_mat &b);                                                                      // Element wise multiplication
        cu_mat pow(const double &n);                                                                 // Element wise power
        void replace(const size_t &r, const size_t &c, const cu_mat &n);                             // Replace an element with a 'cu_mat' value
        void replace(const size_t &r_begin, const size_t &r_end, const size_t &c_begin, const size_t &c_end, const cu_mat &n);// Replace submatrix with a 'cu_mat' matrix
        void get();                                                                                 // Print matrix data
        void print(ofstream &print);                                                                // Print matrix to a file
        size_t rows();                                                                              // Get number of rows
        size_t cols();                                                                              // Get number of columns

        /***** Supported external functions *****/
        friend cu_mat randn(const size_t &r, const size_t &c);                                        // Generate a matrix with normalized random numbers
        friend cu_mat mld(const cu_mat &a, const cu_mat &b);                                          // Matrix left divide operator
        friend cu_mat eye(const size_t &r, const size_t &c);                                          // Generate a non-square identity matrix
        friend cu_mat ones(const size_t &r, const size_t &c);                                         // Matrix with all values 1
        friend cu_mat zeros(const size_t &r, const size_t &c);                                        // Matrix with all values 0
        friend cu_mat trans(const cu_mat &a);                                                        // Transpose of the matrix
        friend cu_mat horzcat(const cu_mat &a, const cu_mat &b);                                      // Horizontal concatenation of two matrices
        friend cu_mat vertcat(const cu_mat &a, const cu_mat &b);                                      // Vertical concatenation of two matrices
        friend cu_mat stepspace(const double &i, const double &step, const double &f);                 // MATLAB colon operator
        friend cu_mat sum(const cu_mat &a);                                                          // Sum of the elements of the matrix (Working on it)
        friend cu_mat norm(cu_mat &a, const double &p);                                                     // Norm of the matrix

        /***** Supported math functions *****/
        friend cu_mat acos(const cu_mat &a);                                                         // Calculate arc cosine of each element
        friend cu_mat acosh(const cu_mat &a);                                                        // Calculate arc hyperbolic cosine of each element
        friend cu_mat asin(const cu_mat &a);                                                         // Calculate arc sine of each element
        friend cu_mat asinh(const cu_mat &a);                                                        // Calculate arc hyperbolic sine of each element
        friend cu_mat atan(const cu_mat &a);                                                         // Calculate arc tangent of each element
        friend cu_mat atan2(const cu_mat &y, const cu_mat &x);                                        // Calculate four quadrant arc tangent of each element
        friend cu_mat atanh(const cu_mat &a);                                                        // Calculate hyperbolic arc tangent of each element
        friend cu_mat ceil(const cu_mat &a);                                                         // Calculate ceiling of each element
        friend cu_mat cos(const cu_mat &a);                                                          // Calculate cosine of each element
        friend cu_mat cosh(const cu_mat &a);                                                         // Calculate hyperbolic cosine of each element
        friend cu_mat exp(const cu_mat &a);                                                          // Calculate e^x of each element
        friend cu_mat exp10(const cu_mat &a);                                                        // Calculate 10^x of each element
        friend cu_mat exp2(const cu_mat &a);                                                         // Calculate 2^x of each element
        friend cu_mat abs(const cu_mat &a);                                                          // Calculate absolute of each element
        friend cu_mat floor(const cu_mat &a);                                                        // Calculate floor value of each element
        friend cu_mat mod(const cu_mat &a, const cu_mat &b);                                          // Calculate modulo of a/b
        friend cu_mat isfinite(const cu_mat &a);                                                     // Check if each element of matrix is not inf or nan
        friend cu_mat isinf(const cu_mat &a);                                                        // Check if each element is inf
        friend cu_mat isnan(const cu_mat &a);                                                        // Check if each element is nan
        friend bool isempty(const cu_mat &a);                                                        // Check if 'cu_mat' object is empty
        friend bool isscalar(const cu_mat &a);                                                       // Check if 'cu_mat' object is scalar
        friend bool isvector(const cu_mat &a);                                                       // Check if 'cu_mat' object is vector
        friend bool ismatrix(const cu_mat &a);                                                       // Check if 'cu_mat' object is matrix
        friend cu_mat log(const cu_mat &a);                                                          // Calculate log_e of each element
        friend cu_mat log10(const cu_mat &a);                                                        // Calculate log_10 of each element
        friend cu_mat log2(const cu_mat &a);                                                         // Calculate log_2 of each element
        friend cu_mat rem(const cu_mat &a, const cu_mat &b);                                          // Calculate remainder of each element of a/b
        friend cu_mat round(const cu_mat &a);                                                        // Calculate rounded value of each element
        friend cu_mat sign(const cu_mat &a);                                                         // Calculate sign of each element
        friend cu_mat sin(const cu_mat &a);                                                          // Calculate sine of each element
        friend cu_mat sinh(const cu_mat &a);                                                         // Calculate hyperbolic sine of each element
        friend cu_mat sqrt(const cu_mat &a);                                                         // Calculate square root of each element
        friend cu_mat tan(const cu_mat &a);                                                          // Calculate tangent of each element
        friend cu_mat tanh(const cu_mat &a);                                                         // Calculate hyperbolic tangent of each element
        friend cu_mat cos(const cu_mat &a);                                                          // Calculate cosine of each element

        /***** Destructor *****/
        ~cu_mat()                                                                                   // Destructor to free the memory
        {
            // cout << "Destructor called." << endl;
            if (del==1)
            HANDLE_ERROR( cudaFree(p) );
        }
};

#include "constructors.cu"
#include "math_functions.cu"
#include "operators.cu"
#include "member_functions.cu"
#include "friend_functions.cu"

#endif