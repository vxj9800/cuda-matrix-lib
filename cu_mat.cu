#include "cu_mat.hcu"
#define n_blocks(n_ele,n_threads) (n_ele%n_threads != 0) ? (n_ele/n_threads)+1 : n_ele/n_threads 
/*
 * cu_mat.cu
 *
 *  Created on: Jun 5, 2019
 *      Author: vatsal
 */


/*
 * cumat.cu
 *
 *  Created on: Jun 3, 2019
 *      Author: vatsal
 */


// $$$$$$\                                  $$\                                     $$\
//$$  __$$\                                 $$ |                                    $$ |
//$$ /  \__| $$$$$$\  $$$$$$$\   $$$$$$$\ $$$$$$\    $$$$$$\  $$\   $$\  $$$$$$$\ $$$$$$\    $$$$$$\   $$$$$$\   $$$$$$$\
//$$ |      $$  __$$\ $$  __$$\ $$  _____|\_$$  _|  $$  __$$\ $$ |  $$ |$$  _____|\_$$  _|  $$  __$$\ $$  __$$\ $$  _____|
//$$ |      $$ /  $$ |$$ |  $$ |\$$$$$$\    $$ |    $$ |  \__|$$ |  $$ |$$ /        $$ |    $$ /  $$ |$$ |  \__|\$$$$$$\
//$$ |  $$\ $$ |  $$ |$$ |  $$ | \____$$\   $$ |$$\ $$ |      $$ |  $$ |$$ |        $$ |$$\ $$ |  $$ |$$ |       \____$$\
//\$$$$$$  |\$$$$$$  |$$ |  $$ |$$$$$$$  |  \$$$$  |$$ |      \$$$$$$  |\$$$$$$$\   \$$$$  |\$$$$$$  |$$ |      $$$$$$$  |
// \______/  \______/ \__|  \__|\_______/    \____/ \__|       \______/  \_______|   \____/  \______/ \__|      \_______/
//	Constructors
/**************************************   Single argument constructor with 'double' values   *******************************************/
cu_mat::cu_mat(const std::initializer_list<std::initializer_list<double>> &mat) : n_rows(mat.size()), n_cols(mat.begin()->size())
{
    // ' -> ' Means:  pointer to an object -> member function. Essentially accessing a member function with the help of a pointer to that object.
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


/**************************************   Single argument constructor with 'cu_mat' values   *******************************************/
__global__ void copymat(double* dest, double* src, size_t bias, size_t src_rows, size_t main_rows_bias, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[bias+idx+idx/src_rows*main_rows_bias] = src[idx];
}
cu_mat::cu_mat(const std::initializer_list<std::initializer_list<cu_mat>> &mat)
{
    // Calculate total number of columns
    for(int i = 0; i<mat.begin()->size(); ++i)
        n_cols += ((mat.begin())->begin()+i)->n_cols;

    // Check equal number of rows for horizontal concatenation and calculate total number of rows.
    for(int i = 0; i<mat.size(); ++i)
    {
        size_t cols = ((mat.begin()+i)->begin())->n_cols;
        for(int j = 0; j<(mat.begin()+i)->size()-1; ++j)
        {
            confirm(((mat.begin()+i)->begin()+j)->n_rows==((mat.begin()+i)->begin()+j+1)->n_rows,"Error: Dimensions of arrays being horizontally concatenated are not consistent.");
            cols += ((mat.begin()+i)->begin()+j+1)->n_cols;
        }
        confirm(cols == n_cols,"Error: Dimensions of arrays being vertically concatenated are not consistent.")
        n_rows += ((mat.begin()+i)->begin())->n_rows;
    }

    // Allocate memory and copy data
    if ((n_rows>0)&&(n_cols>0))
    {
        HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) ); // Allocate memory on GPU.
        size_t bias, src_rows, src_cols;
        size_t main_rows_bias, n_ele, n_threads;
        size_t r_sum = 0, c_sum = 0;
        for(int i = 0; i<mat.size(); ++i){
            for(int j = 0; j<(mat.begin()+i)->size(); ++j){
                bias = c_sum*n_rows+r_sum; src_rows = ((mat.begin()+i)->begin()+j)->n_rows; src_cols = ((mat.begin()+i)->begin()+j)->n_cols;
                main_rows_bias = n_rows-src_rows; n_ele = src_rows*src_cols; n_threads = block_dim(n_ele); c_sum += src_cols;
                copymat<<<n_blocks(n_ele,n_threads),n_threads>>>(p,((mat.begin()+i)->begin()+j)->p,bias,src_rows,main_rows_bias,n_ele);
                HANDLE_ERROR( cudaPeekAtLastError() );
            }
            r_sum += src_rows; c_sum = 0;
        }
    }
}
/***********************************************************************************************************************/


/************************************   Single value constructor   ***********************************************/
cu_mat::cu_mat(const double &n) : n_rows(1), n_cols(1)
{
    HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) ); // Allocate memory on GPU.
//    std::cout << p << std::endl;
    HANDLE_ERROR( cudaMemcpy(p,&n,n_rows*n_cols*sizeof(double),cudaMemcpyHostToDevice) ); // Copy array from CPU to GPU
}
/***********************************************************************************************************************/


/************************************   Copy constructor   ***********************************************/
cu_mat::cu_mat(const cu_mat &to_b_copied) : n_rows(to_b_copied.n_rows), n_cols(to_b_copied.n_cols)
{
//    std::cout << "Copy constructor called." << std::endl;
    if ((n_rows>0)&&(n_cols>0))
    {
		HANDLE_ERROR( cudaFree(p) );
		HANDLE_ERROR( cudaMalloc((void**)&p,n_rows*n_cols*sizeof(double)) ); // Allocate memory on GPU.
		HANDLE_ERROR( cudaMemcpy(p,to_b_copied.p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy array from CPU to GPU
    }
}
/***********************************************************************************************************************/


/************************************   Two argument constructor with initialization   ***********************************************/
__global__ void set_data(double* p, const double n, const double n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        p[idx] = n;
}
cu_mat::cu_mat(const size_t &r, const size_t &c, const double &n=0) : n_rows(r), n_cols(c)
{
    if ((n_rows>0)&&(n_cols>0))
    {
        HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) );
        if (n!=0)
        {
            size_t n_ele = n_rows*n_cols;
            size_t n_threads = block_dim(n_ele);
            set_data<<<n_blocks(n_ele,n_threads),n_threads>>>(p,n,n_ele);
            HANDLE_ERROR( cudaPeekAtLastError() );
        }
        else
        {
            HANDLE_ERROR( cudaMemset(p,0,n_rows*n_cols*sizeof(double)) );
        }
    }
}
/***********************************************************************************************************************/


/************************************   Move constructor   ***********************************************/
cu_mat::cu_mat(cu_mat&& to_b_moved)
{
//    std::cout << "Move constructor called." << std::endl;
    size_t nrows = to_b_moved.n_rows, ncols = to_b_moved.n_cols; double *ptr = to_b_moved.p;
    to_b_moved.n_rows = 0; to_b_moved.n_cols = 0; to_b_moved.p = NULL;
    n_rows = nrows; n_cols = ncols; p = ptr;
}
/***********************************************************************************************************************/


















































//   $$$$$$\                                          $$\
//  $$  __$$\                                         $$ |
//  $$ /  $$ | $$$$$$\   $$$$$$\   $$$$$$\  $$$$$$\ $$$$$$\    $$$$$$\   $$$$$$\   $$$$$$$\
//  $$ |  $$ |$$  __$$\ $$  __$$\ $$  __$$\ \____$$\\_$$  _|  $$  __$$\ $$  __$$\ $$  _____|
//  $$ |  $$ |$$ /  $$ |$$$$$$$$ |$$ |  \__|$$$$$$$ | $$ |    $$ /  $$ |$$ |  \__|\$$$$$$\
//  $$ |  $$ |$$ |  $$ |$$   ____|$$ |     $$  __$$ | $$ |$$\ $$ |  $$ |$$ |       \____$$\
//   $$$$$$  |$$$$$$$  |\$$$$$$$\ $$ |     \$$$$$$$ | \$$$$  |\$$$$$$  |$$ |      $$$$$$$  |
//   \______/ $$  ____/  \_______|\__|      \_______|  \____/  \______/ \__|      \_______/
//            $$ |
//            $$ |
//            \__|
//	Operators
/**************************************   Sub-matrix access with 'cu_mat'   *******************************************/
// __global__ void check_integer()
// cu_mat cu_mat::operator()(const cu_mat rows, const cu_mat cols)
// {
//     confirm((rows.n_rows==1) || (rows.n_cols==1), "Error: 'rows' has to be a vector.");
//     confirm((cols.n_rows==1) || (cols.n_cols==1), "Error: 'rows' has to be a vector.");
//     confirm(idx > 0, "Indexing starts from 1 for this library.")
//     confirm(idx <= n_rows*n_cols,"Error: Index exceeds matrix bounds. Matrix has " << n_rows*n_cols << "elements in it.");
//     cu_mat temp(1,1);
//     HANDLE_ERROR( cudaMemcpy(temp.p,p+(idx-1),sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
//     return temp;
// }
/***********************************************************************************************************************/


/**************************************   Matrix element access based on index   *******************************************/
cu_mat cu_mat::operator()(const size_t &idx) const
{
    confirm(idx > 0, "Indexing starts from 1 for this library.")
    confirm(idx <= n_rows*n_cols,"Error: Index exceeds matrix bounds. Matrix has " << n_rows*n_cols << " elements in it.");
    cu_mat temp(1,1);
    HANDLE_ERROR( cudaMemcpy(temp.p,p+(idx-1),sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
    return std::move(temp);
}
/***********************************************************************************************************************/


/**************************************   Access single element of the matrix   *******************************************/
cu_mat cu_mat::operator()(const size_t &r, const size_t &c) const
{
    confirm((r>0)&&(c>0), "Indexing starts from 1 for this library.")
    confirm((r<=n_rows)&&(c<=n_cols),"Error: Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".");
    cu_mat temp(1,1);
    HANDLE_ERROR( cudaMemcpy(temp.p,p+(c-1)*n_rows+r-1,sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy value from GPU to GPU
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(temp);
}
/***********************************************************************************************************************/


/**************************************   Access sub-matrix   *******************************************/
__global__ void submat(double* dest, double* src, size_t bias, size_t dest_rows, size_t main_rows_bias, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = src[bias+idx+idx/dest_rows*main_rows_bias];
}
cu_mat cu_mat::operator()(const size_t &r_begin, const size_t &r_end, const size_t &c_begin, const size_t &c_end) const
{
    confirm((r_begin>0)&&(c_begin>0), "Indexing starts from 1 for this library.")
    confirm((r_end<=n_rows)&&(c_end<=n_cols),"Error: Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".")
    cu_mat temp(r_end-r_begin+1,c_end-c_begin+1);
    size_t bias = (c_begin-1)*n_rows+r_begin-1;
    size_t main_rows_bias = n_rows-temp.n_rows;
    size_t n_ele = temp.n_rows*temp.n_cols;
    size_t n_threads = block_dim(n_ele);
    submat<<<n_blocks(n_ele,n_threads),n_threads>>>(temp.p,p,bias,temp.n_rows,main_rows_bias,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(temp);
}
/***********************************************************************************************************************/


/***************************************   Assignment operator to copy 'cu_mat'   **************************************/
cu_mat& cu_mat::operator=(const cu_mat &b)
{
//	std::cout << "Copy assignment operator called." << std::endl;
	if ((n_rows*n_cols)!=(b.n_rows*b.n_cols))
	{
		HANDLE_ERROR( cudaFree(p) );
		HANDLE_ERROR( cudaMalloc((void**)&p, b.n_rows*b.n_cols*sizeof(double)) ); // Allocate memory on GPU.
	}
	n_rows = b.n_rows; n_cols = b.n_cols;
	HANDLE_ERROR( cudaMemcpy(p,b.p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToDevice) ); // Copy array from GPU to GPU
    return *this;
}
/***********************************************************************************************************************/


/***************************************   Assignment operator to move 'cu_mat'   **************************************/
cu_mat& cu_mat::operator=(cu_mat &&b)
{
//    std::cout << "Move assignment operator called." << std::endl;
    n_rows = b.n_rows; b.n_rows = 0;
    n_cols = b.n_cols; b.n_cols = 0;
    HANDLE_ERROR( cudaFree(p) );
    p = b.p; b.p = NULL;
    return *this;
}
/***********************************************************************************************************************/


/***************************************   Matrix multiplication   **************************************/
__global__ void const_mat_mult(double *dest, double *src, double *n, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = (*n)*src[idx];
}
cu_mat cu_mat::operator*(const cu_mat &b) const &
{
    if (isscalar(*this))
    {
        cu_mat c(b.n_rows,b.n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
    	cu_mat c(n_rows,n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(n_cols == b.n_rows,"Error : Matrix multiplication is not possible. Inner matrix dimensions must agree.");
        cu_mat c(n_rows,b.n_cols);
        HANDLE_ERROR( cudaMalloc((void**)&c.p,c.n_rows*c.n_cols*sizeof(double)) ); // Allocate memory on GPU.
        double alf = 1.0, bet = 0;
        cublasHandle_t handle;
        HANDLE_ERROR( cublasCreate(&handle) );
        HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        HANDLE_ERROR( cublasDestroy(handle) );
        return std::move(c);
    }
}
cu_mat cu_mat::operator*(cu_mat &&b) const &
{
    if (isscalar(*this))
    {
        cu_mat c = std::move(b);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
    	cu_mat c(n_rows,n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(n_cols == b.n_rows,"Error : Matrix multiplication is not possible. Inner matrix dimensions must agree.");
        cu_mat c(n_rows,b.n_cols);
        HANDLE_ERROR( cudaMalloc((void**)&c.p,c.n_rows*c.n_cols*sizeof(double)) ); // Allocate memory on GPU.
        double alf = 1.0, bet = 0;
        cublasHandle_t handle;
        HANDLE_ERROR( cublasCreate(&handle) );
        HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        HANDLE_ERROR( cublasDestroy(handle) );
        return std::move(c);
    }
}
cu_mat cu_mat::operator*(const cu_mat &b)&&
{
    if (isscalar(*this))
    {
        cu_mat c(b.n_rows,b.n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
    	cu_mat c = std::move(*this);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(n_cols == b.n_rows,"Error : Matrix multiplication is not possible. Inner matrix dimensions must agree.");
        cu_mat c(n_rows,b.n_cols);
        HANDLE_ERROR( cudaMalloc((void**)&c.p,c.n_rows*c.n_cols*sizeof(double)) ); // Allocate memory on GPU.
        double alf = 1.0, bet = 0;
        cublasHandle_t handle;
        HANDLE_ERROR( cublasCreate(&handle) );
        HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        HANDLE_ERROR( cublasDestroy(handle) );
        return std::move(c);
    }
}
cu_mat cu_mat::operator*(cu_mat &&b)&&
{
    if (isscalar(*this))
    {
        cu_mat c = std::move(b);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
    	cu_mat c = std::move(*this);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_mat_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(n_cols == b.n_rows,"Error : Matrix multiplication is not possible. Inner matrix dimensions must agree.");
        cu_mat c(n_rows,b.n_cols);
        HANDLE_ERROR( cudaMalloc((void**)&c.p,c.n_rows*c.n_cols*sizeof(double)) ); // Allocate memory on GPU.
        double alf = 1.0, bet = 0;
        cublasHandle_t handle;
        HANDLE_ERROR( cublasCreate(&handle) );
        HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        HANDLE_ERROR( cublasDestroy(handle) );
        return std::move(c);
    }
}
/***********************************************************************************************************************/


/***************************************   Matrix right divide operator   **************************************/
__global__ void const_a_mat_div(double *dest, double *src_a, double *src_b, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = (*src_a)/src_b[idx];
}
__global__ void const_b_mat_div(double *dest, double *src_a, double *src_b, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = src_a[idx]/(*src_b);
}
cu_mat cu_mat::operator/(const cu_mat &b) const &
{
    if (isscalar(*this))
    {
        cu_mat c(b.n_rows,b.n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        const_a_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
        cu_mat c(n_rows,n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_b_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(0,"Error : Still working on '/' operator.");
        //cu_mat c(n_rows,b.n_cols); c.del = 0;
        // double alf = 1.0, bet = 0;
        // cublasHandle_t handle;
        // HANDLE_ERROR( cublasCreate(&handle) );
        // HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        // HANDLE_ERROR( cublasDestroy(handle) );
        //return c;
    }
}
cu_mat cu_mat::operator/(cu_mat &&b) const &
{
    if (isscalar(*this))
    {
        cu_mat c = std::move(b);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_a_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,c.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
        cu_mat c(n_rows,n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_b_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(0,"Error : Still working on '/' operator.");
        //cu_mat c(n_rows,b.n_cols); c.del = 0;
        // double alf = 1.0, bet = 0;
        // cublasHandle_t handle;
        // HANDLE_ERROR( cublasCreate(&handle) );
        // HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        // HANDLE_ERROR( cublasDestroy(handle) );
        //return c;
    }
}
cu_mat cu_mat::operator/(const cu_mat &b) &&
{
    if (isscalar(*this))
    {
        cu_mat c(b.n_rows,b.n_cols);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_a_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
        cu_mat c = std::move(*this);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_b_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(0,"Error : Still working on '/' operator.");
        //cu_mat c(n_rows,b.n_cols); c.del = 0;
        // double alf = 1.0, bet = 0;
        // cublasHandle_t handle;
        // HANDLE_ERROR( cublasCreate(&handle) );
        // HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        // HANDLE_ERROR( cublasDestroy(handle) );
        //return c;
    }
}
cu_mat cu_mat::operator/(cu_mat &&b) &&
{
    if (isscalar(*this))
    {
        cu_mat c = std::move(b);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_a_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,c.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else if (isscalar(b))
    {
        cu_mat c = std::move(*this);
        size_t n_ele = c.n_rows*c.n_cols, n_threads = block_dim(n_ele);
        const_b_mat_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,b.p,n_ele);
        HANDLE_ERROR( cudaPeekAtLastError() );
        return std::move(c);
    }
    else
    {
        confirm(0,"Error : Still working on '/' operator.");
        //cu_mat c(n_rows,b.n_cols); c.del = 0;
        // double alf = 1.0, bet = 0;
        // cublasHandle_t handle;
        // HANDLE_ERROR( cublasCreate(&handle) );
        // HANDLE_ERROR( cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n_rows,b.n_cols,n_cols,&alf,p,n_rows,b.p,n_cols,&bet,c.p,n_rows) );
        // HANDLE_ERROR( cublasDestroy(handle) );
        //return c;
    }
}
/***********************************************************************************************************************/


/***************************************   Matrix addition   ****************************************/
__global__ void addition(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = a[idx] + b[idx];
}
cu_mat cu_mat::operator+(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    addition<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator+(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    addition<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator+(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    addition<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator+(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    addition<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/**********************************************************************************************************************/


/***************************************   Matrix negation   ****************************************/
__global__ void negate_mat(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = a[idx] - b[idx];
}
cu_mat cu_mat::operator-(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
	cu_mat c(n_rows,n_cols);
	size_t n_ele = c.n_rows*c.n_cols;
	size_t n_threads = block_dim(n_ele);
    negate_mat<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator-(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
	cu_mat c = std::move(b);
	size_t n_ele = c.n_rows*c.n_cols;
	size_t n_threads = block_dim(n_ele);
    negate_mat<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator-(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
	cu_mat c = std::move(*this);
	size_t n_ele = c.n_rows*c.n_cols;
	size_t n_threads = block_dim(n_ele);
    negate_mat<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator-(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix addition is not possible. Matrices must have same dimensions.");
	cu_mat c = std::move(*this);
	size_t n_ele = c.n_rows*c.n_cols;
	size_t n_threads = block_dim(n_ele);
    negate_mat<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/**********************************************************************************************************************/


/************************************   NOT operator   ***********************************************/
__global__ void elem_negoperator(double* a, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = -a[idx];
}
cu_mat cu_mat::operator-() const &
{
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_negoperator<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator-() &&
{
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_negoperator<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/***************************************   Matrix power   **************************************/
cu_mat cu_mat::operator^(const unsigned int &n)
{
    confirm(n_rows == n_cols,"Error: Matrix has to be square for matrix power(^) to be executed.")
    // confirm(n>=0,"Error: So far, only natural numbers are supported for powers.")
    if (n == 0)
    {
        return std::move(eye(n_rows,n_cols));
    }
    else if (n == 1)
    {
        return std::move(*this);
    }
    else
    {
        cu_mat tmp = *this;
        for(int i = 1; i<n; ++i)
        {
            tmp = tmp*(*this);
        }
        return std::move(tmp);
    }
}
/***********************************************************************************************************************/


/************************************   Greater than operator   ***********************************************/
__global__ void elem_greater(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] > b[idx]);
}
cu_mat cu_mat::operator>(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greater<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator>(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greater<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator>(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greater<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator>(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greater<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Smaller than operator   ***********************************************/
__global__ void elem_smaller(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] < b[idx]);
}
cu_mat cu_mat::operator<(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smaller<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator<(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smaller<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator<(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smaller<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator<(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smaller<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Greather than or equal to operator   ***********************************************/
__global__ void elem_greateroreq(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] >= b[idx]);
}
cu_mat cu_mat::operator>=(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greateroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator>=(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greateroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator>=(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greateroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator>=(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_greateroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Smaller than or equal to operator   ***********************************************/
__global__ void elem_smalleroreq(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] <= b[idx]);
}
cu_mat cu_mat::operator<=(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smalleroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator<=(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smalleroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator<=(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smalleroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator<=(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_smalleroreq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   NOT operator   ***********************************************/
__global__ void elem_notoperator(double* a, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = !a[idx];
}
cu_mat cu_mat::operator!() const &
{
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_notoperator<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator!() &&
{
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_notoperator<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Comparison equal to operator   ***********************************************/
__global__ void elem_chkeq(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] == b[idx]);
}
cu_mat cu_mat::operator==(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkeq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator==(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkeq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator==(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkeq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator==(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkeq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Comparison not equal to operator   ***********************************************/
__global__ void elem_chkneq(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] != b[idx]);
}
cu_mat cu_mat::operator!=(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkneq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator!=(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkneq<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator!=(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkneq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator!=(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_chkneq<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Logical 'AND' operator   ***********************************************/
__global__ void elem_and(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] && b[idx]);
}
cu_mat cu_mat::operator&&(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_and<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator&&(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_and<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator&&(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_and<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator&&(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_and<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Logical 'AND' operator   ***********************************************/
__global__ void elem_or(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = (a[idx] || b[idx]);
}
cu_mat cu_mat::operator||(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_or<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator||(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_or<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator||(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_or<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::operator||(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Boolean check is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_or<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/***************************************   Type conversion from cu_mat to double   **************************************/
cu_mat::operator double()
{
    confirm((n_rows==1) && (n_cols==1), "Error: Type conversion is only possible in the case of 1x1 matrix.");
    double val;
    // Copy data from GPU to CPU.
    HANDLE_ERROR( cudaMemcpy(&val,p,sizeof(double),cudaMemcpyDeviceToHost) );
    return std::move(val);
}
/***********************************************************************************************************************/


















































//  $$\      $$\                         $$\
//  $$$\    $$$ |                        $$ |
//  $$$$\  $$$$ | $$$$$$\  $$$$$$\$$$$\  $$$$$$$\   $$$$$$\   $$$$$$\
//  $$\$$\$$ $$ |$$  __$$\ $$  _$$  _$$\ $$  __$$\ $$  __$$\ $$  __$$\
//  $$ \$$$  $$ |$$$$$$$$ |$$ / $$ / $$ |$$ |  $$ |$$$$$$$$ |$$ |  \__|
//  $$ |\$  /$$ |$$   ____|$$ | $$ | $$ |$$ |  $$ |$$   ____|$$ |
//  $$ | \_/ $$ |\$$$$$$$\ $$ | $$ | $$ |$$$$$$$  |\$$$$$$$\ $$ |
//  \__|     \__| \_______|\__| \__| \__|\_______/  \_______|\__|
//	Member Functions
/************************************   Two argument memory allocation with initialization   ***********************************************/
void cu_mat::init(const size_t &r, const size_t &c)
{
    n_rows = r; n_cols = c;
    if ((n_rows>0)&&(n_cols>0))
    {
        HANDLE_ERROR( cudaMalloc((void**)&p, n_rows*n_cols*sizeof(double)) );
        HANDLE_ERROR( cudaMemset(p,0,n_rows*n_cols*sizeof(double)) );
    }
}
/***********************************************************************************************************************/


/************************************   Element wise division   ***********************************************/
__global__ void elem_div(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = a[idx] / b[idx];
}
cu_mat cu_mat::div(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_div<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::div(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_div<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::div(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::div(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_div<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Element wise multiplication   ***********************************************/
__global__ void elem_mult(double* a, double* b, double* c, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        c[idx] = a[idx] * b[idx];
}
cu_mat cu_mat::mult(const cu_mat &b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::mult(cu_mat &&b) const &
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(b);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(p,c.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::mult(const cu_mat &b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::mult(cu_mat &&b) &&
{
    confirm((n_rows == b.n_rows) && (n_cols == b.n_cols),"Error : Matrix multiplication is not possible. Matrices must have same dimensions.");
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_mult<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,b.p,c.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Element wise power   ***********************************************/
__global__ void elem_power(double* dest, double* src, double n, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = pow(src[idx],n);
}
cu_mat cu_mat::pow(const double &n) const &
{
    cu_mat c(n_rows,n_cols);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_power<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,p,n,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
cu_mat cu_mat::pow(const double &n) &&
{
    cu_mat c = std::move(*this);
    size_t n_ele = c.n_rows*c.n_cols;
    size_t n_threads = block_dim(n_ele);
    elem_power<<<n_blocks(n_ele,n_threads),n_threads>>>(c.p,c.p,n,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(c);
}
/***********************************************************************************************************************/


/************************************   Replace an element with a 'cu_mat' value   ***********************************************/
void cu_mat::replace(const size_t &r, const size_t &c, const cu_mat &n)
{
    confirm((n.n_rows==1) && (n.n_cols==1),"Error: Value being replaced with has to be scalar.");
    size_t bias = c*n_rows+r, src_rows = 1, src_cols = 1;
    size_t main_rows_bias = n_rows-src_rows, n_ele = src_rows*src_cols, n_threads = block_dim(n_ele);
    copymat<<<n_blocks(n_ele,n_threads),n_threads>>>(p,n.p,bias,src_rows,main_rows_bias,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
}
/***********************************************************************************************************************/


/************************************   Replace submatrix with a 'cu_mat' matrix   ***********************************************/
void cu_mat::replace(const size_t &r_begin, const size_t &r_end, const size_t &c_begin, const size_t &c_end, const cu_mat &n)
{
    confirm((r_end<=n_rows) && (c_end<=n_cols),"Error: Index exceeds matrix bounds. The size of the matrix is " << n_rows << "x" << n_cols << ".");
    confirm((n.n_rows==r_end-r_begin+1) && (n.n_cols==c_end-c_begin+1),"Error: Unable to replace the data due to size mismatch.");
    size_t bias = (c_begin-1)*n_rows+r_begin-1, src_rows = n.n_rows, src_cols = n.n_cols;
    size_t main_rows_bias = n_rows-src_rows, n_ele = src_rows*src_cols, n_threads = block_dim(n_ele);
    copymat<<<n_blocks(n_ele,n_threads),n_threads>>>(p,n.p,bias,src_rows,main_rows_bias,n_ele);
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

    std::cout << std::scientific << std::setprecision(4);
    for(int i = 0; i<n_rows; ++i)
    {
        for(int j = 0; j<n_cols; ++j)
        {
            std::cout<<m[j*n_rows+i]<<" ";
        }
        std::cout<<std::endl;
    }
    delete[] m;
}
/***********************************************************************************************************************/


/************************************   Print matrix to a file   ***********************************************/
void cu_mat::print(std::ofstream &print)
{
    double *m = new double[n_rows*n_cols]();    // Allocate space on CPU memory.
    confirm(m,"Error: Memory allocation failed in 'print()'.") // Check proper allocation.

    // Copy data from GPU to CPU.
    HANDLE_ERROR( cudaMemcpy(m,p,n_rows*n_cols*sizeof(double),cudaMemcpyDeviceToHost) );

    // Print the matrix
    print << std::scientific << std::setprecision(8);
    for(int i = 0; i<n_rows; ++i)
    {
        print << " ";
        for(int j = 0; j<n_cols; ++j)
        {
            print << " " << m[j*n_rows+i] << " ";
        }
        print << std::endl;
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


/***************************************   Get GPU memory pointer   *****************************************/
double* cu_mat::pointer(){return p;}
/***********************************************************************************************************************/


/***************************************   Get number of rows   *****************************************/
size_t cu_mat::rows() const {return n_rows;}
/***********************************************************************************************************************/


/***************************************   Get number of columns   *****************************************/
size_t cu_mat::cols() const {return n_cols;}
/***********************************************************************************************************************/


/***************************************   Get GPU memory pointer   *****************************************/
double* cu_mat::pointer() const {return p;}
/***********************************************************************************************************************/


















































//  $$$$$$$$\        $$\                           $$\
//  $$  _____|       \__|                          $$ |
//  $$ |    $$$$$$\  $$\  $$$$$$\  $$$$$$$\   $$$$$$$ |
//  $$$$$\ $$  __$$\ $$ |$$  __$$\ $$  __$$\ $$  __$$ |
//  $$  __|$$ |  \__|$$ |$$$$$$$$ |$$ |  $$ |$$ /  $$ |
//  $$ |   $$ |      $$ |$$   ____|$$ |  $$ |$$ |  $$ |
//  $$ |   $$ |      $$ |\$$$$$$$\ $$ |  $$ |\$$$$$$$ |
//  \__|   \__|      \__| \_______|\__|  \__| \_______|
//	Friend Functions
/**************************************   Matrix with random numbers   ***********************************************/
cu_mat randn(const size_t &r, const size_t &c)
{
    size_t r_new = r, c_new = c;
    if ((r%2!=0)&&(c%2!=0))
    {
        r_new = r+1; c_new = c+1;
    }
    cu_mat a(r_new,c_new);
    curandGenerator_t prng;
	HANDLE_ERROR( curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW) );
	HANDLE_ERROR( curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long) clock()) );
	HANDLE_ERROR( curandGenerateNormalDouble(prng,a.p,a.n_rows*a.n_cols,0.0,1.0) ); //The number of values requested has to be multiple of 2.
    HANDLE_ERROR( curandDestroyGenerator(prng) );
    return std::move(a(1,r,1,c));
}
cu_mat randn(const size_t &n=1){return std::move(randn(n,n));}
/***************************************************************************************************************************/


/****************************************   Identity matrix   *******************************************/
__global__ void eye_mat(double* p, const int r, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        p[idx*r+idx] = 1.0;
}
cu_mat eye(const size_t &r, const size_t &c)
{
    cu_mat temp(r,c);
    size_t n_ele = min(r,c);
    size_t n_threads = block_dim(n_ele);
    eye_mat<<<n_blocks(n_ele,n_threads),n_threads>>>(temp.p,r,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(temp);
}
cu_mat eye(const size_t &n){return std::move(eye(n,n));}
/***************************************************************************************************************************/


/*****************************************   Matrix left divide   *****************************************/
cu_mat mld(const cu_mat &a, const cu_mat &b)
{
	return std::move(mld(cu_mat(a),cu_mat(b)));
}
cu_mat mld(const cu_mat &a, cu_mat &&b)
{
	return std::move(mld(cu_mat(a),std::move(b)));
}
cu_mat mld(cu_mat &&a, const cu_mat &b)
{
	return std::move(mld(std::move(a),cu_mat(b)));
}
cu_mat mld(cu_mat &&a, cu_mat &&b) // Adapted from CUSOLVER_Library.pdf QR examples
{
    confirm(a.n_rows == b.n_rows,"Error: 'mld()' operation cannot be performed. Matrix dimensions must agree.")
    cu_mat A = std::move(a), B = std::move(b); // Copy current matrix to a new matrix for calculations.

    double *d_tau = NULL;
    double *d_work = NULL, alf = 1.0;
    int *devInfo = NULL, lwork = 0, info_gpu = 0;
    cusolverDnHandle_t cusolver_handle = NULL;
    cublasHandle_t cublas_handle = NULL;

    // step 1: create cusolver/cublas handle
    HANDLE_ERROR( cusolverDnCreate(&cusolver_handle) );
    HANDLE_ERROR( cublasCreate(&cublas_handle) );

    // step 2: allocate required extra memory on GPU.
    HANDLE_ERROR( cudaMalloc((void**)&d_tau,sizeof(double)*A.n_cols) );
    HANDLE_ERROR( cudaMalloc((void**)&devInfo,sizeof(int)) );

    // step 3: query working space of geqrf and ormqr
    HANDLE_ERROR( cusolverDnDgeqrf_bufferSize(cusolver_handle,A.n_rows,A.n_cols,A.p,A.n_rows,&lwork) );
    HANDLE_ERROR( cudaMalloc((void**)&d_work, sizeof(double)*lwork) );

    // step 4: compute QR factorization
    HANDLE_ERROR( cusolverDnDgeqrf(cusolver_handle,A.n_rows,A.n_cols,A.p,A.n_rows,d_tau,d_work,lwork,devInfo) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    // check if QR is good or not
    HANDLE_ERROR( cudaMemcpy(&info_gpu, devInfo, sizeof(int),cudaMemcpyDeviceToHost) );
    confirm(info_gpu == 0,"Error: 'mld()' operation cannot be performed. QR decomposition failed.");

    // step 5: compute Q^T*B (CUSOLVER documentation has typos. Follow LAPACK documentation.)
    HANDLE_ERROR( cusolverDnDormqr(cusolver_handle,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,B.n_rows,B.n_cols,A.n_cols,A.p,A.n_rows,d_tau,B.p,B.n_rows,d_work,lwork,devInfo) );
    HANDLE_ERROR( cudaDeviceSynchronize() );
    // check if QR is good or not
    HANDLE_ERROR( cudaMemcpy(&info_gpu, devInfo, sizeof(int),cudaMemcpyDeviceToHost) );
    confirm(info_gpu == 0,"Error: 'mld()' operation cannot be performed. QR decomposition failed.");

    // step 6: compute x = R \ (Q^T*B)
    HANDLE_ERROR( cublasDtrsm(cublas_handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,A.n_cols,B.n_cols,&alf,A.p,A.n_rows,B.p,A.n_cols) );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // Free resources
    HANDLE_ERROR( cudaFree(d_tau) );
    HANDLE_ERROR( cudaFree(devInfo) );
    HANDLE_ERROR( cudaFree(d_work) );

    HANDLE_ERROR( cublasDestroy(cublas_handle) );
    HANDLE_ERROR( cusolverDnDestroy(cusolver_handle) );

    return std::move(B);
}
/***************************************************************************************************************************/


/*****************************************   Matrix with all values 1   *****************************************/
cu_mat ones(const size_t &r, const size_t &c)
{
    cu_mat tmp(r,c,1); //tmp.del = 0;
    return std::move(tmp);
}
cu_mat ones(const size_t &n){return std::move(ones(n,n));}
/***************************************************************************************************************************/

/*****************************************   Matrix with all values 0   *****************************************/
cu_mat zeros(const size_t &r, const size_t &c)
{
    cu_mat tmp(r,c); //tmp.del = 0;
    return std::move(tmp);
}
cu_mat zeros(const size_t &n){return std::move(zeros(n,n));}
/***************************************************************************************************************************/


/***************************************   Transpose current matrix   *****************************************/
__global__ void mat_trans(double* a, double* b, size_t rows, size_t cols, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t r = idx%rows, c = idx/rows;
    if (idx<n_ele)
        a[c+r*cols] = b[idx];
}
cu_mat trans(cu_mat &a)
{
    cu_mat tmp(a.n_cols,a.n_rows);
    size_t n_ele = tmp.n_rows*tmp.n_cols;
    size_t n_threads = block_dim(n_ele);
    mat_trans<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,a.n_rows,a.n_cols,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/***************************************   Horizontal concatenation of two matrices   *****************************************/
cu_mat horzcat(cu_mat &a, cu_mat &b)
{
    confirm(a.n_rows==b.n_rows,"Error: Dimensions of arrays being horizontally concatenated are not consistent.");
    cu_mat tmp(a.n_rows,a.n_cols+b.n_cols);
    HANDLE_ERROR( cudaMemcpy(tmp.p,a.p,a.n_rows*a.n_cols*sizeof(double),cudaMemcpyDeviceToDevice) );
    size_t n_ele = b.n_rows*b.n_cols, n_threads = block_dim(n_ele);
    copymat<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,b.p,a.n_cols*tmp.n_rows,tmp.n_rows,0,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***************************************************************************************************************************/


/***************************************   Vertical concatenation of two matrices   *****************************************/
    // cu_mat temp(r_end-r_begin+1,c_end-c_begin+1);
    // size_t bias = (c_begin-1)*n_rows+r_begin-1;
    // size_t main_rows_bias = n_rows-temp.n_rows;
    // size_t n_ele = temp.n_rows*temp.n_cols;
    // size_t n_threads = block_dim(n_ele);
    // copymat<<<n_blocks(n_ele,n_threads),n_threads>>>(p,temp.p,bias,temp.n_rows,main_rows_bias,n_ele);
    // HANDLE_ERROR( cudaPeekAtLastError() );
cu_mat vertcat(cu_mat &a, cu_mat &b)
{
    confirm(a.n_cols==b.n_cols,"Error: Dimensions of arrays being vertically concatenated are not consistent.");
    cu_mat tmp(a.n_rows+b.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    copymat<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,0,a.n_rows,tmp.n_rows-a.n_rows,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    n_ele = b.n_rows*b.n_cols; n_threads = block_dim(n_ele);
    copymat<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,b.p,a.n_rows,b.n_rows,tmp.n_rows-b.n_rows,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***************************************************************************************************************************/


/***************************************   MATLAB colon operator   *****************************************/
__global__ void ss_mat_fill(double* dest, double i, double step, size_t n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx<n_ele)
        dest[idx] = i+step*idx;
}
cu_mat stepspace(const double &i, const double &f, const double &step=1)
{
    size_t n;
    if (((f-i)/step)>=0)
    	n = (f-i)/step+1;
    else
    	n = 0;
    cu_mat tmp(n,1);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    ss_mat_fill<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,i,step,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***************************************************************************************************************************/


/***************************************   Norm of the matrix   *****************************************/
cu_mat norm(const cu_mat &a, const double &p = 2)
{
    confirm((p==1) || (p==2) || isinf(p),"Error: 'norm' is available only for 1, 2 and inf types.");
    cu_mat c = 0;
    cublasHandle_t handle;
    HANDLE_ERROR( cublasCreate(&handle) );
    if(p==1)
    {
        HANDLE_ERROR( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
        HANDLE_ERROR( cublasDasum(handle,static_cast<int>(a.rows()*a.cols()),a.pointer(),1,c.p) );
    }
    else if(p==2)
    {
        HANDLE_ERROR( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) );
        HANDLE_ERROR( cublasDnrm2(handle,static_cast<int>(a.rows()*a.cols()),a.pointer(),1,c.p) );
    }
    else
    {
        int idx;
        HANDLE_ERROR( cublasIdamax(handle,a.n_rows*a.n_cols,a.p,1,&idx) );
        c = abs(a(idx));
    }
    HANDLE_ERROR( cublasDestroy(handle) );
    return std::move(c);
}
/***************************************************************************************************************************/


















































//  $$\      $$\            $$\     $$\
//  $$$\    $$$ |           $$ |    $$ |
//  $$$$\  $$$$ | $$$$$$\ $$$$$$\   $$$$$$$\
//  $$\$$\$$ $$ | \____$$\\_$$  _|  $$  __$$\
//  $$ \$$$  $$ | $$$$$$$ | $$ |    $$ |  $$ |
//  $$ |\$  /$$ |$$  __$$ | $$ |$$\ $$ |  $$ |
//  $$ | \_/ $$ |\$$$$$$$ | \$$$$  |$$ |  $$ |
//  \__|     \__| \_______|  \____/ \__|  \__|
//	Math Functions
/************************************   Calculate arc cosine of each element   ***********************************************/
__global__ void mat_arccosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = acos(src[idx]);
}
cu_mat acos(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_arccosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat acos(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_arccosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate arc hyperbolic cosine of each element   ***********************************************/
__global__ void mat_archypcosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = acosh(src[idx]);
}
cu_mat acosh(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_archypcosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat acosh(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_archypcosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate arc sine of each element   ***********************************************/
__global__ void mat_arcsine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = asin(src[idx]);
}
cu_mat asin(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_arcsine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat asin(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_arcsine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate arc hyperbolic sine of each element   ***********************************************/
__global__ void mat_archypsine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = asinh(src[idx]);
}
cu_mat asinh(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_archypsine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat asinh(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_archypsine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate arc tangent of each element   ***********************************************/
__global__ void mat_arctan(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = atan(src[idx]);
}
cu_mat atan(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_arctan<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat atan(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_arctan<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate four quadrant arc tangent of each element   ***********************************************/
__global__ void mat_arctangent2(double* dest, double* src_a, double* src_b, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = atan2(src_a[idx],src_b[idx]);
}
cu_mat atan2(const cu_mat &y, const cu_mat &x)
{
    confirm((y.n_rows==x.n_rows)&&(y.n_cols==x.n_cols),"Error: 'atan2' cannot be used. Both matrices has to be of the same size.")
    cu_mat tmp(x.rows(),x.cols());
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_arctangent2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,y.p,x.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat atan2(const cu_mat &y, cu_mat &&x)
{
    confirm((y.n_rows==x.n_rows)&&(y.n_cols==x.n_cols),"Error: 'atan2' cannot be used. Both matrices has to be of the same size.")
    cu_mat tmp = std::move(x);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_arctangent2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,y.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat atan2(cu_mat &&y, const cu_mat &x)
{
    confirm((y.n_rows==x.n_rows)&&(y.n_cols==x.n_cols),"Error: 'atan2' cannot be used. Both matrices has to be of the same size.")
    cu_mat tmp = std::move(y);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_arctangent2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,x.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat atan2(cu_mat &&y, cu_mat &&x)
{
    confirm((y.n_rows==x.n_rows)&&(y.n_cols==x.n_cols),"Error: 'atan2' cannot be used. Both matrices has to be of the same size.")
    cu_mat tmp = std::move(y);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_arctangent2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,x.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic arc tangent of each element   ***********************************************/
__global__ void mat_archyptan(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = atanh(src[idx]);
}
cu_mat atanh(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_archyptan<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat atanh(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_archyptan<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate ceiling of each element   ***********************************************/
__global__ void mat_ceiling(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = ceil(src[idx]);
}
cu_mat ceil(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_ceiling<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat ceil(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_ceiling<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate cosine of each element   ***********************************************/
__global__ void mat_cosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = cos(src[idx]);
}
cu_mat cos(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_cosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat cos(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_cosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic cosine of each element   ***********************************************/
__global__ void mat_hypcosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = cosh(src[idx]);
}
cu_mat cosh(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_hypcosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat cosh(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_hypcosine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate e^x of each element   ***********************************************/
__global__ void mat_exponent(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = exp(src[idx]);
}
cu_mat exp(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_exponent<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat exp(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_exponent<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate 10^x of each element   ***********************************************/
__global__ void mat_exponent10(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = exp10(src[idx]);
}
cu_mat exp10(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_exponent10<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat exp10(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_exponent10<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate 2^x of each element   ***********************************************/
__global__ void mat_exponent2(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = exp2(src[idx]);
}
cu_mat exp2(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_exponent2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat exp2(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_exponent2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate absolute of each element   ***********************************************/
__global__ void mat_absolute(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = fabs(src[idx]);
}
cu_mat abs(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_absolute<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat abs(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_absolute<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate floor value of each element   ***********************************************/
__global__ void mat_floor(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = floor(src[idx]);
}
cu_mat floor(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_floor<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat floor(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_floor<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate modulo of a/b   ***********************************************/
__global__ void mat_modulo(double* dest, double* src_a, double* src_b, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = fmod(src_a[idx],src_b[idx]);
}
cu_mat mod(const cu_mat &a, const cu_mat &b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'mod' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp(a.rows(),a.cols());
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_modulo<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat mod(const cu_mat &a, cu_mat &&b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'mod' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp = std::move(b);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_modulo<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat mod(cu_mat &&a, const cu_mat &b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'mod' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_modulo<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat mod(cu_mat &&a, cu_mat &&b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'mod' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_modulo<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Check if each element of matrix is not inf or nan   ***********************************************/
__global__ void mat_isfinite(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = isfinite(src[idx]);
}
cu_mat isfinite(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_isfinite<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat isfinite(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_isfinite<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Check if each element is inf   ***********************************************/
__global__ void mat_isinfinite(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = isinf(src[idx]);
}
cu_mat isinf(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_isinfinite<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat isinf(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_isinfinite<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Check if each element is nan   ***********************************************/
__global__ void mat_isnan(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = isnan(src[idx]);
}
cu_mat isnan(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_isnan<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat isnan(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_isnan<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Check if 'cu_mat' object is empty   ***********************************************/
bool isempty(cu_mat &a)
{
    return ((a.n_rows*a.n_cols)==0);
}
/***********************************************************************************************************************/


/************************************   Check if 'cu_mat' object is scalar   ***********************************************/
bool isscalar(const cu_mat &a)
{
    return ((a.n_rows*a.n_cols)==1);
}
/***********************************************************************************************************************/


/************************************   Check if 'cu_mat' object is scalar   ***********************************************/
bool isscalar(cu_mat &a)
{
    return ((a.n_rows*a.n_cols)==1);
}
/***********************************************************************************************************************/


/************************************   Check if 'cu_mat' object is vector   ***********************************************/
bool isvector(cu_mat &a)
{
    return (((a.n_rows*a.n_cols)==a.n_rows) || ((a.n_rows*a.n_cols)==a.n_cols));
}
/***********************************************************************************************************************/


/************************************   Check if 'cu_mat' object is matrix   ***********************************************/
// bool ismatrix(const cu_mat &a)
// {
//     return (!(isscalar(a)||isvector(a)||isempty(a)));
// }
/***********************************************************************************************************************/


/************************************   Calculate log_e of each element   ***********************************************/
__global__ void mat_log_e(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = log(src[idx]);
}
cu_mat log(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_log_e<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat log(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_log_e<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate log_10 of each element   ***********************************************/
__global__ void mat_log_10(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = log10(src[idx]);
}
cu_mat log10(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_log_10<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat log10(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_log_10<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate log_2 of each element   ***********************************************/
__global__ void mat_log_2(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = log2(src[idx]);
}
cu_mat log2(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_log_2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat log2(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_log_2<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate remainder of each element of a/b   ***********************************************/
__global__ void mat_remainder(double* dest, double* src_a, double* src_b, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = remainder(src_a[idx],src_b[idx]);
}
cu_mat rem(const cu_mat &a, const cu_mat &b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'rem' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp(a.rows(),a.cols());
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_remainder<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat rem(const cu_mat &a, cu_mat &&b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'rem' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp = std::move(b);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_remainder<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat rem(cu_mat &&a, const cu_mat &b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'rem' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_remainder<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
cu_mat rem(cu_mat &&a, cu_mat &&b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'rem' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_remainder<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate rounded value of each element   ***********************************************/
__global__ void mat_round(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = round(src[idx]);
}
cu_mat round(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_round<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat round(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_round<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate sign of each element   ***********************************************/
__global__ void mat_signbit(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        if(signbit(src[idx])==0)
            dest[idx] = 1;
        else
            dest[idx] = -1;
}
cu_mat sign(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_signbit<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat sign(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_signbit<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate sine of each element   ***********************************************/
__global__ void mat_sine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = sin(src[idx]);
}
cu_mat sin(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_sine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat sin(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_sine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic sine of each element   ***********************************************/
__global__ void mat_hypsine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = sinh(src[idx]);
}
cu_mat sinh(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_hypsine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat sinh(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_hypsine<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate square root of each element   ***********************************************/
__global__ void mat_sqrt(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = sqrt(src[idx]);
}
cu_mat sqrt(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_sqrt<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat sqrt(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_sqrt<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate tangent of each element   ***********************************************/
__global__ void mat_tangent(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = tan(src[idx]);
}
cu_mat tan(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_tangent<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat tan(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_tangent<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic tangent of each element   ***********************************************/
__global__ void mat_hyptangent(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
        dest[idx] = tanh(src[idx]);
}
cu_mat tanh(const cu_mat &a)
{
	cu_mat tmp(a.rows(),a.cols());
	size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
	mat_hyptangent<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,a.p,n_ele);
	HANDLE_ERROR( cudaPeekAtLastError() );
	return std::move(tmp);
}
cu_mat tanh(cu_mat &&a)
{
    cu_mat tmp = std::move(a);
    size_t n_ele = tmp.n_rows*tmp.n_cols, n_threads = block_dim(n_ele);
    mat_hyptangent<<<n_blocks(n_ele,n_threads),n_threads>>>(tmp.p,tmp.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return std::move(tmp);
}
/***********************************************************************************************************************/


















































//  $$$$$$$\                        $$\                                     $$\
//  $$  __$$\                       $$ |                                    $$ |
//  $$ |  $$ | $$$$$$\   $$$$$$$\ $$$$$$\    $$$$$$\  $$\   $$\  $$$$$$$\ $$$$$$\    $$$$$$\   $$$$$$\
//  $$ |  $$ |$$  __$$\ $$  _____|\_$$  _|  $$  __$$\ $$ |  $$ |$$  _____|\_$$  _|  $$  __$$\ $$  __$$\
//  $$ |  $$ |$$$$$$$$ |\$$$$$$\    $$ |    $$ |  \__|$$ |  $$ |$$ /        $$ |    $$ /  $$ |$$ |  \__|
//  $$ |  $$ |$$   ____| \____$$\   $$ |$$\ $$ |      $$ |  $$ |$$ |        $$ |$$\ $$ |  $$ |$$ |
//  $$$$$$$  |\$$$$$$$\ $$$$$$$  |  \$$$$  |$$ |      \$$$$$$  |\$$$$$$$\   \$$$$  |\$$$$$$  |$$ |
//  \_______/  \_______|\_______/    \____/ \__|       \______/  \_______|   \____/  \______/ \__|
//	Destructor
cu_mat::~cu_mat() {
	HANDLE_ERROR( cudaFree(p) );
	p = NULL;
}
