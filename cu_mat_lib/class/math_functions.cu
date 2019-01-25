#ifndef _CU_MATRIX_CLASS_MATH_FUNCTIONS_INCLUDED_
#define _CU_MATRIX_CLASS_MATH_FUNCTIONS_INCLUDED_

/************************************   Calculate arc cosine of each element   ***********************************************/
__global__ void mat_arccosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = acos(src[idx]);
    }
}
cu_mat acos(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_arccosine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate arc hyperbolic cosine of each element   ***********************************************/
__global__ void mat_archypcosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = acosh(src[idx]);
    }
}
cu_mat acosh(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_archypcosine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate arc sine of each element   ***********************************************/
__global__ void mat_arcsine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = asin(src[idx]);
    }
}
cu_mat asin(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_arcsine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate arc hyperbolic sine of each element   ***********************************************/
__global__ void mat_archypsine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = asinh(src[idx]);
    }
}
cu_mat asinh(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_archypsine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate arc tangent of each element   ***********************************************/
__global__ void mat_arctan(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = atan(src[idx]);
    }
}
cu_mat atan(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_arctan<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate four quadrant arc tangent of each element   ***********************************************/
__global__ void mat_arctangent2(double* dest, double* src_a, double* src_b, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = atan2(src_a[idx],src_b[idx]);
    }
}
cu_mat atan2(const cu_mat &y, const cu_mat &x)
{
    confirm((y.n_rows==x.n_rows)&&(y.n_cols==x.n_cols),"Error: 'atan2' cannot be used. Both matrices has to be of the same size.")
    cu_mat tmp(y.n_rows,y.n_cols);
    size_t n_ele = y.n_rows*y.n_cols, n_threads = block_dim(n_ele);
    mat_arctangent2<<<n_ele/n_threads,n_threads>>>(tmp.p,y.p,x.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic arc tangent of each element   ***********************************************/
__global__ void mat_archyptan(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = atanh(src[idx]);
    }
}
cu_mat atanh(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_archyptan<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate ceiling of each element   ***********************************************/
__global__ void mat_ceiling(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = ceil(src[idx]);
    }
}
cu_mat ceil(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_ceiling<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate cosine of each element   ***********************************************/
__global__ void mat_cosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = cos(src[idx]);
    }
}
cu_mat cos(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_cosine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic cosine of each element   ***********************************************/
__global__ void mat_hypcosine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = cosh(src[idx]);
    }
}
cu_mat cosh(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_hypcosine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate e^x of each element   ***********************************************/
__global__ void mat_exponent(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = exp(src[idx]);
    }
}
cu_mat exp(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_exponent<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate 10^x of each element   ***********************************************/
__global__ void mat_exponent10(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = exp10(src[idx]);
    }
}
cu_mat exp10(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_exponent10<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate 2^x of each element   ***********************************************/
__global__ void mat_exponent2(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = exp2(src[idx]);
    }
}
cu_mat exp2(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_exponent2<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate absolute of each element   ***********************************************/
__global__ void mat_absolute(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = fabs(src[idx]);
    }
}
cu_mat abs(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_absolute<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate floor value of each element   ***********************************************/
__global__ void mat_floor(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = floor(src[idx]);
    }
}
cu_mat floor(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_floor<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate modulo of a/b   ***********************************************/
__global__ void mat_modulo(double* dest, double* src_a, double* src_b, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = fmod(src_a[idx],src_b[idx]);
    }
}
cu_mat mod(const cu_mat &a, const cu_mat &b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'mod' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_modulo<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Check if each element of matrix is not inf or nan   ***********************************************/
__global__ void mat_isfinite(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = isfinite(src[idx]);
    }
}
cu_mat isfinite(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_isfinite<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Check if each element is inf   ***********************************************/
__global__ void mat_isinfinite(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = isinf(src[idx]);
    }
}
cu_mat isinf(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_isinfinite<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Check if each element is nan   ***********************************************/
__global__ void mat_isnan(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = isnan(src[idx]);
    }
}
cu_mat isnan(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_isnan<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Check if 'cu_mat' object is empty   ***********************************************/
bool isempty(const cu_mat &a)
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


/************************************   Check if 'cu_mat' object is vector   ***********************************************/
bool isvector(const cu_mat &a)
{
    return (((a.n_rows*a.n_cols)==a.n_rows) || ((a.n_rows*a.n_cols)==a.n_cols));
}
/***********************************************************************************************************************/


/************************************   Check if 'cu_mat' object is matrix   ***********************************************/
bool ismatrix(const cu_mat &a)
{
    return (!(isscalar(a)||isvector(a)||isempty(a)));
}
/***********************************************************************************************************************/


/************************************   Calculate log_e of each element   ***********************************************/
__global__ void mat_log_e(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = log(src[idx]);
    }
}
cu_mat log(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_log_e<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate log_10 of each element   ***********************************************/
__global__ void mat_log_10(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = log10(src[idx]);
    }
}
cu_mat log10(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_log_10<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate log_2 of each element   ***********************************************/
__global__ void mat_log_2(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = log2(src[idx]);
    }
}
cu_mat log2(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_log_2<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate remainder of each element of a/b   ***********************************************/
__global__ void mat_remainder(double* dest, double* src_a, double* src_b, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = remainder(src_a[idx],src_b[idx]);
    }
}
cu_mat rem(const cu_mat &a, const cu_mat &b)
{
    confirm((a.n_rows==b.n_rows)&&(a.n_cols==b.n_cols),"Error: 'rem' cannot be calculated. Both matrices has to be of same size.")
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_remainder<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,b.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate rounded value of each element   ***********************************************/
__global__ void mat_round(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = round(src[idx]);
    }
}
cu_mat round(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_round<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate sign of each element   ***********************************************/
__global__ void mat_signbit(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        if(signbit(src[idx])==0)
            dest[idx] = 1;
        else
            dest[idx] = -1;
    }
}
cu_mat sign(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_signbit<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate sine of each element   ***********************************************/
__global__ void mat_sine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = sin(src[idx]);
    }
}
cu_mat sin(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_sine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic sine of each element   ***********************************************/
__global__ void mat_hypsine(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = sinh(src[idx]);
    }
}
cu_mat sinh(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_hypsine<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate square root of each element   ***********************************************/
__global__ void mat_sqrt(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = sqrt(src[idx]);
    }
}
cu_mat sqrt(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_sqrt<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate tangent of each element   ***********************************************/
__global__ void mat_tangent(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = tan(src[idx]);
    }
}
cu_mat tan(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_tangent<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/


/************************************   Calculate hyperbolic tangent of each element   ***********************************************/
__global__ void mat_hyptangent(double* dest, double* src, const int n_ele)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx<n_ele)
    {
        dest[idx] = tanh(src[idx]);
    }
}
cu_mat tanh(const cu_mat &a)
{
    cu_mat tmp(a.n_rows,a.n_cols);
    size_t n_ele = a.n_rows*a.n_cols, n_threads = block_dim(n_ele);
    mat_hyptangent<<<n_ele/n_threads,n_threads>>>(tmp.p,a.p,n_ele);
    HANDLE_ERROR( cudaPeekAtLastError() );
    return tmp;
}
/***********************************************************************************************************************/

#endif