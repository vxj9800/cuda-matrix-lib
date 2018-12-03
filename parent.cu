#include <iostream>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"

int main()
{
    // cu_matrix a = {{1,2,3},{4,5,6}};
    // cu_matrix b = {{1,2},{3,4},{5,6}};
    // cu_matrix c = a*b;
    // c.get();
    cu_matrix a = randn(2,3);
    cu_matrix b = randn(3,2);
    cu_matrix c = a*b;
    a.get(); b.get(); c.get();
    return (0);
}