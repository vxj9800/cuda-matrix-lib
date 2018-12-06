#include <iostream>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"

int main()
{
    cu_mat a = {{1,2,3},{4,5,6}};
    a(2,2).get();
    // cu_mat b = {{1,2},{4,5}};
    // cu_mat c = a*b;
    // a.get(); b.get(); c.get();

    // cu_mat a = randn(4,4);
    // cu_mat b = randn(4,1);
    // a.get(); b.get(); mld(a,b).get();
    
    return (0);
}