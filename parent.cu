#include <iostream>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"

int main()
{
    look_for_errors;
    cu_mat a = {{1,2,3,4},{4,5,6}};
    a.get();
    // cu_mat b = {{1,2},{4,5}};
    // cu_mat c = a*b;
    // a.get(); b.get(); c.get();

    // cu_mat a = randn(4,4);
    // cu_mat b = randn(4,1);
    // a.get(); b.get(); mld(a,b).get();
    report_errors;
    
    return (0);
}