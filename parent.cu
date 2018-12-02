#include <iostream>
using namespace std;

#include "cu_matrix.cu"

int main()
{
    cu_matrix a = {{1,2,3},{4,5,6}};
    a.get();
    return (0);
}