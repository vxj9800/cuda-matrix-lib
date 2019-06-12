/*
 * cu_mat_test.cu
 *
 *  Created on: Jun 6, 2019
 *      Author: vatsal
 */

#include "cu_mat.hcu"

int main()
{
//	cu_mat a = {{1,1,1},{0,2,5},{2,5,-1}};
	cu_mat b = {{6},{-4},{27}};
	cu_mat c = mld(cu_mat({{1,1,1},{0,2,5},{2,5,-1}}),b);
//	std::cout << (cu_mat(3.0)+a).is_rvalue() << std::endl;
	b.get(); c.get();
	return 0;
}
