#include "imp_includes.hcu"

size_t block_dim(size_t n_ele)
{
	size_t tc;
	if (n_ele<=1024) return(n_ele);

	else if (n_ele%2==0){
		for (tc=1024;tc>1;tc=tc-2){
			if (n_ele%tc==0) return(tc);}}

	else{
		for (tc=1023;tc>1;tc=tc-2){
			if (n_ele%tc==0) return(tc);}}

	std::cout << "Warning: Calculations cannot be divided equally. Be careful when making the CUDA kernel." << std::endl;
	return(256);
}
