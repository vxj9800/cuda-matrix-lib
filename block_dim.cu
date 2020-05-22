#include "imp_includes.hcu"

// size_t block_dim(size_t n_ele)
// {
// 	size_t tc;
// 	if (n_ele<=1024) return(n_ele);

// 	else if (n_ele%2==0){
// 		for (tc=1024;tc>1;tc=tc-2){
// 			if (n_ele%tc==0) return(tc);}}

// 	else{
// 		for (tc=1023;tc>1;tc=tc-2){
// 			if (n_ele%tc==0) return(tc);}}

// 	std::cout << "Warning: Calculations cannot be divided equally. Be careful when making the CUDA kernel." << std::endl;
// 	return(256);
// }

size_t block_dim(size_t n_ele)
{
	size_t tc = 0;
	if (n_ele<=32) return(32);

	else if (n_ele%32==0)
	{
		for (tc=1024;tc>31;tc=tc-32)
		{
			if (n_ele%tc==0) return(tc);
		}
	}

	else
	{
		size_t minDiff = 1024, minDiffTc;
		for (tc=32;tc<1025;tc=tc+32)
		{
			// std::cout << tc << std::endl;
			if (((n_ele/tc)+1)*tc-n_ele <= minDiff)
			{
				minDiffTc = tc; minDiff = ((n_ele/tc)+1)*tc-n_ele;
			}
		}
		std::cout << "Warning: Block of " << minDiffTc << " threads chosen for " << n_ele << " elements. Be careful when making the CUDA kernel." << std::endl;
		return (minDiffTc);
	}

	std::cout << "Warning: Calculations cannot be divided equally. Be careful when making the CUDA kernel." << std::endl;
	return(256);
}