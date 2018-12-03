cu_matrix randn(size_t r = 1, size_t c = 1)
{
    cu_matrix a(r,c);
    curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(prng,(unsigned long long) clock());
	curandGenerateNormalDouble(prng,a.p,r*c,0.0,1.0); //The number of values requested has to be multiple of 2.
    curandDestroyGenerator(prng);
    return a;
}