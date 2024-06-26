//
// Created by rick on 30/11/23.
//


#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V0_CUH
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V0_CUH


#include "../graph_generator.h"
#include <omp.h>



#ifdef __cplusplus
extern "C"
#endif

int cuda_bellman_ford_v0(Graph* graph, int source, int* dist, int *d_predecessor,int threads_per_block, double  *kernels_time);

#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V0_CUH
