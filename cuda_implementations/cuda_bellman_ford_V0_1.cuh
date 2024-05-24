//
// Created by rick on 03/12/23.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V0_1_CUH
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V0_1_CUH

#include "../graph_generator.h"
#include <omp.h>




#ifdef __cplusplus
extern "C"
#endif

int cuda_bellman_ford_v0_1(Graph* graph, int source, int* dist, int threads_per_block, double  *kernels_time);
#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V0_1_CUH
