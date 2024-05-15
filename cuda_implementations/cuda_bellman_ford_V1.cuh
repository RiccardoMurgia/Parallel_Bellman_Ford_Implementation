//
// Created by rick on 26/04/24.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V1_CUH
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V1_CUH


#include "../graph_generator.h"


#ifdef __cplusplus
extern "C"
#endif

int cuda_bellman_ford_v1(Graph* graph, int source, int* dist, int threads_per_block);


#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_BELLMAN_FORD_V1_CUH
