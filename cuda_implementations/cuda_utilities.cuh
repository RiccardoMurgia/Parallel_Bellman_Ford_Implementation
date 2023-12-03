//
// Created by rick on 03/12/23.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_UTILITIES_CUH
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_UTILITIES_CUH

#include "../graph_generator.h"

__global__ void link_graph_device_pointers(Graph *d_graph, int *d_nodes, Edge *d_edges, int **d_adjacency_matrix);
__global__ void get_graph_device_pointers(Graph *d_graph, int **nodes, Edge **edges, int ***adjacency_matrix);

#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_UTILITIES_CUH
