//
// Created by rick on 03/12/23.
//


#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_UTILITIES_CUH
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_UTILITIES_CUH


#include "../graph_generator.h"
#include "../general_utilities.h"



__global__ void link_graph_device_pointers(Graph *d_graph, int *d_nodes, Edge *d_edges, int **d_adjacency_matrix);
__global__ void get_graph_device_pointers(Graph *d_graph, int **nodes, Edge **edges, int ***adjacency_matrix);
__global__ void cuda_initialize_distances(int *d_dist, Graph *d_graph, int d_source);
__global__ void detect_negative_cycle_0(int *d_dist, Graph *d_graph, int *negative_cycle_flag);
__global__ void detect_negative_cycle_1(const int *d_dist, Graph *d_graph, int *negative_cycle_flag, int *d_candidate_dist);

void copy_edge_list_2_GPU(Edge **d_edges, Edge *h_edges, int num_edges);
int** copy_adjacency_matrix_2_GPU(int **hostMatrix, int ***deviceMatrix, int numRows, int numCols);
int** copy_graph_2_GPU(Graph *h_graph, Graph *d_graph);
void freeGraph(Graph *d_graph, int**gpu_pointers_to_free, int num_vertices);


#endif //PARALLEL_BELLMAN_FORD_IMPLEMENTATION_CUDA_UTILITIES_CUH
