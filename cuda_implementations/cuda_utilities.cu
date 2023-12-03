//
// Created by rick on 03/12/23.
//


#include "cuda_utilities.cuh"



__global__ void link_graph_device_pointers(Graph *d_graph, int *d_nodes, Edge *d_edges, int **d_adjacency_matrix) {
    d_graph->nodes = d_nodes;
    d_graph->edges = d_edges;
    d_graph->adjacency_matrix = d_adjacency_matrix;
}


__global__ void get_graph_device_pointers(Graph *d_graph, int **nodes, Edge **edges, int ***adjacency_matrix) {
    *nodes = d_graph->nodes;
    *edges = d_graph->edges;
    *adjacency_matrix = d_graph->adjacency_matrix;
}