//
// Created by rick on 03/12/23.
//


#include <iostream>
#include "cuda_utilities.cuh"



__global__ void link_graph_device_pointers(Graph *d_graph, int *d_nodes, Edge *d_edges, int **d_adjacency_matrix){
    d_graph->nodes = d_nodes;
    d_graph->edges = d_edges;
    d_graph->adjacency_matrix = d_adjacency_matrix;
}


__global__ void get_graph_device_pointers(Graph *d_graph, int **nodes, Edge **edges, int ***adjacency_matrix){
    *nodes = d_graph->nodes;
    *edges = d_graph->edges;
    *adjacency_matrix = d_graph->adjacency_matrix;

}


__global__ void cuda_initialize_distances(int *d_dist, Graph *d_graph, const int *d_source){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < d_graph->num_vertices)
        d_dist[tid] = (tid == *d_source) ? 0 : INT_MAX - d_graph->maximum_weight;

}


__global__ void detect_negative_cycle(int *d_dist, Graph *d_graph, int *negative_cycle_flag) {
    __shared__ bool cycle_detected;
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (threadIdx.x == 0)
        cycle_detected = false;

    __syncthreads();

    if (!cycle_detected && tid < d_graph->num_edges) {
        int origin = d_graph->edges[tid].origin;
        int end = d_graph->edges[tid].end;
        int weight = d_graph->edges[tid].weight;

        if (d_dist[origin] + weight < d_dist[end])
            cycle_detected = true;

    }

    __syncthreads();

    if (cycle_detected && threadIdx.x == 0)
        atomicExch(negative_cycle_flag, 1);

    if (cycle_detected)
        return;

}


void copy_edge_list_2_GPU(Edge **d_edges, Edge *h_edges, int num_edges){
    cudaMalloc((void **) d_edges, sizeof(Edge) * num_edges);
    for (int i=0 ; i<num_edges; i++)
        cudaMemcpy(&(*d_edges)[i], &h_edges[i], sizeof(Edge), cudaMemcpyHostToDevice);
}


int** copy_adjacency_matrix_2_GPU(int **hostMatrix, int ***deviceMatrix, int numRows, int numCols){
    cudaMalloc((void**)deviceMatrix, numRows * sizeof(int*));
    int** gpu_pointers_to_free = (int**) malloc(numRows * sizeof(int*));

    for (int i = 0; i < numRows; ++i){
        int* deviceRow;
        cudaMalloc((void**)&deviceRow, numCols * sizeof(int));
        cudaMemcpy(deviceRow, hostMatrix[i], numCols * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&((*deviceMatrix)[i]), &deviceRow, sizeof(int*), cudaMemcpyHostToDevice);
        gpu_pointers_to_free[i] = deviceRow;
    }

    return gpu_pointers_to_free;

}


int** copy_graph_2_GPU(Graph *h_graph, Graph *d_graph) {
    int *d_nodes;
    Edge *d_edges;
    int **d_adjacency_matrix;

    cudaMalloc((void**) &(d_nodes), sizeof(int) * h_graph->num_vertices);
    cudaMemcpy(d_nodes, h_graph->nodes, sizeof(int) * h_graph->num_vertices, cudaMemcpyHostToDevice);

    copy_edge_list_2_GPU(&d_edges, h_graph->edges, h_graph->num_edges);
    int **gpu_row_ptr_2_free = copy_adjacency_matrix_2_GPU(h_graph->adjacency_matrix,
                                                           &d_adjacency_matrix,
                                                           h_graph->num_vertices,
                                                           h_graph->num_vertices);

    cudaMemcpy(d_graph, h_graph, sizeof(Graph), cudaMemcpyHostToDevice);

    link_graph_device_pointers<<<1, 1>>>(d_graph, d_nodes, d_edges, d_adjacency_matrix);
    cudaDeviceSynchronize();

    return gpu_row_ptr_2_free;
}


void freeGraph(Graph *d_graph, int**gpu_pointers_to_free, int num_vertices) {
    int **nodes = nullptr;
    Edge **edges = nullptr;
    int ***adjacency_matrix = nullptr;

    cudaMallocManaged((void**) &nodes,  sizeof(int*));
    cudaMallocManaged((void**) &edges,  sizeof(Edge*));
    cudaMallocManaged((void**) &adjacency_matrix,  sizeof(int**));

    get_graph_device_pointers<<<1, 1>>>(d_graph, nodes, edges, adjacency_matrix);
    cudaDeviceSynchronize();

    cudaFree(nodes);
    cudaFree(edges);

    for (int i=0; i<num_vertices; i++)
        cudaFree(gpu_pointers_to_free[i]);

    cudaFree(adjacency_matrix);
    cudaFree(d_graph);

}
