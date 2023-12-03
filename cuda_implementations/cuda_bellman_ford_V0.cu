//
// Created by rick on 03/12/23.
//

#include <cstdio>
#include "cuda_bellman_ford_V0.cuh"



__global__ void cuda_initialize_distances(int *distances, Graph *d_graph, const int *d_source){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d_graph->num_vertices) {
        distances[tid] = (tid == *d_source) ? 0 : INT_MAX - d_graph->maximum_weight;
    }
}



__global__ void cuda_parallel_relax_edges(int *d_distances, Graph *d_graph){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d_graph->num_edges) {
        int origin = d_graph->edges[tid].origin;
        int end = d_graph->edges[tid].end;
        int weight = d_graph->edges[tid].weight;

        if (d_distances[origin] + weight < d_distances[end])
            atomicMin(&d_distances[end], d_distances[origin] + weight);
    }
}



void copy_edge_list_2_GPU(Edge **d_edges, Edge *h_edges, int num_edges){
    cudaMalloc((void **) d_edges, sizeof(Edge) * num_edges);
    for (int i=0 ; i<num_edges; i++)
        cudaMemcpy(&(*d_edges)[i], &h_edges[i], sizeof(Edge), cudaMemcpyHostToDevice);
}



int** copy_adjacency_matrix_2_GPU(int **hostMatrix, int ***deviceMatrix, int numRows, int numCols) {
    cudaMalloc((void**)deviceMatrix, numRows * sizeof(int*));
    int** gpu_pointers_to_free = (int**) malloc(numRows * sizeof(int*));

    for (int i = 0; i < numRows; ++i) {
        int* deviceRow;
        cudaMalloc((void**)&deviceRow, numCols * sizeof(int));
        cudaMemcpy(deviceRow, hostMatrix[i], numCols * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&((*deviceMatrix)[i]), &deviceRow, sizeof(int*), cudaMemcpyHostToDevice);
        gpu_pointers_to_free[i] = deviceRow;
    }

    return gpu_pointers_to_free;
}



int** copy_graph_2_GPU(Graph *h_graph, Graph* d_graph) {
    int *d_nodes;
    Edge *d_edges;
    int **d_adjacency_matrix;

    cudaMalloc((void **) &(d_nodes), sizeof(int) * h_graph->num_vertices);
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



void freeGraph(Graph *d_graph, int **gpu_pointers_to_free, int num_vertices) {
    int* nodes;
    Edge* edges;
    int** adjacency_matrix;

    get_graph_device_pointers<<<1, 1>>>(d_graph, &nodes, &edges, &adjacency_matrix);
    cudaDeviceSynchronize();

    cudaFree(nodes);
    cudaFree(edges);
    for (int i=0; i<num_vertices; i++)
        cudaFree(gpu_pointers_to_free[i]);

    cudaFree(adjacency_matrix);
    cudaFree(d_graph);


}



extern "C" int cuda_bellman_ford_v0(Graph *graph, int source, int *distances, int threads_per_block){
    int num_blocks = (graph->num_edges + threads_per_block -1) / threads_per_block;

    Graph *d_graph;
    int *d_source;
    int *d_distances;

    cudaMalloc((void **) &d_graph, sizeof(Graph));
    int** gpu_adjacency_matrix_ptrs_2_free = copy_graph_2_GPU(graph, d_graph);


     cudaMalloc((void **) &d_source, sizeof(int));
     cudaMemcpy(d_source, &source, sizeof(int), cudaMemcpyHostToDevice);

     cudaMalloc((void **) &d_distances, sizeof(int) * graph->num_vertices);
     cudaMemcpy(d_distances, distances, sizeof(int) * graph->num_vertices, cudaMemcpyHostToDevice);

     cuda_initialize_distances<<<num_blocks, threads_per_block>>>(d_distances, d_graph, d_source);
     cudaDeviceSynchronize();

     for (int i = 0; i < graph->num_vertices - 1; i++) {
         cuda_parallel_relax_edges<<<num_blocks, threads_per_block>>>(d_distances, d_graph);
         cudaDeviceSynchronize();
     }

     cudaMemcpy(distances, d_distances, sizeof(int) * graph->num_vertices , cudaMemcpyDeviceToHost);

     freeGraph(d_graph, gpu_adjacency_matrix_ptrs_2_free, graph->num_vertices);
     cudaFree(d_source);
     cudaFree(d_distances);


     for (int i = 0; i < graph->num_edges; i++) {
         int origin = graph->edges[i].origin;
         int end = graph->edges[i].end;
         int weight = graph->edges[i].weight;

         if (distances[origin] + weight < distances[end])
             return 1;
     }

    return 0;
}


