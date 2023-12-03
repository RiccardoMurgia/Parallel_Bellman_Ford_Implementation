//
// Created by rick on 03/12/23.
//


#include "cuda_bellman_ford_V0.cuh"
#include "cuda_utilities.cuh"


__global__ void cuda_parallel_relax_edges(int *d_distances, Graph *d_graph){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d_graph->num_edges){

        int origin = d_graph->edges[tid].origin;
        int end = d_graph->edges[tid].end;
        int weight = d_graph->edges[tid].weight;

        if (d_distances[origin] + weight < d_distances[end])
            atomicMin(&d_distances[end], d_distances[origin] + weight);
    }
}


extern "C" int cuda_bellman_ford_v0(Graph *graph, int source, int *distances, int threads_per_block){
    int num_blocks = (graph->num_edges + threads_per_block - 1) / threads_per_block;

    Graph *d_graph;
    int *d_source;
    int *d_distances;

    cudaMalloc((void **) &d_graph, sizeof(Graph));
    int **gpu_adjacency_matrix_ptrs_2_free = copy_graph_2_GPU(graph, d_graph);


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

    cudaMemcpy(distances, d_distances, sizeof(int) * graph->num_vertices, cudaMemcpyDeviceToHost);

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
