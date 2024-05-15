//
// Created by rick on 03/12/23.
//


#include "cuda_bellman_ford_V0.cuh"
#include "cuda_utilities.cuh"



__global__ void cuda_parallel_relax_edges_0(int *d_dist, Graph *d_graph){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d_graph->num_edges){

        int origin = d_graph->edges[tid].origin;
        int end = d_graph->edges[tid].end;
        int weight = d_graph->edges[tid].weight;

        atomicMin(&d_dist[end], d_dist[origin] + weight);
    }
}


extern "C" int cuda_bellman_ford_v0(Graph *graph, int source, int *dist, int threads_per_block){
    int num_blocks = (graph->num_edges + threads_per_block - 1) / threads_per_block;
    int negative_cycles = 0;

    int *d_dist = nullptr;
    Graph *d_graph = nullptr;
    int *d_negative_cycles = nullptr;

    cudaMalloc((void **) &d_dist, sizeof(int) * graph->num_vertices);
    cudaMalloc((void **) &d_graph, sizeof(Graph));
    cudaMalloc((void **) &d_negative_cycles, sizeof(int));


    int **gpu_adjacency_matrix_ptrs_2_free = copy_graph_2_GPU(graph, d_graph);
    cuda_initialize_distances<<<num_blocks, threads_per_block>>>(d_dist, d_graph, source);
    cudaMemcpy(d_negative_cycles, &negative_cycles, sizeof(int), cudaMemcpyHostToDevice);


    for (int i = 0; i < graph->num_vertices - 1; i++){
        cuda_parallel_relax_edges_0<<<num_blocks, threads_per_block>>>(d_dist, d_graph);
        cudaDeviceSynchronize();
    }


    detect_negative_cycle_0<<<num_blocks, threads_per_block>>>(d_dist, d_graph, d_negative_cycles);
    cudaMemcpy(&negative_cycles, d_negative_cycles, sizeof(int), cudaMemcpyDeviceToHost);
    if(!negative_cycles)
        cudaMemcpy(dist, d_dist, sizeof(int) * graph->num_vertices, cudaMemcpyDeviceToHost);


    cudaFree(d_dist);
    freeGraph(d_graph, gpu_adjacency_matrix_ptrs_2_free, graph->num_vertices);
    cudaFree(d_negative_cycles);

    return negative_cycles;
}
