//
// Created by rick on 03/12/23.
//


#include "cuda_bellman_ford_V0.cuh"
#include "cuda_utilities.cuh"



__global__ void cuda_parallel_relax_edges_0(int *d_dist, int *d_predecessor, Graph *d_graph){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d_graph->num_edges){

        int origin = d_graph->edges[tid].origin;
        int end = d_graph->edges[tid].end;
        int weight = d_graph->edges[tid].weight;

        int new_dist = d_dist[origin] + weight;

        int old_dist = atomicMin(&d_dist[end], new_dist);
        if (new_dist < old_dist)
            atomicExch(&d_predecessor[end], origin);

    }
}


extern "C" int cuda_bellman_ford_v0(Graph *graph, int source, int *dist, int *predecessor,int threads_per_block, double *kernels_time){
    int num_blocks = (graph->num_edges + threads_per_block - 1) / threads_per_block;
    int negative_cycles = 0;

    int *d_dist = nullptr;
    int *d_predecessor = nullptr;
    Graph *d_graph = nullptr;
    int *d_negative_cycles = nullptr;

    cudaMalloc((void **) &d_dist, sizeof(int) * graph->num_vertices);
    cudaMalloc((void **) &d_predecessor, sizeof(int) * graph->num_vertices);
    cudaMalloc((void **) &d_graph, sizeof(Graph));
    cudaMalloc((void **) &d_negative_cycles, sizeof(int));


    int **gpu_adjacency_matrix_ptrs_2_free = copy_graph_2_GPU(graph, d_graph);
    cudaMemcpy(d_negative_cycles, &negative_cycles, sizeof(int), cudaMemcpyHostToDevice);

    double start_time = omp_get_wtime();
    cuda_initialize_distances<<<num_blocks, threads_per_block>>>(d_dist, d_graph, source);
    cudaDeviceSynchronize();


    for (int i = 0; i < graph->num_vertices - 1; i++){
        cuda_parallel_relax_edges_0<<<num_blocks, threads_per_block>>>(d_dist, d_predecessor, d_graph);
        cudaDeviceSynchronize();
    }


    detect_negative_cycle_0<<<num_blocks, threads_per_block>>>(d_dist, d_graph, d_negative_cycles);
    cudaDeviceSynchronize();
    *kernels_time = omp_get_wtime() - start_time;

    cudaMemcpy(&negative_cycles, d_negative_cycles, sizeof(int), cudaMemcpyDeviceToHost);
    if(!negative_cycles) {
        cudaMemcpy(dist, d_dist, sizeof(int) * graph->num_vertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(predecessor, d_predecessor, sizeof(int) * graph->num_vertices, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_dist);
    cudaFree(d_predecessor);
    freeGraph(d_graph, gpu_adjacency_matrix_ptrs_2_free, graph->num_vertices);
    cudaFree(d_negative_cycles);

    return negative_cycles;
}
