//
// Created by rick on 03/12/23.
//


#include "cuda_bellman_ford_V0.cuh"
#include "cuda_utilities.cuh"


__global__ void cuda_parallel_relax_edges(int *d_distances, Graph *d_graph){
    //__shared__ int min_candidate_list[d_graph->num_vertices];

    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d_graph->num_edges){

        int origin = d_graph->edges[tid].origin;
        int end = d_graph->edges[tid].end;
        int weight = d_graph->edges[tid].weight;

        if (d_distances[origin] + weight < d_distances[end])                 //fixme  check if resynchronization thread is needed
            atomicMin(&d_distances[end], d_distances[origin] + weight);
    }
}

__global__ void detect_negative_cycle(int *d_distances, Graph *d_graph, int *negative_cycle_flag) {
    __shared__ bool cycle_detected;
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (threadIdx.x == 0)
        cycle_detected = false;

    __syncthreads();

    if (!cycle_detected && tid < d_graph->num_edges) {
        int origin = d_graph->edges[tid].origin;
        int end = d_graph->edges[tid].end;
        int weight = d_graph->edges[tid].weight;

        if (d_distances[origin] + weight < d_distances[end])
            cycle_detected = true;

    }

    __syncthreads();

    if (cycle_detected && threadIdx.x == 0)
        atomicExch(negative_cycle_flag, 1);

    if (cycle_detected)
        return;

}


extern "C" int cuda_bellman_ford_v0(Graph *graph, int source, int *dist, int threads_per_block){
    int num_blocks = (graph->num_edges + threads_per_block - 1) / threads_per_block;
    int negative_cycles = 0;

    int *d_source = nullptr;
    int *d_dist = nullptr;
    Graph *d_graph = nullptr;
    int *d_negative_cycles = nullptr;

    cudaMalloc((void **) &d_source, sizeof(int));
    cudaMalloc((void **) &d_dist, sizeof(int) * graph->num_vertices);
    cudaMalloc((void **) &d_graph, sizeof(Graph));
    cudaMalloc((void **) &d_negative_cycles, sizeof(int));


    int **gpu_adjacency_matrix_ptrs_2_free = copy_graph_2_GPU(graph, d_graph);
    cudaMemcpy(d_source, &source, sizeof(int), cudaMemcpyHostToDevice);
    cuda_initialize_distances<<<num_blocks, threads_per_block>>>(d_dist, d_graph, d_source);
    cudaMemcpy(d_negative_cycles, &negative_cycles, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    for (int i = 0; i < graph->num_vertices - 1; i++){
        cuda_parallel_relax_edges<<<num_blocks, threads_per_block>>>(d_dist, d_graph);
        cudaDeviceSynchronize();
    }

    detect_negative_cycle<<<num_blocks, threads_per_block>>>(d_dist, d_graph, d_negative_cycles);

    cudaMemcpy(&negative_cycles, d_negative_cycles, sizeof(int), cudaMemcpyDeviceToHost);

    if(!negative_cycles)
        cudaMemcpy(dist, d_dist, sizeof(int) * graph->num_vertices, cudaMemcpyDeviceToHost);

    cudaFree(d_source);
    cudaFree(d_dist);
    freeGraph(d_graph, gpu_adjacency_matrix_ptrs_2_free, graph->num_vertices);
    cudaFree(d_negative_cycles);

    return negative_cycles;
}
