//
// Created by rick on 26/04/24.
//


#include "cuda_bellman_ford_V1.cuh"
#include "cuda_utilities.cuh"



__device__ MinResult d_find_min_value(const int *array, int num_vertices, unsigned int tid){
    MinResult result;
    result.value = INT_MAX;
    result.index = -1;
    unsigned int start_index = tid * num_vertices;
    unsigned int end_index = start_index + num_vertices;

    for (unsigned int i = start_index; i < end_index; i++){
        if (array[i] < result.value) {
            result.value = array[i];
            result.index = (int)i;
        }
    }

    return result;
}


__global__ void update_distances(const int *d_dist, Graph *d_graph, int *d_new_dist, int *d_candidate_dist){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < d_graph->num_vertices){

        for (int u = 0; u < d_graph->num_vertices; u++)
           d_candidate_dist[tid * d_graph->num_vertices + u] = d_dist[u] + d_graph->adjacency_matrix[u][tid];

        MinResult min_candidate_dist = d_find_min_value(d_candidate_dist, d_graph->num_vertices, tid);

        if (min_candidate_dist.value < d_dist[tid])
            d_new_dist[tid] = min_candidate_dist.value;
        else
            d_new_dist[tid] = d_dist[tid];

    }

}


extern "C" int cuda_bellman_ford_v1(Graph *graph, int source, int *dist, int threads_per_block){
    int num_blocks = (graph->num_vertices + threads_per_block - 1) / threads_per_block;
    int negative_cycles = 0;

    int *d_dist = nullptr;
    Graph *d_graph = nullptr;
    int *d_negative_cycles = nullptr;
    int *d_candidate_dist = nullptr;
    int *d_new_dist = nullptr;
    int *d_new_predecessor = nullptr;
    int *d_tmp;

    cudaMalloc((void **) &d_dist, sizeof(int) * graph->num_vertices);
    cudaMalloc((void **) &d_graph, sizeof(Graph));
    cudaMalloc((void **) &d_negative_cycles, sizeof(int));
    cudaMalloc((void **) &d_candidate_dist, sizeof(int) * graph->num_vertices * graph->num_vertices);
    cudaMalloc((void **) &d_new_dist, sizeof(int) * graph->num_vertices);
    cudaMalloc((void **) &d_new_predecessor, sizeof(int) * graph->num_vertices);

    int **gpu_adjacency_matrix_ptrs_2_free = copy_graph_2_GPU(graph, d_graph);
    cuda_initialize_distances<<<num_blocks, threads_per_block>>>(d_dist, d_graph, source);
    cudaMemcpy(d_negative_cycles, &negative_cycles, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    for (int i = 0; i < graph->num_vertices; i++){
        update_distances<<<num_blocks, threads_per_block>>>(d_dist, d_graph, d_new_dist, d_candidate_dist);
        cudaDeviceSynchronize();

        d_tmp = d_dist;
        d_dist = d_new_dist;
        d_new_dist = d_tmp;
    }


    detect_negative_cycle_1<<<num_blocks, threads_per_block>>>(d_dist, d_graph, d_negative_cycles, d_candidate_dist);
    cudaMemcpy(&negative_cycles, d_negative_cycles, sizeof(int), cudaMemcpyDeviceToHost);

    if(!negative_cycles)
        cudaMemcpy(dist, d_dist, sizeof(int) * graph->num_vertices, cudaMemcpyDeviceToHost);


    cudaFree(d_new_dist);
    cudaFree(d_new_predecessor);

    cudaFree(d_dist);
    freeGraph(d_graph, gpu_adjacency_matrix_ptrs_2_free, graph->num_vertices);
    cudaFree(d_negative_cycles);
    cudaFree(d_candidate_dist);

    return negative_cycles;
}
