//
// Created by rick on 03/12/23.
//


#include "cuda_bellman_ford_V0_1.cuh"
#include "cuda_utilities.cuh"
#include <cuda_runtime.h>



__global__ void cuda_parallel_relax_edges_1(int d_group_size,  int *d_dist, Graph *d_graph,
                                            volatile int *d_n_block_processed, volatile int *d_semaphore){
    unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int num_blocks = gridDim.x;

    for (int i = 1; i < d_graph->num_vertices; i++){

        for (int j = 0; j < d_group_size; j++){
            unsigned int g_index = tid * (d_group_size) + j;
            if (g_index< d_graph->num_edges){
                int origin = d_graph->edges[g_index].origin;
                int end = d_graph->edges[g_index].end;
                int weight = d_graph->edges[g_index].weight;

                atomicMin(&d_dist[end], d_dist[origin] + weight);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)  // The first thread of each block notifies the end of its work
            atomicAdd((int *)d_n_block_processed, 1);

        if (tid == 0) {
            while (*d_n_block_processed != num_blocks); // The Master wait for the others block have finished
            *d_n_block_processed = 0;
            *d_semaphore = (i + 1) % 2;

        }
        while (*d_semaphore != (i + 1) % 2); // The other thread wait for the master to change the semaphore value

    }
}


extern "C" int cuda_bellman_ford_v0_1(Graph *graph, int source, int *dist, int threads_per_block){
    int negative_cycles = 0;


    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    int n_multi_processors = deviceProp.multiProcessorCount;
    int num_blocks = n_multi_processors + 1;
    int group_size = 0;
    int total_n_threads;

    while (num_blocks > n_multi_processors) {
        group_size++;
        total_n_threads = (graph->num_vertices * graph->num_vertices + group_size - 1) / group_size; // parte intera superiore numero archi / g
        num_blocks = (total_n_threads + threads_per_block - 1) / threads_per_block;
    }


    int n_block_processed = 0;
    int semaphore = 1;

    int *d_dist = nullptr;
    Graph *d_graph = nullptr;
    volatile int *d_n_block_processed = nullptr;
    volatile int *d_semaphore = nullptr;
    int *d_negative_cycles = nullptr;

    cudaMalloc((void **) &d_dist, sizeof(int) * graph->num_vertices);
    cudaMalloc((void **) &d_graph, sizeof(Graph));
    cudaMalloc((void **) &d_n_block_processed, sizeof(int));
    cudaMalloc((void **) &d_semaphore, sizeof(int));
    cudaMalloc((void **) &d_negative_cycles, sizeof(int));

    int **gpu_adjacency_matrix_ptrs_2_free = copy_graph_2_GPU(graph, d_graph);
    cuda_initialize_distances<<<num_blocks, threads_per_block>>>(d_dist, d_graph, source);
    cudaMemcpy((void *)d_n_block_processed, (void *)&n_block_processed, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_semaphore, (void *)&semaphore, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative_cycles, &negative_cycles, sizeof(int), cudaMemcpyHostToDevice);


    cuda_parallel_relax_edges_1<<<num_blocks, threads_per_block>>>(group_size, d_dist, d_graph, d_n_block_processed, d_semaphore);


    detect_negative_cycle_0<<<num_blocks, threads_per_block>>>(d_dist, d_graph, d_negative_cycles);
    cudaMemcpy(&negative_cycles, d_negative_cycles, sizeof(int), cudaMemcpyDeviceToHost);
    if(!negative_cycles)
        cudaMemcpy(dist, d_dist, sizeof(int) * graph->num_vertices, cudaMemcpyDeviceToHost);


    cudaFree(d_dist);
    freeGraph(d_graph, gpu_adjacency_matrix_ptrs_2_free, graph->num_vertices);
    cudaFree((int *) d_semaphore);
    cudaFree((int *) d_n_block_processed);
    cudaFree(d_negative_cycles);

    return negative_cycles;
}