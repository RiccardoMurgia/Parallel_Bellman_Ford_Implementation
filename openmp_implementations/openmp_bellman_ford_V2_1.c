//
// Created by rick on 04/12/23.
//


#include "openmp_bellman_ford_V2_1.h"



int bellman_ford_v2_1(Graph *graph, int source, int *dist, int *predecessor){
    int negative_cycles = 0;
    int *new_dist = (int*) malloc(graph->num_vertices * sizeof(int));
    int *candidate_dist = (int*) malloc(graph->num_vertices * graph->num_vertices * sizeof(int));


    #pragma omp parallel default(none) shared(graph, negative_cycles, dist, new_dist, predecessor) firstprivate(source, candidate_dist)
    {
        parallel_initialize_distances_1(dist, graph->num_vertices, source, graph->maximum_weight);


        for (int i = 1; i < graph->num_vertices; i++) {
            #pragma omp for
                for (int v = 0; v < graph->num_vertices; v++) {
                    #pragma omp parallel for default(none) shared(graph, dist, new_dist, predecessor) firstprivate(source, candidate_dist, v)
                    for (int u = 0; u < graph->num_vertices; u++)
                        candidate_dist[v * graph->num_vertices + u] = dist[u] + graph->adjacency_matrix[u][v];

                    MinResult min_candidate_dist = parallel_find_min_value(&candidate_dist[v * graph->num_vertices], graph->num_vertices);

                    if (min_candidate_dist.value < dist[v]) {
                        new_dist[v] = min_candidate_dist.value;
                        predecessor[v] = min_candidate_dist.index;
                    }
                    else
                        new_dist[v] = dist[v];

                }

            #pragma omp single
                memcpy(dist, new_dist, graph->num_vertices * sizeof(int));

        }

        #pragma omp for reduction(+:negative_cycles)
            for (int v = 0; v < graph->num_vertices; v++) {
                for (int u = 0; u < graph->num_vertices; u++) {
                    if (dist[u] + graph->adjacency_matrix[u][v] < dist[v])
                        negative_cycles += 1;
                }
            }
    }

    free(candidate_dist);
    free(new_dist);

    return negative_cycles;
}