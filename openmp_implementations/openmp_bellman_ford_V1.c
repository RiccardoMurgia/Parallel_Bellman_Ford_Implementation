//
// Created by rick on 25/11/23.
//


#include "openmp_bellman_ford_V1.h"



int bellman_ford_v1(Graph *graph, int source, int *dist){
    int negative_cycles = 0;
    parallel_initialize_distances_0(dist, graph->num_vertices, source, graph->maximum_weight);


    for (int i = 0; i < graph->num_vertices; i++){
        int *new_dist = (int*) malloc(graph->num_vertices * sizeof(int));

        #pragma omp parallel for default(none) shared(graph, dist, new_dist)
            for (int v = 0; v < graph->num_vertices; v++) {
                int *candidate_dist = malloc(graph->num_vertices * sizeof(int));

                for (int u = 0; u < graph->num_vertices; u++)
                    candidate_dist[u] = dist[u] + graph->adjacency_matrix[u][v];

                MinResult min_candidate_dist = find_min_value(candidate_dist, graph->num_vertices);

                if (min_candidate_dist.value < dist[v]){
                    new_dist[v] = min_candidate_dist.value;
                }
                else
                    new_dist[v] = dist[v];

                free(candidate_dist);
            }

            memcpy(dist, new_dist, graph->num_vertices * sizeof(int));

            free(new_dist);

            new_dist = NULL;
    }

    #pragma omp parallel for default(none) shared(graph, dist) firstprivate(source) reduction(+:negative_cycles)
            for (int v = 0; v < graph->num_vertices; v++) {
                for (int u = 0; u < graph->num_vertices; u++) {
                    if (dist[u] + graph->adjacency_matrix[u][v] < dist[v])
                        negative_cycles += 1;
                }
            }


    return negative_cycles;
}
