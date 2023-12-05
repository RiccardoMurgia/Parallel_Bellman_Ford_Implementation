//
// Created by rick on 28/11/23.
//


#include "openmp_bellman_ford_V2.h"



int bellman_ford_v2(Graph *graph, int source, int *dist){
    int negative_cycle = 0;
    parallel_initialize_distances_0(dist, graph->num_vertices, source, graph->maximum_weight);

    int *predecessor = (int*) malloc(graph->num_vertices * sizeof(int));

    for (int i = 0; i < graph->num_vertices; i++){
        int *new_dist = (int*) malloc(graph->num_vertices * sizeof(int));
        int *new_predecessor = (int*) malloc(graph->num_vertices * sizeof(int));

        #pragma omp parallel for default(none) shared(graph, dist, predecessor, new_dist, new_predecessor) firstprivate(source)

            for (int v = 0; v < graph->num_vertices; v++){
                int *candidate_dist = (int*) malloc(graph->num_vertices * sizeof(int));

                for (int u = 0; u < graph->num_vertices; u++)
                    candidate_dist[u] = dist[u] + graph->adjacency_matrix[u][v];

                MinResult min_candidate_dist = parallel_find_min_value_0(candidate_dist, graph->num_vertices);

                if (min_candidate_dist.value < dist[v]){
                    new_dist[v] = min_candidate_dist.value;
                    new_predecessor[v] = min_candidate_dist.index;
                } else {
                    new_dist[v] = dist[v];
                    new_predecessor[v] = predecessor[v];
                }
                free(candidate_dist);
            }

        #pragma omp single
                {
                    memcpy(dist, new_dist, graph->num_vertices * sizeof(int));
                    memcpy(predecessor, new_predecessor, graph->num_vertices * sizeof(int));

                    free(new_dist);
                    free(new_predecessor);

                    new_dist = NULL;
                    new_predecessor = NULL;
                }
    }

    #pragma omp parallel for default(none) shared(graph, dist) firstprivate(source) reduction(+:negative_cycle)
        for (int v = 0; v < graph->num_vertices; v++){
            for (int u = 0; u < graph->num_vertices; u++){
                if (dist[u] + graph->adjacency_matrix[u][v] < dist[v])
                    negative_cycle += 1;
            }
        }

    free(predecessor);

    return negative_cycle;
}
