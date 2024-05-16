//
// Created by rick on 04/12/23.
//

#include "openmp_bellman_ford_V1_1.h"



int bellman_ford_v1_1(Graph *graph, int source, int *dist){
    int negative_cycle = 0;
    int *predecessor = (int*) malloc(graph->num_vertices * sizeof(int));


    #pragma omp parallel default(none) shared(graph, dist) firstprivate(source) reduction(+:negative_cycle)
        parallel_initialize_distances_1(dist, graph->num_vertices, source, graph->maximum_weight);


    for (int i = 0; i < graph->num_vertices; i++){
        int *new_dist = (int*) malloc(graph->num_vertices * sizeof(int));
        int *new_predecessor = (int*) malloc(graph->num_vertices * sizeof(int));
        int *candidate_dist = (int*) malloc(graph->num_vertices * sizeof(int));

        #pragma omp for
            for (int v = 0; v < graph->num_vertices; v++) {

                for (int u = 0; u < graph->num_vertices; u++)
                    candidate_dist[u] = dist[u] + graph->adjacency_matrix[u][v];

                MinResult min_candidate_dist = find_min_value(candidate_dist, graph->num_vertices);

                if (min_candidate_dist.value < dist[v]){
                    new_dist[v] = min_candidate_dist.value;
                    new_predecessor[v] = min_candidate_dist.index;
                }
                else {
                    new_dist[v] = dist[v];
                    new_predecessor[v] = predecessor[v];
                }
        }

        memcpy(dist, new_dist, graph->num_vertices * sizeof(int));
        memcpy(predecessor, new_predecessor, graph->num_vertices * sizeof(int));

        free(new_dist);
        free(new_predecessor);
        free(candidate_dist);

        new_dist = NULL;
        new_predecessor = NULL;
    }

    #pragma omp  for
        for (int v = 0; v < graph->num_vertices; v++) {
            for (int u = 0; u < graph->num_vertices; u++) {
                if (dist[u] + graph->adjacency_matrix[u][v] < dist[v])
                    negative_cycle += 1;
            }
        }

    free(predecessor);

    return negative_cycle;
}
