//
// Created by rick on 28/11/23.
//


#include "openmp_bellman_ford_V2.h"



int bellman_ford_v2(Graph *graph, int source, int *dist){
    int negative_cycles = 0;
    parallel_initialize_distances_0(dist, graph->num_vertices, source, graph->maximum_weight);
    int *new_dist = (int*) malloc(graph->num_vertices * sizeof(int));
    int *candidate_dist = (int*) malloc(graph->num_vertices * graph->num_vertices * sizeof(int));


    for (int i = 0; i < graph->num_vertices - 1; i++){

        #pragma omp parallel for default(none) shared(graph, dist, new_dist) firstprivate(source, candidate_dist)

            for (int v = 0; v < graph->num_vertices; v++){

                for (int u = 0; u < graph->num_vertices; u++)
                    candidate_dist[v * graph->num_vertices + u] = dist[u] + graph->adjacency_matrix[u][v];

                MinResult min_candidate_dist = parallel_find_min_value(&candidate_dist[v * graph->num_vertices],
                                                                       graph->num_vertices);

                if (min_candidate_dist.value < dist[v])
                    new_dist[v] = min_candidate_dist.value;
                else
                    new_dist[v] = dist[v];

            }

        #pragma omp single
            memcpy(dist, new_dist, graph->num_vertices * sizeof(int));

    }

    #pragma omp parallel for default(none) shared(graph, dist) firstprivate(source) reduction(+:negative_cycles)
        for (int v = 0; v < graph->num_vertices; v++){
            for (int u = 0; u < graph->num_vertices; u++){
                if (dist[u] + graph->adjacency_matrix[u][v] < dist[v])
                    negative_cycles += 1;
            }
        }

    free(candidate_dist);
    free(new_dist);


    return negative_cycles;
}
