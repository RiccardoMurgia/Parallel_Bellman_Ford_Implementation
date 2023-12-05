//
// Created by rick on 04/12/23.
//


#include "openmp_bellman_ford_V0_1.h"



void parallel_relax_edges_1(int *distances, Edge *edges, int num_edges){
    #pragma omp  for
        for (int i = 0; i < num_edges; i++) {
            int origin = edges[i].origin;
            int end = edges[i].end;
            int weight = edges[i].weight;

            if (distances[origin] + weight < distances[end]) {
                #pragma omp critical
                    distances[end] = distances[origin] + weight;
            }
        }
}



int bellman_ford_v0_1(Graph *graph, int source, int *dist){
    int negative_cycle = 0;

    #pragma omp parallel default(none) shared(graph, dist) firstprivate(source) reduction(+:negative_cycle)
        parallel_initialize_distances_1(dist, graph->num_vertices, source, graph->maximum_weight);

        for (int i = 0; i < graph->num_vertices - 1; i++)
                parallel_relax_edges_1(dist, graph->edges, graph->num_edges);

        #pragma omp for
            for (int i = 0; i < graph->num_edges; i++){
                int origin = graph->edges[i].origin;
                int end = graph->edges[i].end;
                int weight = graph->edges[i].weight;

                if (dist[origin] + weight < dist[end])
                    negative_cycle += 1;
            }

    return negative_cycle;
}
