//
// Created by rick on 25/11/23.
//


#include "openmp_bellman_ford_V0.h"



void parallel_relax_edges_0(int *distances, Edge *edges, int num_edges){
    #pragma omp parallel for default(none) shared(distances, edges, num_edges)
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


int bellman_ford_v0(Graph *graph, int source, int *dist){
    int negative_cycle = 0;
    parallel_initialize_distances_0(dist, graph->num_vertices, source, graph->maximum_weight);

    for (int i = 0; i < graph->num_vertices - 1; i++)
        parallel_relax_edges_0(dist, graph->edges, graph->num_edges);


    #pragma omp parallel for default(none) shared(graph, dist) firstprivate(source) reduction(+:negative_cycle)
        for (int i = 0; i < graph->num_edges; i++){
            int origin = graph->edges[i].origin;
            int end = graph->edges[i].end;
            int weight = graph->edges[i].weight;

            if (dist[origin] + weight < dist[end])
                negative_cycle += 1;
        }

    return negative_cycle;
}
