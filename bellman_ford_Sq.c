//
// Created by rick on 25/11/23.
//


#include "bellman_ford_Sq.h"



void initialize_distances(int *distances, int num_vertices, int source, int maximum_weight){
    for (int i = 0; i < num_vertices; i++)
        distances[i] = (i == source) ? 0 : INT_MAX - maximum_weight;
}


void serial_relax_edges(int *dist, int *predecessor, Edge *edges, int num_edges, int num_vertices){
    int *new_dist = (int*) malloc(num_vertices * sizeof(int));
    memcpy(new_dist, dist, num_vertices * sizeof(int));

    for (int i = 0; i < num_edges; i++){
        int origin = edges[i].origin;
        int end = edges[i].end;
        int weight = edges[i].weight;

        if (dist[origin] + weight < new_dist[end]) {
            new_dist[end] = dist[origin] + weight;
            predecessor[end] = origin;
        }
    }
    memcpy(dist, new_dist, num_vertices * sizeof(int));

}


int bellman_ford_serial(Graph *graph, int source, int *distances, int *predecessor){
    initialize_distances(distances, graph->num_vertices, source, graph->maximum_weight);

    for (int i = 0; i < graph->num_vertices - 1; i++)
        serial_relax_edges(distances, predecessor, graph->edges, graph->num_edges, graph->num_vertices);

    for (int i = 0; i < graph->num_edges; i++){
        int origin = graph->edges[i].origin;
        int end = graph->edges[i].end;
        int weight = graph->edges[i].weight;

        if (distances[origin] + weight < distances[end])
            return 1;
    }
    return 0;
}
