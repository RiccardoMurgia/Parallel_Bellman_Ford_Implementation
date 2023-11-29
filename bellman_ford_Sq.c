//
// Created by rick on 25/11/23.
//

#include <limits.h>

#include "graph_generator.h"



void initialize_distances(int* distances, int numVertices, int source, int maximum_weight) {
    for (int i = 0; i < numVertices; i++) {
        distances[i] = (i == source) ? 0 : INT_MAX - maximum_weight;
    }
}


void serial_relax_edges(int* distances, Edge* edges, int numEdges) {
    for (int i = 0; i < numEdges; i++) {
        int origin = edges[i].origin;
        int end = edges[i].end;
        int weight = edges[i].weight;

        if (distances[origin] != INT_MAX && distances[origin] + weight < distances[end]) {
            distances[end] = distances[origin] + weight;
        }
    }
}


int bellman_ford_serial(Graph* graph, int source, int* distances) {
    initialize_distances(distances, graph->num_vertices, source, graph->maximum_weight);

    for (int i = 0; i < graph->num_vertices - 1; i++) {
        serial_relax_edges(distances, graph->edges, graph->num_edges);
    }

    // Check for negative cycles
    for (int i = 0; i < graph->num_edges; i++) {
        int origin = graph->edges[i].origin;
        int end = graph->edges[i].end;
        int weight = graph->edges[i].weight;

        if (distances[origin] != INT_MAX && distances[origin] + weight < distances[end]) {
            return 1; // Negative cycle detected
        }
    }

    return 0; // No negative cycle
}
