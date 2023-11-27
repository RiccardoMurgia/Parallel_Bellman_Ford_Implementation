//
// Created by rick on 25/11/23.
//

#include <stdio.h>
#include <stdlib.h>
#include "graph_generator.h"

// Function to generate a complete undirected graph with random weights
Graph generate_complete_undirected_graph(int numVertices, int lower_bound, int upper_bound) {
    Graph graph;
    graph.num_vertices = numVertices;
    graph.num_edges = numVertices * (numVertices - 1);

    // Allocate memory for nodes
    graph.nodes = (int*)malloc(numVertices * sizeof(int));
    for (int i = 0; i < numVertices; i++) {
        graph.nodes[i] = i;
    }

    // Allocate memory for edges
    graph.edges = (Edge*)malloc(numVertices * (numVertices - 1) * sizeof(Edge));

    // Allocate memory for adjacency matrix
    graph.adjacency_matrix = (int**)malloc(numVertices * sizeof(int*));
    for (int i = 0; i < numVertices; i++) {
        graph.adjacency_matrix[i] = (int*)malloc(numVertices * sizeof(int));
        for (int j = 0; j < numVertices; j++) {
            graph.adjacency_matrix[i][j] = 0;  // Initialize to 0
        }
    }

    int edgeCount = 0;

    // Add edges to create a complete graph
    for (int i = 0; i < numVertices; i++) {
        for (int j = i + 1; j < numVertices; j++) {
            int weight_ij = rand() % (upper_bound - lower_bound + 1) + lower_bound;
            int weight_ji = rand() % (upper_bound - lower_bound + 1) + lower_bound;

            // Add edge (i, j)
            graph.edges[edgeCount].origin = i;
            graph.edges[edgeCount].end = j;
            graph.edges[edgeCount].weight = weight_ij;
            graph.adjacency_matrix[i][j] = weight_ij;

            if (weight_ij > graph.maximum_weight)
                graph.maximum_weight = weight_ij;

            edgeCount++;

            // Add edge (j, i)
            graph.edges[edgeCount].origin = j;
            graph.edges[edgeCount].end = i;
            graph.edges[edgeCount].weight = weight_ji;
            graph.adjacency_matrix[j][i] = weight_ji;

            if (weight_ji > graph.maximum_weight) {
                graph.maximum_weight = weight_ji;
            }

            edgeCount++;

        }
    }

    return graph;
}

// Function to free allocated memory for a graph
void free_graph(Graph* graph) {
    free(graph->nodes);
    free(graph->edges);

    // Free memory for adjacency matrix
    for (int i = 0; i < graph->num_vertices; i++) {
        free(graph->adjacency_matrix[i]);
    }
    free(graph->adjacency_matrix);
}

// Function to print the graph as an adjacency list
void print_graph_adjacency_list(Graph* graph) {
    printf("- Adjacency List:\n");
    for (int i = 0; i < graph->num_vertices; i++) {
        printf("%d: ", graph->nodes[i]);

        for (int j = 0; j < graph->num_vertices; j++) {
            if (graph->adjacency_matrix[i][j] != 0) {
                printf("(%d, %d, %d) ", i, j, graph->adjacency_matrix[i][j]);
            }
        }
        printf("\n");
    }
    printf("The graph has %d edges\n\n", graph->num_edges);

}

// Function to print the graph as an adjacency matrix
void print_graph_adjacency_matrix(Graph* graph) {
    printf("- Adjacency Matrix:\n");
    for (int i = 0; i < graph->num_vertices; i++) {
        for (int j = 0; j < graph->num_vertices; j++) {
            printf("%d ", graph->adjacency_matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
