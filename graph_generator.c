//
// Created by rick on 25/11/23.
//


#include "graph_generator.h"



Graph generate_complete_undirected_graph(int num_vertices, int lower_bound, int upper_bound){
    Graph graph;
    graph.num_vertices = num_vertices;
    graph.num_edges = num_vertices * (num_vertices - 1);

    graph.nodes = (int*)malloc(num_vertices * sizeof(int));
    graph.edges = (Edge*)malloc(num_vertices * (num_vertices - 1) * sizeof(Edge));
    graph.adjacency_matrix = (int**)malloc(num_vertices * sizeof(int*));

    for (int i = 0; i < num_vertices; i++)
        graph.nodes[i] = i;


    for (int i = 0; i < num_vertices; i++){
        graph.adjacency_matrix[i] = (int*)malloc(num_vertices * sizeof(int));
        for (int j = 0; j < num_vertices; j++)
            graph.adjacency_matrix[i][j] = 0;
    }

    int edgeCount = 0;

    for (int i = 0; i < num_vertices; i++){
        for (int j = i + 1; j < num_vertices; j++){
            int weight_ij = rand() % (upper_bound - lower_bound + 1) + lower_bound;
            int weight_ji = rand() % (upper_bound - lower_bound + 1) + lower_bound;

            graph.edges[edgeCount].origin = i;
            graph.edges[edgeCount].end = j;
            graph.edges[edgeCount].weight = weight_ij;
            graph.adjacency_matrix[i][j] = weight_ij;

            if (weight_ij > graph.maximum_weight)
                graph.maximum_weight = weight_ij;

            edgeCount++;

            graph.edges[edgeCount].origin = j;
            graph.edges[edgeCount].end = i;
            graph.edges[edgeCount].weight = weight_ji;
            graph.adjacency_matrix[j][i] = weight_ji;

            if (weight_ji > graph.maximum_weight)
                graph.maximum_weight = weight_ji;

            edgeCount++;
        }
    }

    return graph;
}


void free_graph(Graph* graph){
    free(graph->nodes);
    free(graph->edges);

    for (int i = 0; i < graph->num_vertices; i++){
        free(graph->adjacency_matrix[i]);
    }
    free(graph->adjacency_matrix);
}


void print_graph_adjacency_list(Graph *graph){
    printf("- Adjacency List:\n");
    for (int i = 0; i < graph->num_vertices; i++){
        printf("%d: ", graph->nodes[i]);

        for (int j = 0; j < graph->num_vertices; j++){
            if (graph->adjacency_matrix[i][j] != 0)
                printf("(%d, %d, %d) ", i, j, graph->adjacency_matrix[i][j]);
        }

        printf("\n");
    }
    printf("The graph has %d edges\n\n", graph->num_edges);

}


void print_graph_adjacency_matrix(Graph *graph){
    printf("- Adjacency Matrix:\n");
    for (int i = 0; i < graph->num_vertices; i++){
        for (int j = 0; j < graph->num_vertices; j++)
            printf("%d ", graph->adjacency_matrix[i][j]);

        printf("\n");
    }
    printf("\n");
}
