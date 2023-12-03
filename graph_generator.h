//
// Created by rick on 25/11/23.
//

#ifndef GRAPH_GENERATOR_H
#define GRAPH_GENERATOR_H



typedef struct {
    int origin;
    int end;
    int weight;
} Edge;

typedef struct {
    int num_vertices;
    int num_edges;
    int *nodes;
    Edge *edges;
    int **adjacency_matrix;
    int maximum_weight;
} Graph;


Graph generate_complete_undirected_graph(int num_vertices, int lower_bound, int upper_bound);
void free_graph(Graph *graph);
void print_graph_adjacency_list(Graph *graph);
void print_graph_adjacency_matrix(Graph *graph);

#endif
