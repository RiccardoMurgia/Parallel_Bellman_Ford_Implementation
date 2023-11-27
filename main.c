//
// Created by rick on 24/11/23.
//


#include <stdio.h>
#include <omp.h>

#include <stdlib.h>
#include "graph_generator.h"
#include "serial_bellman_ford.h"
#include "openmp_bellman_ford_V0.h"
#include "openmp_bellman_ford_V1.h"


int main() {
    unsigned int seed = 42;
    srand(seed);
    omp_set_num_threads(20);  // Imposta il numero di thread a 4


    int numVertices, lower_bound, upper_bound;
    double start_time_serial, end_time_serial, start_time_v0, end_time_v0, start_time_v1, end_time_v1;

    int source = 0;


    // Input parameters
    printf("Enter the number of vertices: ");
    scanf("%d", &numVertices);

    printf("Enter the lower bound for random weights: ");
    scanf("%d", &lower_bound);

    printf("Enter the upper bound for random weights: ");
    scanf("%d", &upper_bound);

    // Generate and print the graph
    Graph myGraph = generate_complete_undirected_graph(numVertices, lower_bound, upper_bound);

    // Access the list of nodes in the main
    printf("List of Nodes: ");
    for (int i = 0; i < myGraph.num_vertices; i++) {
        printf("%d ", myGraph.nodes[i]);
    }
    printf("\n");

    // Print the adjacency list and matrix
    print_graph_adjacency_list(&myGraph);
    print_graph_adjacency_matrix(&myGraph);

    int* distances_v0 = (int*)malloc(numVertices * sizeof(int));
    int* distances_v1 = (int*)malloc(numVertices * sizeof(int));
    int* distances_serial = (int*)malloc(numVertices * sizeof(int));

    printf("======================================\n");
    printf("Serial solution:\n");
    printf("======================================\n");

    start_time_serial = omp_get_wtime();
    int negative_cycles_bellman_ford_serial = bellman_ford_serial(&myGraph, source, distances_serial);
    end_time_serial =  omp_get_wtime();

    if (negative_cycles_bellman_ford_serial) {
        printf("Negative cycle detected.\n");
    }
    else {
        printf("Shortest distances from node %d:\n", source);
        for (int i = 0; i < numVertices; i++) {
            printf("To node %d: %d\n", i, distances_serial[i]);
        }
    }

    printf("Bellman-Ford serial require: %f seconds\n\n", end_time_serial - start_time_serial);

    printf("======================================\n");
    printf("V0 solution:\n");
    printf("======================================\n");

    start_time_v0 = omp_get_wtime();
    int negative_cycles_bellman_ford_v0 = bellman_ford_v0(&myGraph, source, distances_v0);
    end_time_v0 = omp_get_wtime();

    if (negative_cycles_bellman_ford_v0) {
        printf("Negative cycle detected.\n");
    }
    else {
        printf("Shortest distances from node %d:\n", source);
        for (int i = 0; i < numVertices; i++) {
            printf("To node %d: %d\n", i, distances_v0[i]);
        }
    }
    printf("Bellman-Ford V0 require: %f seconds\n\n", end_time_v0 - start_time_v0);

    printf("======================================\n");
    printf("V1 solution:\n");
    printf("======================================\n");

    start_time_v1 = omp_get_wtime();
    int negative_cycles_bellman_ford_v1 = bellman_ford_v1(&myGraph, source, distances_v1);
    end_time_v1 = omp_get_wtime();

    if (negative_cycles_bellman_ford_v1) {
        printf("Negative cycle detected.\n");
    }
    else {
        printf("Shortest distances from node %d:\n", source);
        for (int i = 0; i < numVertices; i++) {
            printf("To node %d: %d\n", i, distances_v1[i]);
        }
    }

    printf("Bellman-Ford V1 require: %f seconds\n\n", end_time_v1 - start_time_v1);


    // Free allocated memory
    free(distances_serial);
    free(distances_v0);
    free(distances_v1);

    // Free allocated memory
    free_graph(&myGraph);

    return 0;
}
