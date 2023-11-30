//
// Created by rick on 24/11/23.
//


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "utilities.h"
#include "graph_generator.h"
#include "bellman_ford_Sq.h"
#include "openmp_bellman_ford_V0.h"
#include "openmp_bellman_ford_V1.h"
#include "openmp_bellman_ford_V2.h"

#include "cuda_bellman_ford_V0.cuh"


int areArraysEqual(int *arr1, int *arr2, int size) {
    for (int i = 0; i < size; ++i) {
        if (arr1[i] != arr2[i]) {
            return 0;  // Gli array non sono uguali
        }
    }
    return 1;  // Gli array sono uguali
}



int main() {
    omp_set_num_threads(20);
    helloCUDA();



    int graph_flag;
    int solutions_flag;
    int times_table_flag;
    int statistics_table_flag;

    int source = 0;

    int numVertices, lower_bound, upper_bound;
    int number_of_test;

    double start_time_serial, end_time_serial, start_time_v0, end_time_v0, start_time_v1, end_time_v1, start_time_v2, end_time_v2;

    const char* versions[] = {"Sq","V0", "V1", "V2"};
    int number_of_version_to_test = 4;

    // Input parameters
    printf("Enter the number of vertices: ");
    scanf("%d", &numVertices);

    printf("Enter the lower bound for random weights: ");
    scanf("%d", &lower_bound);

    printf("Enter the upper bound for random weights: ");
    scanf("%d", &upper_bound);

    printf("Enter number of tests: ");
    scanf("%d", &number_of_test);

    printf("Enter 1 if you want see the generated graph 0 otherwise: ");
    scanf("%d", &graph_flag);

    printf("Enter 1 if you want the solutions and times for each version 0 otherwise: ");
    scanf("%d", &solutions_flag);

    printf("Enter 1 if you want a summary time table 0 otherwise: ");
    scanf("%d", &times_table_flag);

    printf(("Enter 1 if you want a statistic table 0 otherwise: "));
    scanf("%d", &statistics_table_flag);


    double time_matrix[4][number_of_test];


    for(int test_id = 0; test_id <  number_of_test; test_id++) {

        int *distances_serial = (int *) malloc(numVertices * sizeof(int));
        int *distances_v0 = (int *) malloc(numVertices * sizeof(int));
        int *distances_v1 = (int *) malloc(numVertices * sizeof(int));
        int *distances_v2 = (int *) malloc(numVertices * sizeof(int));


        if(solutions_flag) {
            for (int i = 0 ;i < 100;i++)
                printf("_");
            printf("\n");

            printf("Test number: %d \n", test_id);

            for (int i = 0 ;i < 100;i++)
                printf("_");
            printf("\n");
        }

        //fix seed
        srand(test_id);

        // Generate and print the graph
        Graph myGraph = generate_complete_undirected_graph(numVertices, lower_bound, upper_bound);

        // print the graph
        if(graph_flag) {
            printf("List of Nodes: ");
            for (int i = 0; i < myGraph.num_vertices; i++) {
                printf("%d ", myGraph.nodes[i]);
            }
            printf("\n");

            // Print the adjacency list and matrix
            print_graph_adjacency_list(&myGraph);
            print_graph_adjacency_matrix(&myGraph);

        }


        start_time_serial = omp_get_wtime();
        int negative_cycles_bellman_ford_sq = bellman_ford_serial(&myGraph, source, distances_serial);
        end_time_serial = omp_get_wtime();
        time_matrix[0][test_id] = end_time_serial - start_time_serial;


        start_time_v0 = omp_get_wtime();
        int negative_cycles_bellman_ford_v0 = bellman_ford_v0(&myGraph, source, distances_v0);
        end_time_v0 = omp_get_wtime();
        time_matrix[1][test_id] = end_time_v0 - start_time_v0;


        start_time_v1 = omp_get_wtime();
        int negative_cycles_bellman_ford_v1 = bellman_ford_v1(&myGraph, source, distances_v1);
        end_time_v1 = omp_get_wtime();
        time_matrix[2][test_id] = end_time_v1 - start_time_v1;


        start_time_v2 = omp_get_wtime();
        int negative_cycles_bellman_ford_v2 = bellman_ford_v2(&myGraph, source, distances_v2);
        end_time_v2 = omp_get_wtime();
        time_matrix[3][test_id] = end_time_v2 - start_time_v2;


        if(solutions_flag) {
            print_version_results("Sq", distances_serial, numVertices, source, negative_cycles_bellman_ford_sq, time_matrix[0][test_id]);
            print_version_results("V0", distances_v0, numVertices, source, negative_cycles_bellman_ford_v0, time_matrix[1][test_id]);
            print_version_results("V1", distances_v1, numVertices, source, negative_cycles_bellman_ford_v1, time_matrix[2][test_id]);
            print_version_results("V2", distances_v2, numVertices, source, negative_cycles_bellman_ford_v2, time_matrix[3][test_id]);
        }

        printf("porcodio %d\n",areArraysEqual(distances_serial, distances_v0,  numVertices));


        free(distances_serial);
        free(distances_v0);
        free(distances_v1);
        free(distances_v2);
        free_graph(&myGraph);
    }


    if(times_table_flag)
        print_time_matrix(versions, (double *) time_matrix, number_of_version_to_test, number_of_test);

    if(statistics_table_flag)
    print_statistics(versions, (double *) time_matrix, number_of_version_to_test, number_of_test);




    return 0;
}
