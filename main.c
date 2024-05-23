//
// Created by rick on 24/11/23.
//


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "general_utilities.h"
#include "graph_generator.h"
#include "bellman_ford_Sq.h"

#include "openmp_implementations/openmp_bellman_ford_V0.h"
#include "openmp_implementations/openmp_bellman_ford_V0_1.h"
#include "openmp_implementations/openmp_bellman_ford_V1.h"
#include "openmp_implementations/openmp_bellman_ford_V1_1.h"
#include "openmp_implementations/openmp_bellman_ford_V2.h"
#include "openmp_implementations/openmp_bellman_ford_V2_1.h"

#include "cuda_implementations/cuda_bellman_ford_V0.cuh"
#include "cuda_implementations/cuda_bellman_ford_V0_1.cuh"
#include "cuda_implementations/cuda_bellman_ford_V1.cuh"
#include "cuda_implementations/cuda_bellman_ford_V1_1.cuh"


void open_mp_test_1(int num_vertices, int lower_bound, int upper_bound, int number_of_test, int maximum_n_thread,
                    int source){

    const char *versions[] = {"Sq", "V0", "V0_1", "V1", "V1_1", "V2", "V2_1"};
    int number_of_versions = sizeof(versions) / sizeof(versions[0]);;


    int *distances_sq = (int *) malloc(num_vertices * sizeof(int));

    int *openmp_distances_v0 = (int *) malloc(num_vertices * sizeof(int));
    int *openmp_distances_v0_1 = (int *) malloc(num_vertices * sizeof(int));

    int *openmp_distances_v1 = (int *) malloc(num_vertices * sizeof(int));
    int *openmp_distances_v1_1 = (int *) malloc(num_vertices * sizeof(int));

    int *openmp_distances_v2 = (int *) malloc(num_vertices * sizeof(int));
    int *openmp_distances_v2_1 = (int *) malloc(num_vertices * sizeof(int));

    int *parallel_distances[] = {openmp_distances_v0, openmp_distances_v0_1,
                                 openmp_distances_v1, openmp_distances_v1_1,
                                 openmp_distances_v2, openmp_distances_v2_1};

    double start_time_sq, end_time_sq;

    double openmp_start_time_v0, openmp_end_time_v0, openmp_start_time_v1, openmp_end_time_v1, openmp_start_time_v2, openmp_end_time_v2;
    double openmp_start_time_v0_1, openmp_end_time_v0_1, openmp_start_time_v1_1, openmp_end_time_v1_1, openmp_start_time_v2_1, openmp_end_time_v2_1;

    double *time_statistics = (double *) malloc(number_of_versions * (maximum_n_thread * 2) * sizeof(double));


    for(int n_thread=0; n_thread<maximum_n_thread; n_thread++){
        omp_set_num_threads(n_thread+1);

        double *time_matrix = (double *) malloc(number_of_versions * number_of_test * sizeof(double));
        for(int test_id=0; test_id<number_of_test; test_id++){
            srand(test_id + 1);

            Graph myGraph = generate_complete_undirected_graph(num_vertices, lower_bound, upper_bound);

            //Sequential Algorithm
            start_time_sq = omp_get_wtime();
            int negative_cycles_bellman_ford_sq = bellman_ford_serial(&myGraph, source, distances_sq);
            end_time_sq = omp_get_wtime();
            time_matrix[0 * number_of_test + test_id] = end_time_sq - start_time_sq;


            //OPENMP Parallel version working on the arcs
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v0 = omp_get_wtime();
            bellman_ford_v0(&myGraph, source, openmp_distances_v0);
            openmp_end_time_v0 = omp_get_wtime();
            time_matrix[1 * number_of_test + test_id] = openmp_end_time_v0 - openmp_start_time_v0;


            //Version that reuse threads
            openmp_start_time_v0_1 = omp_get_wtime();
            bellman_ford_v0_1(&myGraph, source, openmp_distances_v0_1);
            openmp_end_time_v0_1 = omp_get_wtime();
            time_matrix[2 * number_of_test + test_id] = openmp_end_time_v0_1 - openmp_start_time_v0_1;


            //OPENMP Parallel version working on the nodes
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v1 = omp_get_wtime();
            bellman_ford_v1(&myGraph, source, openmp_distances_v1);
            openmp_end_time_v1 = omp_get_wtime();
            time_matrix[3 * number_of_test + test_id] = openmp_end_time_v1 - openmp_start_time_v1;

            //Version that reuse threads
            openmp_start_time_v1_1 = omp_get_wtime();
            bellman_ford_v1_1(&myGraph, source, openmp_distances_v1_1);
            openmp_end_time_v1_1 = omp_get_wtime();
            time_matrix[4 * number_of_test + test_id] = openmp_end_time_v1_1 - openmp_start_time_v1_1;


            //OPENMP Parallel version working on the nodes and parallelize also the minimum candidate search
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v2 = omp_get_wtime();
            bellman_ford_v2(&myGraph, source, openmp_distances_v2);
            openmp_end_time_v2 = omp_get_wtime();
            time_matrix[5 * number_of_test + test_id] = openmp_end_time_v2 - openmp_start_time_v2;

            //Version that reuse threads at every relaxation
            openmp_start_time_v2_1 = omp_get_wtime();
            bellman_ford_v2_1(&myGraph, source, openmp_distances_v2_1);
            openmp_end_time_v2_1 = omp_get_wtime();
            time_matrix[6 * number_of_test + test_id] = openmp_end_time_v2_1 - openmp_start_time_v2_1;


            if (!negative_cycles_bellman_ford_sq)
                check_differences(distances_sq, parallel_distances, versions, number_of_versions - 1, num_vertices);


            free_graph(&myGraph);
        }

        for (int algorithm_id=0; algorithm_id<number_of_versions; algorithm_id++){
            double mean = calculate_mean(number_of_test, &time_matrix[algorithm_id * number_of_test]);
            double std_dev = calculate_std_dev(number_of_test, &time_matrix[algorithm_id * number_of_test], mean);
            time_statistics[algorithm_id * (maximum_n_thread * 2) + 2 * n_thread] = mean;
            time_statistics[algorithm_id*(maximum_n_thread * 2) + 2 * n_thread + 1] = std_dev;
        }
    }

    write_results_in_txt("output1.txt", versions, time_statistics, number_of_versions, maximum_n_thread, "TH_");

    free(distances_sq);

    free(openmp_distances_v0);
    free(openmp_distances_v0_1);
    free(openmp_distances_v1);
    free(openmp_distances_v1_1);
    free(openmp_distances_v2);
    free(openmp_distances_v2_1);

    free(time_statistics);

}


void open_mp_test_2(int initial_num_vertices, int lower_bound, int upper_bound, int number_of_test, int max_k, int source){

    const char *versions[] = {"Sq", "V0", "V0_1", "V1", "V1_1", "V2", "V2_1"};
    int number_of_versions = sizeof(versions) / sizeof(versions[0]);;

    double start_time_sq, end_time_sq;

    double openmp_start_time_v0, openmp_end_time_v0, openmp_start_time_v1, openmp_end_time_v1, openmp_start_time_v2, openmp_end_time_v2;
    double openmp_start_time_v0_1, openmp_end_time_v0_1, openmp_start_time_v1_1, openmp_end_time_v1_1, openmp_start_time_v2_1, openmp_end_time_v2_1;

    double *time_statistics = (double *) malloc(number_of_versions * (max_k * 2) * sizeof(double));

    for(int k=0; k<max_k; k++){
        omp_set_num_threads(k+1);

        double work_load_increment_factor = cbrt(k+1);
        int num_vertices = (int)round(work_load_increment_factor * initial_num_vertices);

        int *distances_sq = (int *) malloc(num_vertices * sizeof(int));

        int *openmp_distances_v0 = (int *) malloc(num_vertices * sizeof(int));
        int *openmp_distances_v0_1 = (int *) malloc(num_vertices * sizeof(int));

        int *openmp_distances_v1 = (int *) malloc(num_vertices * sizeof(int));
        int *openmp_distances_v1_1 = (int *) malloc(num_vertices * sizeof(int));

        int *openmp_distances_v2 = (int *) malloc(num_vertices * sizeof(int));
        int *openmp_distances_v2_1 = (int *) malloc(num_vertices * sizeof(int));

        int *parallel_distances[] = {openmp_distances_v0, openmp_distances_v0_1,
                                     openmp_distances_v1, openmp_distances_v1_1,
                                     openmp_distances_v2, openmp_distances_v2_1};


        double *time_matrix = (double *) malloc(number_of_versions * number_of_test * sizeof(double));

        for(int test_id=0; test_id<number_of_test; test_id++){
            srand(test_id + 1);


            Graph myGraph = generate_complete_undirected_graph(num_vertices, lower_bound, upper_bound);


            //Sequential Algorithm
            start_time_sq = omp_get_wtime();
            int negative_cycles_bellman_ford_sq = bellman_ford_serial(&myGraph, source, distances_sq);
            end_time_sq = omp_get_wtime();
            time_matrix[0 * number_of_test + test_id] = end_time_sq - start_time_sq;


            //OPENMP Parallel version working on the arcs
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v0 = omp_get_wtime();
            bellman_ford_v0(&myGraph, source, openmp_distances_v0);
            openmp_end_time_v0 = omp_get_wtime();
            time_matrix[1 * number_of_test + test_id] = openmp_end_time_v0 - openmp_start_time_v0;


            //Version that reuse threads
            openmp_start_time_v0_1 = omp_get_wtime();
            bellman_ford_v0_1(&myGraph, source, openmp_distances_v0_1);
            openmp_end_time_v0_1 = omp_get_wtime();
            time_matrix[2 * number_of_test + test_id] = openmp_end_time_v0_1 - openmp_start_time_v0_1;


            //OPENMP Parallel version working on the nodes
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v1 = omp_get_wtime();
            bellman_ford_v1(&myGraph, source, openmp_distances_v1);
            openmp_end_time_v1 = omp_get_wtime();
            time_matrix[3 * number_of_test + test_id] = openmp_end_time_v1 - openmp_start_time_v1;

            //Version that reuse threads
            openmp_start_time_v1_1 = omp_get_wtime();
            bellman_ford_v1_1(&myGraph, source, openmp_distances_v1_1);
            openmp_end_time_v1_1 = omp_get_wtime();
            time_matrix[4 * number_of_test + test_id] = openmp_end_time_v1_1 - openmp_start_time_v1_1;


            //OPENMP Parallel version working on the nodes and parallelize also the minimum candidate search
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v2 = omp_get_wtime();
            bellman_ford_v2(&myGraph, source, openmp_distances_v2);
            openmp_end_time_v2 = omp_get_wtime();
            time_matrix[5 * number_of_test + test_id] = openmp_end_time_v2 - openmp_start_time_v2;

            //Version that reuse threads at every relaxation
            openmp_start_time_v2_1 = omp_get_wtime();
            bellman_ford_v2_1(&myGraph, source, openmp_distances_v2_1);
            openmp_end_time_v2_1 = omp_get_wtime();
            time_matrix[6 * number_of_test + test_id] = openmp_end_time_v2_1 - openmp_start_time_v2_1;



            if (!negative_cycles_bellman_ford_sq)
                check_differences(distances_sq, parallel_distances, versions, number_of_versions - 1, initial_num_vertices);

            free_graph(&myGraph);

        }

        for (int algorithm_id=0; algorithm_id<number_of_versions; algorithm_id++){
            double mean = calculate_mean(number_of_test, &time_matrix[algorithm_id * number_of_test]);
            double std_dev = calculate_std_dev(number_of_test, &time_matrix[algorithm_id * number_of_test], mean);
            time_statistics[algorithm_id * (max_k * 2) + 2 * k] = mean;
            time_statistics[algorithm_id * (max_k * 2) + 2 * k + 1] = std_dev;
        }


        free(distances_sq);
        free(openmp_distances_v0);
        free(openmp_distances_v0_1);
        free(openmp_distances_v1);
        free(openmp_distances_v1_1);
        free(openmp_distances_v2);
        free(openmp_distances_v2_1);
        free(time_matrix);

    }

    write_results_in_txt("output2.txt", versions, time_statistics, number_of_versions, max_k, "K=");

    free(time_statistics);

}



int main(int argc, char *argv[]) {

    int graph_flag;
    int solutions_flag;
    int times_table_flag;
    int statistics_table_flag;

    int source = 0;

    int num_vertices, lower_bound, upper_bound;
    int number_of_test;

    double start_time_sq, end_time_sq;

    double openmp_start_time_v0, openmp_end_time_v0, openmp_start_time_v1, openmp_end_time_v1, openmp_start_time_v2, openmp_end_time_v2;
    double openmp_start_time_v0_1, openmp_end_time_v0_1, openmp_start_time_v1_1, openmp_end_time_v1_1, openmp_start_time_v2_1, openmp_end_time_v2_1;

    double cuda_start_time_v0, cuda_end_time_v0, cuda_start_time_v1, cuda_end_time_v1;
    double cuda_start_time_v0_1, cuda_end_time_v0_1, cuda_start_time_v1_1, cuda_end_time_v1_1;

    const char *versions[] = {"Sq", "V0", "V0_1", "V1", "V1_1", "V2", "V2_1", "cuda_V0", "cuda_V0_1", "cuda_V1", "cuda_V1_1"};
    int number_of_versions = sizeof(versions) / sizeof(versions[0]);;

    // Test using manual input atoi(argv[1])
    if (0 == 0){
        omp_set_num_threads(14);
        int thread_per_block = 1024;

        // Input parameters
        {
            printf("Enter the number of vertices: ");
            scanf("%d", &num_vertices);

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
        }

        double time_matrix[number_of_versions][number_of_test];


        for (int test_id = 0; test_id < number_of_test; test_id++) {

            int *distances_sq = (int *) malloc(num_vertices * sizeof(int));

            int *openmp_distances_v0 = (int *) malloc(num_vertices * sizeof(int));
            int *openmp_distances_v0_1 = (int *) malloc(num_vertices * sizeof(int));

            int *openmp_distances_v1 = (int *) malloc(num_vertices * sizeof(int));
            int *openmp_distances_v1_1 = (int *) malloc(num_vertices * sizeof(int));

            int *openmp_distances_v2 = (int *) malloc(num_vertices * sizeof(int));
            int *openmp_distances_v2_1 = (int *) malloc(num_vertices * sizeof(int));

            int *cuda_distances_v0 = (int *) malloc(num_vertices * sizeof(int));
            int *cuda_distances_v0_1 = (int *) malloc(num_vertices * sizeof(int));

            int *cuda_distances_v1 = (int *) malloc(num_vertices * sizeof(int));
            int *cuda_distances_v1_1 = (int *) malloc(num_vertices * sizeof(int));

            int *parallel_distances[] = {openmp_distances_v0, openmp_distances_v0_1,
                                         openmp_distances_v1, openmp_distances_v1_1,
                                         openmp_distances_v2, openmp_distances_v2_1,
                                         cuda_distances_v0, cuda_distances_v0_1,
                                         cuda_distances_v1, cuda_distances_v1_1};


            if (solutions_flag) {
                for (int i = 0; i < 100; i++)
                    printf("_");
                printf("\n");

                printf("Test number: %d \n", test_id);

                for (int i = 0; i < 100; i++)
                    printf("_");
                printf("\n");
            }

            //fix seed
            srand(test_id + 1);

            // Generate and print the graph
            Graph myGraph = generate_complete_undirected_graph(num_vertices, lower_bound, upper_bound);

            // print the graph
            if (graph_flag) {
                printf("List of Nodes: ");
                for (int i = 0; i < myGraph.num_vertices; i++) {
                    printf("%d ", myGraph.nodes[i]);
                }
                printf("\n");

                print_graph_adjacency_list(&myGraph);
                print_graph_adjacency_matrix(&myGraph);

            }

            //Sequential Algorithm
            start_time_sq = omp_get_wtime();
            int negative_cycles_bellman_ford_sq = bellman_ford_serial(&myGraph, source, distances_sq);
            end_time_sq = omp_get_wtime();
            time_matrix[0][test_id] = end_time_sq - start_time_sq;


            //OPENMP Parallel version working on the arcs
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v0 = omp_get_wtime();
            int negative_cycles_bellman_ford_v0 = bellman_ford_v0(&myGraph, source, openmp_distances_v0);
            openmp_end_time_v0 = omp_get_wtime();
            time_matrix[1][test_id] = openmp_end_time_v0 - openmp_start_time_v0;

            //Version that reuse threads
            openmp_start_time_v0_1 = omp_get_wtime();
            int negative_cycles_bellman_ford_v0_1 = bellman_ford_v0_1(&myGraph, source, openmp_distances_v0_1);
            openmp_end_time_v0_1 = omp_get_wtime();
            time_matrix[2][test_id] = openmp_end_time_v0_1 - openmp_start_time_v0_1;


            //OPENMP Parallel version working on the nodes
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v1 = omp_get_wtime();
            int negative_cycles_bellman_ford_v1 = bellman_ford_v1(&myGraph, source, openmp_distances_v1);
            openmp_end_time_v1 = omp_get_wtime();
            time_matrix[3][test_id] = openmp_end_time_v1 - openmp_start_time_v1;

            //Version that reuse threads
            openmp_start_time_v1_1 = omp_get_wtime();
            int negative_cycles_bellman_ford_v1_1 = bellman_ford_v1_1(&myGraph, source, openmp_distances_v1_1);
            openmp_end_time_v1_1 = omp_get_wtime();
            time_matrix[4][test_id] = openmp_end_time_v1_1 - openmp_start_time_v1_1;


            //OPENMP Parallel version working on the nodes and parallelize also the minimum candidate search
            //Version that destroys creates threads at every relaxation
            openmp_start_time_v2 = omp_get_wtime();
            int negative_cycles_bellman_ford_v2 = bellman_ford_v2(&myGraph, source, openmp_distances_v2);
            openmp_end_time_v2 = omp_get_wtime();
            time_matrix[5][test_id] = openmp_end_time_v2 - openmp_start_time_v2;

            //Version that reuse threads at every relaxation
            openmp_start_time_v2_1 = omp_get_wtime();
            int negative_cycles_bellman_ford_v2_1 = bellman_ford_v2_1(&myGraph, source, openmp_distances_v2_1);
            openmp_end_time_v2_1 = omp_get_wtime();
            time_matrix[6][test_id] = openmp_end_time_v2_1 - openmp_start_time_v2_1;



            //CUDA Parallel version working on the archs
            //Version that destroys creates threads at every relaxation
            cuda_start_time_v0 = omp_get_wtime();
            int cuda_negative_cycles_bellman_ford_v0 = cuda_bellman_ford_v0(&myGraph, source, cuda_distances_v0,
                                                                            thread_per_block); //massimo numero di thread per block 1024 su slurm
            cuda_end_time_v0 = omp_get_wtime();
            time_matrix[7][test_id] = cuda_end_time_v0 - cuda_start_time_v0;

            //Version that reuse threads
            cuda_start_time_v0_1 = omp_get_wtime();
            int cuda_negative_cycles_bellman_ford_v0_1 = cuda_bellman_ford_v0_1(&myGraph, source, cuda_distances_v0_1,
                                                                                thread_per_block);
            cuda_end_time_v0_1 = omp_get_wtime();
            time_matrix[8][test_id] = cuda_end_time_v0_1 - cuda_start_time_v0_1;

            //CUDA Parallel version working on the nodes
            //Version that destroys creates threads at every relaxation
            cuda_start_time_v1 = omp_get_wtime();
            int cuda_negative_cycles_bellman_ford_v1 = cuda_bellman_ford_v1(&myGraph, source, cuda_distances_v1,
                                                                            thread_per_block);
            cuda_end_time_v1 = omp_get_wtime();
            time_matrix[9][test_id] = cuda_end_time_v1 - cuda_start_time_v1;

            //Version that reuse threads
            cuda_start_time_v1_1 = omp_get_wtime();
            int cuda_negative_cycles_bellman_ford_v1_1 = cuda_bellman_ford_v1_1(&myGraph, source, cuda_distances_v1_1,
                                                                                thread_per_block);
            cuda_end_time_v1_1 = omp_get_wtime();
            time_matrix[10][test_id] = cuda_end_time_v1_1 - cuda_start_time_v1_1;

            if (solutions_flag) {
                print_version_results("Sq", distances_sq, num_vertices, source, negative_cycles_bellman_ford_sq,
                                      time_matrix[0][test_id]);

                print_version_results("V0", openmp_distances_v0, num_vertices, source, negative_cycles_bellman_ford_v0,
                                      time_matrix[1][test_id]);
                print_version_results("V0_1", openmp_distances_v0_1, num_vertices, source,
                                      negative_cycles_bellman_ford_v0_1, time_matrix[2][test_id]);

                print_version_results("V1", openmp_distances_v1, num_vertices, source, negative_cycles_bellman_ford_v1,
                                      time_matrix[3][test_id]);
                print_version_results("V1_1", openmp_distances_v1_1, num_vertices, source,
                                      negative_cycles_bellman_ford_v1_1, time_matrix[4][test_id]);

                print_version_results("V2", openmp_distances_v2, num_vertices, source, negative_cycles_bellman_ford_v2,
                                      time_matrix[5][test_id]);
                print_version_results("V2_1", openmp_distances_v2_1, num_vertices, source,
                                      negative_cycles_bellman_ford_v2_1, time_matrix[6][test_id]);


                print_version_results("cuda_V0", cuda_distances_v0, num_vertices, source,
                                      cuda_negative_cycles_bellman_ford_v0, time_matrix[7][test_id]);
                print_version_results("cuda_V0_1", cuda_distances_v0_1, num_vertices, source,
                                      cuda_negative_cycles_bellman_ford_v0_1, time_matrix[8][test_id]);

                print_version_results("cuda_V1", cuda_distances_v1, num_vertices, source,
                                      cuda_negative_cycles_bellman_ford_v1, time_matrix[9][test_id]);
                print_version_results("cuda_V1_1", cuda_distances_v1_1, num_vertices, source,
                                      cuda_negative_cycles_bellman_ford_v1_1, time_matrix[10][test_id]);
            }

            if (!negative_cycles_bellman_ford_sq)
                check_differences(distances_sq, parallel_distances, versions, number_of_versions - 1, num_vertices);

            free(distances_sq);

            free(openmp_distances_v0);
            free(openmp_distances_v0_1);
            free(openmp_distances_v1);
            free(openmp_distances_v1_1);
            free(openmp_distances_v2);
            free(openmp_distances_v2_1);

            free(cuda_distances_v0);
            free(cuda_distances_v0_1);
            free(cuda_distances_v1);
            free(cuda_distances_v1_1);


            free_graph(&myGraph);

        }

        if (times_table_flag)
            print_time_matrix(versions, (double *) time_matrix, number_of_versions, number_of_test);

        if (statistics_table_flag)
            print_statistics(versions, (double *) time_matrix, number_of_versions, number_of_test);
    }
    else{
        open_mp_test_1(200, 1, 100, 1, 3, 0);
        open_mp_test_2(100, 1, 100, 1, 3, 0);

    }

    return 0;
}


