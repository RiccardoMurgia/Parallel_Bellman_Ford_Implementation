//
// Created by rick on 28/11/23.
//

#include "stdio.h"
#include <math.h>

#include "utilities.h"



double calculate_mean(int num_tests, const double times[num_tests]) {
    double sum = 0.0;
    for (int i = 0; i < num_tests; i++) {
        sum += times[i];
    }
    return sum / num_tests;
}


double calculate_std_dev(int num_tests, const double times[num_tests], double mean) {
    double sum_squared_diff = 0.0;
    for (int i = 0; i < num_tests; i++) {
        double diff = times[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sqrt(sum_squared_diff / num_tests);
}


void print_version_results(const char* version_name, int* distances, int numVertices, int source, int has_negative_cycle, double execution_time) {
    printf("======================================\n");
    printf("%s solution:\n", version_name);
    printf("======================================\n");

    if (has_negative_cycle) {
        printf("Negative cycle detected.\n");
    } else {
        printf("Shortest distances from node %d:\n", source);
        for (int i = 0; i < numVertices; i++) {
            printf("To node %d: %d\n", i, distances[i]);
        }
    }

    printf("Bellman-Ford %s require: %f seconds\n\n", version_name, execution_time);
}


void print_time_matrix(const char* versions[], double* time_matrix, int number_of_version, int number_of_test) {
    printf("%-15s ", "Version");
    for (int j = 1; j <= number_of_test; j++) {
        char test_label[10];
        sprintf(test_label, "test%d", j);
        printf("%-15s ", test_label);
    }
    printf("\n");

    for (int i = 0; i < number_of_version; i++) {
        printf("%-15s ", versions[i]);
        for (int j = 0; j < number_of_test; j++) {
            printf("%-15f ", time_matrix[i * number_of_test + j]);
        }
        printf("\n");
    }
    printf("\n");
}


void print_statistics(const char* versions[], double time_matrix[], int number_of_version, int number_of_test) {
    printf("%-15s %-15s %-15s\n", "Version", "Mean", "Standard Deviation");

    for (int i = 0; i < number_of_version; i++) {
        double mean = calculate_mean(number_of_test, &time_matrix[i * number_of_test]);
        double std_dev = calculate_std_dev(number_of_test, &time_matrix[i * number_of_test], mean);

        printf("%-15s %-15f %-15f\n", versions[i], mean, std_dev);
    }
}