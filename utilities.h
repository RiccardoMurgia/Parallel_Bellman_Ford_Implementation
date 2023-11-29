//
// Created by rick on 28/11/23.
//

#ifndef PARALLEL_BELLMAN_FORD_IMPLEMENTATION_UTILITIES_H
#define PARALLEL_BELLMAN_FORD_IMPLEMENTATION_UTILITIES_H


double calculate_mean(int num_tests, const double times[num_tests]);
double calculate_std_dev(int num_tests, const double times[num_tests], double mean);
void print_version_results(const char* version_name, int* distances, int numVertices, int source, int has_negative_cycle, double execution_time);
void print_time_matrix(const char* versions[], double time_matrix[], int number_of_version, int number_of_test);
void print_statistics(const char* versions[], double time_matrix[], int number_of_version, int number_of_test);


#endif

