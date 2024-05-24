//
// Created by rick on 28/11/23.
//


#include "general_utilities.h"
#include <string.h>



double calculate_mean(int num_tests, const double times[num_tests]){
    double sum = 0.0;
    for (int i = 0; i < num_tests; i++){
        sum += times[i];
    }
    return sum / num_tests;
}


double calculate_std_dev(int num_tests, const double times[num_tests], double mean){
    double sum_squared_diff = 0.0;
    for (int i = 0; i < num_tests; i++) {
        double diff = times[i] - mean;
        sum_squared_diff += diff * diff;
    }
    return sqrt(sum_squared_diff / num_tests);
}


void print_version_results(const char *version_name, int *distances, int num_vertices, int source, int has_negative_cycle, double execution_time){
    printf("======================================\n");
    printf("%s solution:\n", version_name);
    printf("======================================\n");

    if (has_negative_cycle){
        printf("Negative cycle detected.\n");
    }
    else{
        printf("Shortest distances from node %d:\n", source);
        for (int i = 0; i < num_vertices; i++)
            printf("To node %d: %d\n", i, distances[i]);
    }
    printf("Bellman-Ford %s require: %f seconds\n\n", version_name, execution_time);
}


void print_time_matrix(const char *versions[], double *time_matrix, int number_of_version, int number_of_test){
    printf("%-15s ", "Version");
    for (int j = 1; j <= number_of_test; j++){
        char test_label[10];
        sprintf(test_label, "test%d", j);
        printf("%-15s ", test_label);
    }
    printf("\n");

    for (int i = 0; i < number_of_version; i++){
        printf("%-15s ", versions[i]);
        for (int j = 0; j < number_of_test; j++)
            printf("%-15f ", time_matrix[i * number_of_test + j]);
        printf("\n");
    }
    printf("\n");
}


void print_statistics(const char *versions[], double time_matrix[], int number_of_version, int number_of_test){
    printf("%-15s %-15s %-15s\n", "Version", "Mean", "Standard Deviation");

    for (int i = 0; i < number_of_version; i++){
        double mean = calculate_mean(number_of_test, &time_matrix[i * number_of_test]);
        double std_dev = calculate_std_dev(number_of_test, &time_matrix[i * number_of_test], mean);

        printf("%-15s %-15f %-15f\n", versions[i], mean, std_dev);

    }
}


void check_differences(const int *serial, int **distances, const char **versions, int numArrays, int length){
    bool *are_equal_flags = (bool*)malloc(numArrays * sizeof(bool));
    char error_message[100];

    for (int i = 0; i < numArrays; i++)
        are_equal_flags[i] = true;

    for (int i = 0; i < numArrays; i++){
        for (int j = 0; j < length; j++){
            if (serial[j] != distances[i][j]){
                are_equal_flags[i] = false;
                break;
            }
        }
    }

    for (int i = 0; i < numArrays; i++){
        if (!are_equal_flags[i]) {
            snprintf(error_message, sizeof(error_message),
                     "ERROR: The Version %s is different from Sequential Version.\n\n", versions[i + 1]);
            printf("\033[1;31m%s\033[0m\n", error_message);

        }
    }

    free(are_equal_flags);

}


void write_results_in_txt(const char *file_name, const char **versions, const double *time_statistics, const int num_versions,
                          const int n_cols, const  char *header_string, const int *additional_strings_components) {
    FILE *file = fopen(file_name, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening the file %s\n", file_name);
        return;
    }

    int tmp ;
    fprintf(file, "Algorithm ");
    for (int i=0; i<(n_cols); i++) {
      if (additional_strings_components == NULL)
          tmp = i+1;
      else
          tmp = additional_strings_components[i];

      fprintf(file, "mean_%s%d ", header_string, tmp);
      fprintf(file, "std_dev_%s%d ", header_string, tmp);
    }

    for (int i=0; i<num_versions; i++) {
        fprintf(file, "\n%s ", versions[i]);
        for (int j = 0; j<(n_cols * 2); j++) {
            fprintf(file, "%f " , time_statistics[i * 2*(n_cols) + j]);
        }
    }

    if (fclose(file) != 0) {
        fprintf(stderr, "Error closing the file %s\n", file_name);
        return;
    }
}
