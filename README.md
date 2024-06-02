# Bellman-Ford-Parallel-Implementation

# OpenMP and CUDA implementations

## Overview

This project implements parallel versions of graph algorithms using CUDA and OpenMP. The goal is to evaluate the performance and efficiency of these implementations across different test scenarios.

## Implementations

The project includes several versions, each implemented with different parallelization strategies:


### OpenMP Versions

- **V0**: This version operates on the edge list similarly to the sequential one but introduces parallelization for distance initialization, the internal computation within each relaxation, and the operation for detecting negative cycles. Specifically, the distance initialization is performed by creating \( |V| \) tasks, each handling a job to assign the value 0 to the source node and the largest representable value to the other nodes. Each relaxation creates \( |E| \) tasks, each handling an edge, with updates performed within a critical section. The detection of negative cycles is also parallelized using \( |E| \) tasks and a reduction mechanism. Each task creates \( T \) threads based on user preferences.

- **V0_1**: This version operates in the same way as V0, but the parallelization parts are implemented by creating the threads only once and reusing them for all operations.

- **V1**: This version parallelizes the distance initialization similarly to V0 and employs a single level of parallelization to perform the individual relaxations. It divides the iteration over all edges into \( |V| \) chunks, each containing edges targeting the same destination vertex. Using the adjacency matrix, it creates \( |V| \) tasks to process edges sharing the same destination, finding the shortest path by computing distances and selecting the minimum. Negative cycle detection is parallelized with \( |V| \) tasks. 

- **V1_1**: This version operates in the same way as V1, but the parallelization parts are implemented by creating the threads only once and reusing them for all operations.

- **V2**: This version implements the same logic as V1 but uses two levels of parallelization. The first-level parallelization is the same as in V1, while the second level parallelizes the minimum-finding step within each task using a nested approach. Each task creates additional \( V \) tasks and \( T \) threads, implementing a custom reduction to find the minimum.

- **V2_1**: This version operates in the same way as V2, but the parallelization parts are implemented by creating the first-level threads only once and reusing them for all operations. The second-level threads are created and destroyed for every relaxation.

### CUDA Versions

- **cuda_V0**: This version is the CUDA implementation of the V0 algorithm. It operates on the list of edges and utilizes three CUDA kernel functions: one for initializing distances, another for performing individual relaxations, and a final one for checking the presence of negative cycles. The first kernel function uses \( |V| \) threads to initialize distances. The second kernel performs the relaxations with \( |E| \) threads, using atomic operations to ensure correct updates. The final kernel checks for negative cycles with \( |E| \) threads using shared memory and atomic operations.

- **cuda_V0_1**: Similar to cuda_V0, but introduces a different kernel for processing relaxations and implements custom synchronization between blocks to avoid deadlock situations. It calculates the number of blocks \( B \) and the group size \( G \) to ensure optimal GPU utilization and proper synchronization.

- **cuda_V1**: This version uses the adjacency matrix for graph representation and follows the same logic as cuda_V0 for initialization and negative cycle checking. It optimizes the relaxation step by operating on the adjacency matrix with three CUDA kernel functions.

- **cuda_V1_1**: Builds on cuda_V1 by incorporating the block synchronization mechanism from cuda_V0_1, ensuring efficient parallel execution and synchronization.

For more detailed descriptions of each version, please refer to the project report.

## Experimental Setup

To evaluate the characteristics of each implementation, four tests were conducted, each selectable by activating a specific flag when launching the executable:

1. **Development Phase Test**:
    - **Description**: This test was used exclusively during the development phase and is currently disabled in the project's final presentation. It allowed for verifying the correctness of the various implementations.
    - **Usage**: Set the first flag to 1.
    - **Details**: Enables manual insertion of parameters to generate graphs of arbitrary size with edges defined by an arbitrary interval \[ lower bound, upper bound]\. Allows specifying the number of tests, displaying the list of edges, adjacency matrices, and solutions produced by each algorithm with corresponding execution times. Calculates average and standard deviation of execution times for multiple tests.
    - **Note**: Graphs are generated randomly based on incremental seeds to ensure reproducibility.

2. **Strong Efficiency Test**:
    - **Description**: Assesses the strong efficiency of the OpenMP versions.
    - **Usage**: Set the second flag to 1.
    - **Details**: Runs the OpenMP versions on graphs with 1000 vertices, varying the number of threads from 1 to 8. For each configuration, three tests are conducted on three different random graphs to ensure reliability.

3. **Weak Efficiency Test**:
    - **Description**: Evaluates the weak efficiency of the OpenMP versions by varying workloads.
    - **Usage**: Set the third flag to 1.
    - **Details**: Starts with graphs of 1000 nodes, incrementing their size by a factor \( k \) from 1 to 6. The same factor is used to increase the number of threads.

4. **Execution Time Collection Test**:
    - **Description**: Collects execution times for each version on random graphs of various sizes.
    - **Usage**: Set the fourth flag to 1.
    - **Details**: Processes random graphs of sizes 100, 500, 1000, 2000, 5000, and 10000. For this test, the number of threads for the OpenMP versions was fixed at 4, while for the CUDA versions, the number of threads per block was set to 1024. For each size, three tests were conducted to calculate averages and standard deviations of execution times.

All the tests described above are aimed at collecting execution times, which are exported to specific text files. These files are utilized by a Python script to generate graphs, making it easier to visualize and analyze the performance metrics.

## How to Run the Tests

1. **Compile the Code**: Ensure you have the necessary compilers and libraries installed for CUDA and OpenMP.
2. **Set the Flags**: Modify the flag settings in  my_bash.sbash to set the desired test flag.
3. **Run the Project**: Execute the `my_bash.sh` script (or `sbatch sulurm.sbatch` if using SLURM), which handles compiling the project and launching the executable.
   - **Note**: Depending on the GPU architecture used, it might be necessary to specify the architecture and correctly set the path to the CUDA libraries.
4. **View Results**: Execution times will be exported to text files, which can be processed by the provided Python script to generate performance graphs.

## Requirements

- **CUDA**: Ensure you have a compatible NVIDIA GPU and CUDA toolkit installed.
- **OpenMP**: Ensure your compiler supports OpenMP (e.g., GCC with OpenMP support).
- **Python**: Required for running the analysis script (with libraries such as matplotlib for plotting).

## Authors

- Riccardo Murgia


## License

This project is licensed under the MIT License - see the LICENSE file for details.

