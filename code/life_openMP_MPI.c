#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define ROWS 2000
#define COLS 2000
#define GENERATIONS 500

unsigned char **allocate_grid(int rows, int cols)
// function to allocate a grid with halo (rows+2 x cols) and initialize it to zero
{
    unsigned char **grid = (unsigned char **)malloc((rows + 2) * sizeof(unsigned char *));
    for (int i = 0; i < rows + 2; i++)
    {
        grid[i] = (unsigned char *)calloc(cols, sizeof(unsigned char));
    }
    return grid;
}

void free_grid(unsigned char **grid, int rows)
// classic function to free allocated grid memory
{
    for (int i = 0; i < rows + 2; i++)
    {
        free(grid[i]);
    }
    free(grid);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    omp_set_num_threads(3);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = ROWS;
    int cols = COLS;

    if (rows % size != 0) // protection against incompatible grid size + number of process
    {
        if (rank == 0)
            fprintf(stderr, "Error: rows (%d) must be divisible by the number of processes (%d).\n", rows, size);
        MPI_Finalize();
        return 1;
    }

    int local_rows = rows / size;

    unsigned char **current_grid = allocate_grid(local_rows, cols);
    unsigned char **next_grid = allocate_grid(local_rows, cols);

    if (rank == 0)
    {
        unsigned char *global_grid = (unsigned char *)malloc(rows * cols * sizeof(unsigned char));
        srand(1);
        for (int i = 0; i < rows * cols; i++)
        {
            global_grid[i] = (rand() % 100 < 30) ? 1 : 0;
        }

        for (int r = 0; r < size; r++)
        {
            if (r == 0)
            {
                for (int i = 1; i <= local_rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        current_grid[i][j] = global_grid[(i - 1) * cols + j];
                    }
                }
            }
            else
            {
                unsigned char *start_ptr = &global_grid[r * local_rows * cols];
                MPI_Send(start_ptr, local_rows * cols, MPI_UNSIGNED_CHAR, r, 0, MPI_COMM_WORLD);
            }
        }
        free(global_grid);
    }
    else
    {
        unsigned char *local_buf = (unsigned char *)malloc(local_rows * cols * sizeof(unsigned char));
        MPI_Recv(local_buf, local_rows * cols, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 1; i <= local_rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                current_grid[i][j] = local_buf[(i - 1) * cols + j];
            }
        }
        free(local_buf);
    }

    int local_alive = 0;
    for (int i = 1; i <= local_rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (current_grid[i][j] == 1)
                local_alive++;
        }
    }
    int global_init_alive = 0;
    MPI_Reduce(&local_alive, &global_init_alive, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Starting MPI/OpenMP simulation (%dx%d, %d iterations)...\n", rows, cols, GENERATIONS);
        printf("Initial state: %d alive cells.\n\n", global_init_alive);
    }

    // configure neighbors for halo exchange
    int up_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    double start_time = MPI_Wtime();

    // Main simulation loop
    for (int g = 0; g < GENERATIONS; g++)
    {
        // exchange halo rows with neighbors
        MPI_Sendrecv(current_grid[1], cols, MPI_UNSIGNED_CHAR, up_neighbor, 0,
                     current_grid[local_rows + 1], cols, MPI_UNSIGNED_CHAR, down_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(current_grid[local_rows], cols, MPI_UNSIGNED_CHAR, down_neighbor, 1,
                     current_grid[0], cols, MPI_UNSIGNED_CHAR, up_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

// calculate next generation in parallel using OpenMP
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= local_rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int alive_neighbors = 0;
                for (int di = -1; di <= 1; di++)
                {
                    for (int dj = -1; dj <= 1; dj++)
                    {
                        if (di == 0 && dj == 0)
                            continue;
                        int nj = j + dj;
                        if (nj >= 0 && nj < cols)
                        {
                            if (current_grid[i + di][nj] == 1)
                                alive_neighbors++;
                        }
                    }
                }

                // rules of the GOL
                if (current_grid[i][j] == 1)
                {
                    next_grid[i][j] = (alive_neighbors == 2 || alive_neighbors == 3) ? 1 : 0;
                }
                else
                {
                    next_grid[i][j] = (alive_neighbors == 3) ? 1 : 0;
                }
            }
        }

        unsigned char **temp = current_grid;
        current_grid = next_grid;
        next_grid = temp;

        if ((g + 1) % 50 == 0)
        {
            int iter_alive = 0;
#pragma omp parallel for collapse(2) reduction(+ : iter_alive)
            for (int i = 1; i <= local_rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (current_grid[i][j] == 1)
                        iter_alive++;
                }
            }
            int global_iter_alive = 0;
            MPI_Reduce(&iter_alive, &global_iter_alive, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0)
            {
                printf("Progress: iteration %3d/%d -> %d alive cells\n", g + 1, GENERATIONS, global_iter_alive);
            }
        }
    }

    double end_time = MPI_Wtime();

    int final_alive = 0;
#pragma omp parallel for collapse(2) reduction(+ : final_alive)
    for (int i = 1; i <= local_rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (current_grid[i][j] == 1)
                final_alive++;
        }
    }
    int global_final_alive = 0;
    MPI_Reduce(&final_alive, &global_final_alive, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("\n=========================================\n");
        printf("SIMULATION MPI/OPENMP DONE\n");
        printf("=========================================\n");
        printf("Grid size : %d x %d\n", rows, cols);
        printf("Generations calculated : %d\n", GENERATIONS);
        printf("Execution time    : %f seconds\n", end_time - start_time);
        printf("Live cells at simulation end : %d\n", global_final_alive);
        printf("=========================================\n");
    }

    free_grid(current_grid, local_rows);
    free_grid(next_grid, local_rows);

    MPI_Finalize();
    return 0;
}

// gcc -fopenmp life_openMP_MPI.c -o mixed.exe -I. -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86" -lmsmpi
// mpiexec -n 2 .\mixed.exe