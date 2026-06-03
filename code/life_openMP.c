#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ROWS 2000
#define COLS 2000
#define ITERATIONS 500

int count_neighbors(unsigned char *grid, int r, int c)
// function to count the number of alive neighbors for a cell
{
    int count = 0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue; // skip the cell itslf

            int ni = r + i;
            int nj = c + j;

            // if in the grid boundaries
            if (ni >= 0 && ni < ROWS && nj >= 0 && nj < COLS)
            {
                count += grid[ni * COLS + nj];
            }
        }
    }
    return count;
}

int main()
{
    omp_set_num_threads(6);
    // 1d array to represent the grid (1 = alive, 0 = dead)
    unsigned char *current_grid = (unsigned char *)malloc(ROWS * COLS * sizeof(unsigned char));
    unsigned char *next_grid = (unsigned char *)malloc(ROWS * COLS * sizeof(unsigned char));

    srand(1);
    for (int i = 0; i < ROWS * COLS; i++)
    {
        current_grid[i] = (rand() % 100 < 30) ? 1 : 0;
    }

    int initial_alive = 0;
    for (int i = 0; i < ROWS * COLS; i++)
    {
        initial_alive += current_grid[i];
    }

    printf("Starting OpenMP simulation (%dx%d, %d iterations)...\n", ROWS, COLS, ITERATIONS);
    printf("Initial state: %d alive cells.\n\n", initial_alive);

    double start_time = omp_get_wtime();

    for (int iter = 0; iter < ITERATIONS; iter++)
    {
#pragma omp parallel for collapse(2)
        for (int r = 0; r < ROWS; r++)
        {
            for (int c = 0; c < COLS; c++)
            {
                int neighbors = count_neighbors(current_grid, r, c);
                int idx = r * COLS + c;

                // GOL rules
                if (current_grid[idx] == 1)
                {
                    next_grid[idx] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
                }
                else
                {
                    next_grid[idx] = (neighbors == 3) ? 1 : 0;
                }
            }
        }

        // pointer swap & double buffering
        unsigned char *temp = current_grid;
        current_grid = next_grid;
        next_grid = temp;

        if ((iter + 1) % 50 == 0)
        {
            int current_alive = 0;
// Parallel reduction to accumulate alive cells in threads
#pragma omp parallel for reduction(+ : current_alive)
            for (int i = 0; i < ROWS * COLS; i++)
            {
                current_alive += current_grid[i];
            }
            printf("Progress: iteration %3d/%d -> %d alive cells\n", iter + 1, ITERATIONS, current_alive);
        }
    }

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    int final_alive = 0;
    for (int i = 0; i < ROWS * COLS; i++)
    {
        final_alive += current_grid[i];
    }

    printf("\nOpenMP simulation finished!\n");
    printf("-> Final state: %d alive cells\n", final_alive);
    printf("OpenMP execution time (Wall-clock): %f seconds\n", execution_time);

    free(current_grid);
    free(next_grid);

    return 0;
}

// gcc life_openmp.c -o openmp -fopenmp
// ./openmp

// 215793
// 870235
// 3468803

// 354 - 10000 iterations, 100x100