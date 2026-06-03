#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define ROWS 8000
#define COLS 8000
#define ITERATIONS 500

// macro to check CUDA calls for errors
#define CUDA_CHECK(call)                                                                                \
    do                                                                                                  \
    {                                                                                                   \
        cudaError_t err = call;                                                                         \
        if (err != cudaSuccess)                                                                         \
        {                                                                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                         \
        }                                                                                               \
    } while (0)

__global__ void compute_next_generation_kernel(const unsigned char *grid_in, unsigned char *grid_out)
{
    // calculate the row and column index
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= 0 && r < ROWS && c >= 0 && c < COLS)
    {
        int count = 0;
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                if (i == 0 && j == 0)
                    continue;

                int ni = r + i;
                int nj = c + j;

                if (ni >= 0 && ni < ROWS && nj >= 0 && nj < COLS)
                {
                    count += grid_in[ni * COLS + nj];
                }
            }
        }

        int idx = r * COLS + c;
        if (grid_in[idx] == 1)
        {
            grid_out[idx] = (count == 2 || count == 3) ? 1 : 0;
        }
        else
        {
            grid_out[idx] = (count == 3) ? 1 : 0;
        }
    }
}

int main()
{
    size_t size = ROWS * COLS * sizeof(unsigned char);

    unsigned char *h_current_grid = (unsigned char *)malloc(size);
    unsigned char *h_next_grid = (unsigned char *)malloc(size);

    srand(1);
    for (int i = 0; i < ROWS * COLS; i++)
    {
        h_current_grid[i] = (rand() % 100 < 30) ? 1 : 0;
    }

    int initial_alive = 0;
    for (int i = 0; i < ROWS * COLS; i++)
    {
        if (h_current_grid[i] == 1)
            initial_alive++;
    }

    printf("Starting CUDA simulation (%dx%d, %d iterations)...\n", ROWS, COLS, ITERATIONS);
    printf("Initial state: %d alive cells.\n\n", initial_alive);

    unsigned char *d_current_grid = NULL;
    unsigned char *d_next_grid = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_current_grid, size));
    CUDA_CHECK(cudaMalloc((void **)&d_next_grid, size));

    CUDA_CHECK(cudaMemcpy(d_current_grid, h_current_grid, size, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, (ROWS + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < ITERATIONS; iter++)
    {
        compute_next_generation_kernel<<<gridSize, blockSize>>>(d_current_grid, d_next_grid);

        // verify kernel launch success
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed at iteration %d: %s\n", iter, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        unsigned char *temp = d_current_grid;
        d_current_grid = d_next_grid;
        d_next_grid = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double execution_time = milliseconds / 1000.0;

    CUDA_CHECK(cudaMemcpy(h_current_grid, d_current_grid, size, cudaMemcpyDeviceToHost));

    int final_alive = 0;
    for (int i = 0; i < ROWS * COLS; i++)
    {
        if (h_current_grid[i] == 1)
            final_alive++;
    }

    printf("CUDA simulation finished!\n");
    printf("Final state: %d alive cells\n", final_alive);
    printf("CUDA execution time: %f seconds\n", execution_time);

    cudaFree(d_current_grid);
    cudaFree(d_next_grid);
    free(h_current_grid);
    free(h_next_grid);

    return 0;
}

// in dev powershell for vs code 2022
// nvcc -allow-unsupported-compiler -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe" life_cuda.cu -o life_cuda.exe

// 215793
// 870235
// 3468803