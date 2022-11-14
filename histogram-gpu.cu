#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define MAX_LENGTH 614400

// nsys profile -t nvtx,cuda --stats=true --force-overwrite true --wait=all -o my_report ./my_app

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void sequential_histogram(char *data, unsigned int *histogram, int length)
{
    for (int i = 0; i < length; i++)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) // check if we have an alphabet char
            histogram[alphabet_position / 6]++;               // we group the letters into blocks of 6
    }
}

__global__ void histogram_kernel(char *data, unsigned int *histogram, int length)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int section_size = (length - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i * section_size;
    // All threads handle blockDim.x * gridDim.x
    // consecutive elements
    for (size_t k = 0; k < section_size; k++)
    {
        if (start + k < length)
        {
            int alphabet_position = data[start + k] - 'a';
            if (alphabet_position >= 0 && alphabet_position < 26)
                atomicAdd(&(histogram[alphabet_position / 6]), 1);
        }
    }
}

int main(int argc, char *argv[])
{
    FILE *fp = fopen("test.txt", "read");

    if (argc != 2)
    {
        printf("Usage: ./exec BLOCKDIM\n");
        return 0;
    }

    int BLOCKDIM = atoi(argv[1]);

    // unsigned char text[MAX_LENGTH];
    char *text = (char *)malloc(sizeof(char) * MAX_LENGTH);
    char *text_d;
    size_t len = 0;
    size_t read;
    unsigned int histogram[5] = {0};
    unsigned int histogram_hw[5] = {0};
    unsigned int *histogram_d;
    double start_cpu, end_cpu, start_gpu, end_gpu;

    if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&text, &len, fp)) != -1)
    {
        printf("Retrieved line of length %ld:\n", read);
    }
    fclose(fp);

    start_cpu = get_time();
    sequential_histogram(text, histogram, len);
    end_cpu = get_time();

    CHECK(cudaMalloc(&text_d, len * sizeof(char)));                              // allocate space for the input array on the GPU
    CHECK(cudaMalloc(&histogram_d, 5 * sizeof(unsigned int)));                   // and for the histogram
    CHECK(cudaMemcpy(text_d, text, len * sizeof(char), cudaMemcpyHostToDevice)); // copy input data on the gpu

    dim3 blocksPerGrid((len + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
    dim3 threadsPerBlock(BLOCKDIM, 1, 1);
    start_gpu = get_time();
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(text_d, histogram_d, len);
    CHECK_KERNELCALL();

    cudaDeviceSynchronize();
    end_gpu = get_time();
    CHECK(cudaMemcpy(histogram_hw, histogram_d, 5 * sizeof(unsigned int), cudaMemcpyDeviceToHost)); // copy data back from the gpu
    for (size_t i = 0; i < 5; i++)
    {
        if (histogram[i] != histogram_hw[i])
        {
            printf("Error on GPU at index: %ld\n", i);
            return 0;
        }
    }
    printf("ALL GPU OK\n");

    printf("CPU Sort Time: %.5lf\n", end_cpu - start_cpu);
    printf("GPU Sort Time: %.5lf\n", end_gpu - start_gpu);

    CHECK(cudaFree(text_d));
    CHECK(cudaFree(histogram_d));

    return 1;
}
