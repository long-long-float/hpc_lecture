#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void put_bucket(int *bucket, int *key, int N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  bucket[id] = 0;
  __syncthreads();

  extern __shared__ int sharedKey[];
  if (id == 0) {
    for (int i = 0; i < N; i++) sharedKey[i] = key[i];
  }
  __syncthreads();

  for (int i = 0; i < N; i++) {
    if (id == sharedKey[i]) bucket[id]++;
  }
}

// Slide Lecture 5 P19
__global__ void scan(int *a, int *b, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=1; j<N; j<<=1) {
    b[i] = a[i];
    __syncthreads();
    a[i] += b[i-j];
    __syncthreads();
  }
}

__global__ void bucket_sort(int *bucket, int *bucket_sum, int *key) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < bucket[id]; i++) {
    key[i + bucket_sum[id] - bucket[id]] = id;
  }
}

int main() {
  int n = 50;
  int range = 5;

  int *bucket, *bucket_sum, *bucket_temp, *key;
  cudaMallocManaged(&bucket, range * sizeof(int));
  cudaMallocManaged(&bucket_sum, range * sizeof(int));
  cudaMallocManaged(&bucket_temp, range * sizeof(int));
  cudaMallocManaged(&key, n * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  put_bucket<<<1, range, n>>>(bucket, key, n);
  cudaDeviceSynchronize();

  cudaMemcpy(bucket_sum, bucket, range * sizeof(int), cudaMemcpyDefault);
  scan<<<1, range>>>(bucket_sum, bucket_temp, range);
  cudaDeviceSynchronize();

  bucket_sort<<<1, range>>>(bucket, bucket_sum, key);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(bucket);
  cudaFree(bucket_sum);
  cudaFree(bucket_temp);
  cudaFree(key);
}
