#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>

#define I(y, x) ((y) * inx + (x))

using namespace std;

using mat2d = vector<float>;

const int nit = 50;
const float nx = 41.0f;
const float ny = 41.0f;
const size_t inx = static_cast<size_t>(nx);
const size_t iny = static_cast<size_t>(ny);
const size_t sizeofmat = inx * iny * sizeof(float);

const dim3 numBlocks(2, 2);
const dim3 threads(inx / 2, iny / 2);

void linspace(vector<float>& v, float b, float e, int s) {
  v.reserve(s);
  for (int i = 0; i < s; i++) {
    v.push_back(b + e / (s - 1) * i);
  }
}

void zeros(vector<float>& v, int row, int col) {
  v.reserve(row * col);
  for (int i = 0; i < row * col; i++) {
    v.push_back(0);
  }
}

template<class T>
string to_json_1d(const vector<T> &v) {
  stringstream ss;
  ss << "[";
  bool fst = true;
  for (auto e : v) {
    if (!fst) ss << ",";
    fst = false;
    ss << e;
  }
  ss << "]";
  return ss.str();
}

template<class T>
string to_json_2d(const vector<T> &v) {
  stringstream ss;
  ss << "[";
  for (int y = 0; y < iny; y++) {
    if (y > 0) ss << ",";
    ss << "[";
    for (int x = 0; x < inx; x++) {
      if (x > 0) ss << ",";
      ss << v[I(y, x)];
    }
    ss << "]";
  }
  ss << "]";
  return ss.str();
}

float* cudaMalloc_with(mat2d& mat) {
  float *mat_dev;
  cudaMalloc(&mat_dev, sizeofmat);
  cudaMemcpy(mat_dev, mat.data(), sizeofmat, cudaMemcpyHostToDevice);
  return mat_dev;
}

void cudaFree_with(mat2d& mat, float* mat_dev) {
  cudaMemcpy(mat.data(), mat_dev, sizeofmat, cudaMemcpyDeviceToHost);
  cudaFree(mat_dev);
}

__global__ void build_up_b_device(int inx, int iny, float* b, float rho, float dt, float* const u, float* const v, float dx, float dy) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || inx - 1 <= x || y < 1 || iny - 1 <= y) return;

  b[I(y, x)] = (rho * (1.0f / dt *
                ((u[I(y, x + 1)] - u[I(y, x - 1)]) / (2 * dx) +
                 (v[I(y + 1, x)] - v[I(y - 1, x)]) / (2 * dy)) -
                pow((u[I(y, x + 1)] - u[I(y, x - 1)]) / (2 * dx), 2) -
                2 * ((u[I(y + 1, x)] - u[I(y - 1, x)]) / (2 * dy) *
                     (v[I(y, x + 1)] - v[I(y, x - 1)]) / (2 * dx)) -
                pow((v[I(y + 1, x)] - v[I(y - 1, x)]) / (2 * dy), 2)));
}

__global__ void pressure_poisson_device(float* p, float *pn, float dx, float dy, float* b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || inx - 1 <= x || y < 1 || iny - 1 <= y) return;

  p[I(y, x)] = (((pn[I(y, x + 1)] + pn[I(y, x - 1)]) * dy * dy +
        (pn[I(y + 1, x)] + pn[I(y - 1, x)]) * dx * dx) /
      (2 * (dx * dx + dy * dy)) -
      dx * dx * dy * dy / (2 * (dx * dx + dy * dy)) *
      b[I(y, x)]);
}

__global__ void pressure_poisson_finish_device(float *p) {
  const int width = inx;
  const int height = iny;

  for (int y = 0; y < height; y++)
    p[I(y, width - 1)] = p[I(y, width - 2)];
  for (int x = 0; x < width; x++)
    p[I(0, x)] = p[I(1, x)];
  for (int y = 0; y < height; y++)
    p[I(0, y)] = p[I(y, 1)];
  for (int x = 0; x < width; x++)
    p[I(height - 1, x)] = 0.0f;
}

void pressure_poisson(float* p_dev, float dx, float dy, float* b_dev) {
  float *pn_dev;
  cudaMalloc(&pn_dev, sizeofmat);

  for (int i = 0; i < nit; i++) {
    cudaMemcpy(pn_dev, p_dev, sizeofmat, cudaMemcpyDeviceToDevice);

    pressure_poisson_device<<<numBlocks, threads>>>(p_dev, pn_dev, dx, dy, b_dev);
    cudaDeviceSynchronize();
    pressure_poisson_finish_device<<<1, 1>>>(p_dev);
    cudaDeviceSynchronize();
  }

  cudaFree(pn_dev);
}

__global__ void cavity_flow_device(float* u, float* v, float* un, float* vn,
    float dt, float dx, float dy, float* p, float rho, float nu) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || inx - 1 <= x || y < 1 || iny - 1 <= y) return;

  u[I(y, x)] = (un[I(y, x)] -
      un[I(y, x)] * dt / dx *
      (un[I(y, x)] - un[I(y, x - 1)]) -
      vn[I(y, x)] * dt / dy *
      (un[I(y, x)] - un[I(y - 1, x)]) -
      dt / (2 * rho * dx) * (p[I(y, x + 1)] - p[I(y, x - 1)]) +
      nu * (dt / (dx * dx) *
        (un[I(y, x + 1)] - 2 * un[I(y, x)] + un[I(y, x - 1)]) +
        dt / (dy * dy) *
        (un[I(y + 1, x)] - 2 * un[I(y, x)] + un[I(y - 1, x)])));

  v[I(y, x)] = (vn[I(y, x)] -
      un[I(y, x)] * dt / dx *
      (vn[I(y, x)] - vn[I(y, x - 1)]) -
      vn[I(y, x)] * dt / dy *
      (vn[I(y, x)] - vn[I(y - 1, x)]) -
      dt / (2 * rho * dy) * (p[I(y + 1, x)] - p[I(y - 1, x)]) +
      nu * (dt / (dx * dx) *
        (vn[I(y, x + 1)] - 2 * vn[I(y, x)] + vn[I(y, x - 1)]) +
        dt / (dy * dy) *
        (vn[I(y + 1, x)] - 2 * vn[I(y, x)] + vn[I(y - 1, x)])));
}

__global__ void cavity_flow_finish_device(float* u, float* v) {
  const int width = inx;
  const int height = iny;

  for (int x = 0; x < width; x++) {
    u[I(0, x)] = 0.0f;
  }
  for (int y = 0; y < height; y++) {
    u[I(y, 0)] = 0.0f;
    u[I(y, width - 1)] = 0.0f;
  }
  for (int x = 0; x < width; x++) {
    u[I(height - 1, x)] = 1.0f;
  }

  for (int x = 0; x < width; x++) {
    v[I(0, x)] = 0.0f;
    v[I(width - 1, x)] = 0.0f;
  }
  for (int y = 0; y < height; y++) {
    v[I(y, 0)] = 0.0f;
    v[I(y, width - 1)] = 0.0f;
  }
}

void cavity_flow(int nt, mat2d& u, mat2d& v, float dt, float dx, float dy, mat2d& p, float rho, float nu) {
  mat2d b;
  zeros(b, iny, inx);

  float *b_dev = cudaMalloc_with(b);
  float *u_dev = cudaMalloc_with(u);
  float *v_dev = cudaMalloc_with(v);
  float *un_dev; cudaMalloc(&un_dev, sizeofmat);
  float *vn_dev; cudaMalloc(&vn_dev, sizeofmat);
  float *p_dev = cudaMalloc_with(p);

  for (int i = 0; i < nt; i++) {
    cudaMemcpy(un_dev, u_dev, sizeofmat, cudaMemcpyDeviceToDevice);
    cudaMemcpy(vn_dev, v_dev, sizeofmat, cudaMemcpyDeviceToDevice);

    build_up_b_device<<<numBlocks, threads>>>(inx, iny, b_dev, rho, dt, u_dev, v_dev, dx, dy);
    cudaDeviceSynchronize();

    pressure_poisson(p_dev, dx, dy, b_dev);

    cavity_flow_device<<<numBlocks, threads>>>(u_dev, v_dev, un_dev, vn_dev, dt, dx, dy, p_dev, rho, nu);
    cudaDeviceSynchronize();

    cavity_flow_finish_device<<<1, 1>>>(u_dev, v_dev);
    cudaDeviceSynchronize();
  }

  cudaFree(b_dev);
  cudaFree_with(u, u_dev);
  cudaFree_with(v, v_dev);
  cudaFree(un_dev);
  cudaFree(vn_dev);
  cudaFree_with(p, p_dev);
}

int main() {
  int nt = 2;
  float dx = 2.0f / (nx - 1.0f);
  float dy = 2.0f / (ny - 1.0f);

  vector<float> x;
  vector<float> y;
  linspace(x, 0.0f, 2.0f, inx);
  linspace(y, 0.0f, 2.0f, iny);

  float rho = 1.0f;
  float nu = 0.1f;
  float dt = 0.001f;

  mat2d u, v, p, b;
  zeros(u, iny, inx);
  zeros(v, iny, inx);
  zeros(p, iny, inx);
  zeros(b, iny, inx);

  cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu);

  ofstream output;
  output.open("cavity-cuda.json");
  output << "{\n" <<
    "\"x\": " << to_json_1d(x) << ", \"y\": " << to_json_1d(y) << ",\n" <<
    "\"p\": " << to_json_2d(p) << ",\n" <<
    "\"u\": " << to_json_2d(u) << ",\n" <<
    "\"v\": " << to_json_2d(v) << "\n" << "}\n";
  output.close();

  return 0;
}
