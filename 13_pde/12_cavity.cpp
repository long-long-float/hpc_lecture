#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>

using namespace std;

using mat2d = vector<vector<float>>;

const int nit = 50;
const float nx = 41.0f;
const float ny = 41.0f;
const size_t inx = static_cast<size_t>(nx);
const size_t iny = static_cast<size_t>(ny);

void linspace(vector<float>& v, float b, float e, int s) {
  v.reserve(s);
  for (int i = 0; i < s; i++) {
    v.push_back(b + e / (s - 1) * i);
  }
}

void zeros(vector<vector<float>>& v, int row, int col) {
  v.reserve(row);
  for (int i = 0; i < col; i++) {
    v.emplace_back(col, 0);
  }
}

template<class T>
string to_json(const vector<T> &v) {
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
string to_json(const vector<vector<T>> &v) {
  stringstream ss;
  ss << "[";
  bool fst = true;
  for (auto col : v) {
    if (!fst) ss << ",";
    fst = false;
    ss << to_json(col);
  }
  ss << "]";
  return ss.str();
}

void build_up_b(mat2d& b, float rho, float dt, const mat2d& u, const mat2d& v, float dx, float dy) {
  for (int y = 1; y < b.size() - 1; y++) {
    for (int x = 1; x < b[0].size() - 1; x++) {
      b[y][x] = (rho * (1.0f / dt *
                ((u[y][x + 1] - u[y][x - 1]) / (2 * dx) +
                 (v[y + 1][x] - v[y - 1][x]) / (2 * dy)) -
                pow((u[y][x + 1] - u[y][x - 1]) / (2 * dx), 2) -
                2 * ((u[y + 1][x] - u[y - 1][x]) / (2 * dy) *
                     (v[y][x + 1] - v[y][x - 1]) / (2 * dx)) -
                pow((v[y + 1][x] - v[y - 1][x]) / (2 * dy), 2)));
    }
  }
}

void pressure_poisson(mat2d& p, float dx, float dy, const mat2d& b) {
  mat2d pn = p;

  for (int i = 0; i < nit; i++) {
    pn = p;

    const int width = p[0].size();
    const int height = p.size();

    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        p[y][x] = (((pn[y][x + 1] + pn[y][x - 1]) * dy * dy +
                    (pn[y + 1][x] + pn[y - 1][x]) * dx * dx) /
                   (2 * (dx * dx + dy * dy)) -
                   dx * dx * dy * dy / (2 * (dx * dx + dy * dy)) *
                   b[y][x]);
      }
    }

    for (int y = 0; y < height; y++)
      p[y][width - 1] = p[y][width - 2];
    for (int x = 0; x < width; x++)
      p[0][x] = p[1][x];
    for (int y = 0; y < height; y++)
      p[y][0] = p[y][1];
    for (int x = 0; x < width; x++)
      p[p.size() - 1][x] = 0.0f;

  }
}

void cavity_flow(int nt, mat2d& u, mat2d& v, float dt, float dx, float dy, mat2d& p, float rho, float nu) {
  mat2d un = u;
  mat2d vn = v;
  mat2d b;
  zeros(b, iny, inx);

  for (int i = 0; i < nt; i++) {
    un = u;
    vn = v;

    build_up_b(b, rho, dt, u, v, dx, dy);
    pressure_poisson(p, dx, dy, b);

    const int width = u[0].size();
    const int height = u.size();

    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        u[y][x] = (un[y][x] -
                   un[y][x] * dt / dx *
                   (un[y][x] - un[y][x - 1]) -
                   vn[y][x] * dt / dy *
                   (un[y][x] - un[y - 1][x]) -
                   dt / (2 * rho * dx) * (p[y][x + 1] - p[y][x - 1]) +
                   nu * (dt / (dx * dx) *
                    (un[y][x + 1] - 2 * un[y][x] + un[y][x - 1]) +
                    dt / (dy * dy) *
                    (un[y + 1][x] - 2 * un[y][x] + un[y - 1][x])));

        v[y][x] = (vn[y][x] -
                   un[y][x] * dt / dx *
                   (vn[y][x] - vn[y][x - 1]) -
                   vn[y][x] * dt / dy *
                   (vn[y][x] - vn[y - 1][x]) -
                   dt / (2 * rho * dy) * (p[y + 1][x] - p[y - 1][x]) +
                   nu * (dt / (dx * dx) *
                    (vn[y][x + 1] - 2 * vn[y][x] + vn[y][x - 1]) +
                    dt / (dy * dy) *
                    (vn[y + 1][x] - 2 * vn[y][x] + vn[y - 1][x])));
      }
    }

    for (int x = 0; x < width; x++) {
      u[0][x] = 0.0f;
    }
    for (int y = 0; y < height; y++) {
      u[y][0] = 0.0f;
      u[y][width - 1] = 0.0f;
    }
    for (int x = 0; x < width; x++) {
      u[width - 1][x] = 1.0f;
    }

    for (int x = 0; x < width; x++) {
      v[0][x] = 0.0f;
      v[width - 1][x] = 0.0f;
    }
    for (int y = 0; y < height; y++) {
      v[y][0] = 0.0f;
      v[y][width - 1] = 0.0f;
    }
  }
}

int main() {
  int nt = 700;
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
  output.open("cavity-cpp.json");
  output << "{\n" <<
    "\"x\": " << to_json(x) << ", \"y\": " << to_json(y) << ",\n" <<
    "\"p\": " << to_json(p) << ",\n" <<
    "\"u\": " << to_json(u) << ",\n" <<
    "\"v\": " << to_json(v) << "\n" << "}\n";
  output.close();

  return 0;
}
