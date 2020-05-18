#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

const int N = 8;

void print_vec(int lineno, __m256 vec) {
  float a[N] = {0};
  _mm256_store_ps(a, vec);
  printf("%d: ", lineno);
  for (int i = 0; i < N; i++) {
    printf("%g, ", a[i]);
  }
  printf("\n");
}

int main() {
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m256 fxvec = _mm256_setzero_ps();
  __m256 fyvec = _mm256_setzero_ps();

  for(int i=0; i<N; i++) {
    __m256 xvec = _mm256_load_ps(x);
    __m256 yvec = _mm256_load_ps(y);
    __m256 mvec = _mm256_set1_ps(m[i]);

    __m256 xvec2 = _mm256_set1_ps(x[i]);
    __m256 yvec2 = _mm256_set1_ps(y[i]);

    __m256 rxvec = _mm256_sub_ps(xvec, xvec2);
    __m256 ryvec = _mm256_sub_ps(yvec, yvec2);

    __m256 rvec  = _mm256_rsqrt_ps(
        _mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));

    float mask[N] = {0};
    mask[i] = -1.0f;
    __m256 maskvec = _mm256_load_ps(mask);
    rvec = _mm256_blendv_ps(rvec, _mm256_setzero_ps(), maskvec);

    __m256 mrrr = _mm256_mul_ps(mvec,
        _mm256_mul_ps(rvec,
        _mm256_mul_ps(rvec, rvec)));

    fxvec = _mm256_sub_ps(fxvec, _mm256_mul_ps(rxvec, mrrr));
    fyvec = _mm256_sub_ps(fyvec, _mm256_mul_ps(ryvec, mrrr));

    /*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    printf("%d %g %g\n",i,fx[i],fy[i]);
    */
  }

  _mm256_store_ps(fx, fxvec);
  _mm256_store_ps(fy, fyvec);

  for (int i = 0; i < N; i++) {
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
