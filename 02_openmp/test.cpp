#include <cstdio>
#include <omp.h>

int main() {
#pragma omp parallel
  for(int i=0; i<4; i++) {
#pragma omp for
    for (int j=0; j<4; j++) {
      printf("%d: %d %d\n",omp_get_thread_num(),i,j);
    }
  }
}
