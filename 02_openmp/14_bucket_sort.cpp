#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

int main() {
  int n = 50;
  int range = 20;
  int num_of_buckets = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<std::vector<int> > buckets(num_of_buckets);
  for (int i=0; i<n; i++) {
    int v = key[i];
    buckets[v / (range / num_of_buckets)].push_back(v);
  }

#pragma omp for
  for (int i = 0; i < num_of_buckets; i++) {
    std::vector<int> &bucket = buckets[i];
    std::sort(bucket.begin(), bucket.end());
  }

  std::vector<int> result;
  for (int i = 0; i < num_of_buckets; i++){
    std::vector<int> &b = buckets[i];
    result.insert(result.end(), b.begin(), b.end());
  }

  for (int i=0; i<n; i++) {
    printf("%d ",result[i]);
  }
  printf("\n");
}
