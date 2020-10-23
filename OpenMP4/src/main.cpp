#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <assert.h>
#include <algorithm>
#include <vector>

using namespace std;

template <typename T>
T div_up(T &&a, T &&b) {
  return (a+b-1)/b;
}

template <typename T>
T min(T &&a, T &&b)
{
    return (b < a) ? b : a;
}


struct Values {
  double sum;
  double term;
};

Values calculate(int beg, int end) {
  double term = 1, sum = 0;
  for (int i = beg; i < end; i++) {
    term /= i;
    sum += term;
  }
  struct Values s;
  s.sum = sum;
  s.term = term;
  return s;
}

double calc(uint32_t x_last, uint32_t num_threads)
{
  int n_task = div_up((int) x_last-1, (int) num_threads);
  vector<Values> res(num_threads);
  #pragma omp parallel num_threads(num_threads) firstprivate(n_task)
  {
    int id = omp_get_thread_num();
    int beg = id * n_task + 1;
    int end = min(beg+n_task, (int) x_last);
    res[id] = calculate(beg, end);
  }
  double mp = 1;
  double sum = 1;
  for (int i = 0; i < (int) num_threads; i++) {
    sum += mp * res[i].sum;
    mp *= res[i].term;
  }
  return sum;
}

int main(int argc, char** argv)
{
  // Check arguments
  if (argc != 3)
  {
    std::cout << "[Error] Usage <inputfile> <output file>\n";
    return 1;
  }

  // Prepare input file
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    std::cout << "[Error] Can't open " << argv[1] << " for write\n";
    return 1;
  }

  // Prepare output file
  std::ofstream output(argv[2]);
  if (!output.is_open())
  {
    std::cout << "[Error] Can't open " << argv[2] << " for read\n";
    input.close();
    return 1;
  }

// Read arguments from input
  uint32_t x_last = 0, num_threads = 0;
  input >> x_last >> num_threads;

  // Calculation
  double res = calc(x_last, num_threads);

  // Write result
  output << std::setprecision(16) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
