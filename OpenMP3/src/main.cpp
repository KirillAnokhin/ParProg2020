#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <cmath>

double func(double x)
{
  return sin(x);
}

double kahan_sum(double *vals, int n) 
{
  double sum = 0.0;
  double c = 0.0;
  double t, y;

  for (int i = 0; i < n; i++) {
    y = vals[i] - c;
    t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }

  return sum;
}

double calc(double x0, double x1, double dx, uint32_t num_threads)
{
  int n = (x1 - x0) / dx;
  double *vals = (double*)calloc(n, sizeof(double));
  #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; i++) {
      vals[i] = (func(x0 + (i+1)*dx) + func(x0 + i*dx))*dx/2;
    }

  double sum = kahan_sum(vals, n);

  free(vals);
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
  double x0 = 0.0, x1 =0.0, dx = 0.0;
  uint32_t num_threads = 0;
  input >> x0 >> x1 >> dx >> num_threads;

  // Calculation
  double res = calc(x0, x1, dx, num_threads);

  // Write result
  output << std::setprecision(13) << res << std::endl;
  // Prepare to exit
  output.close();
  input.close();
  return 0;
}
