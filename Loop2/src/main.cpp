#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

#define ROOT 0

void calc_sin(double* arr, const uint32_t &xSize,
              const uint32_t &beg, const uint32_t &end) {
    for (uint32_t i = 0; i < (end - beg - 1); i++) {
        for (uint32_t j = 3; j < xSize; j++) {
            arr[i * xSize + j] = sin(0.00001 * arr[(i + 1) * xSize + j - 3]);
        }
    }
}

void slave_calc(const uint32_t &xSize, const uint32_t &ySize,
                 const uint32_t &rank, const uint32_t &task_sz) {

    MPI_Status status;
    uint32_t beg, end;
    double *task_arr;
    
    beg = rank * task_sz;
    end = (rank + 1) * task_sz + 1;
    if (end > ySize)
        end = ySize;
    if (beg > ySize)
        beg = ySize;
    
    task_arr = (double*) malloc (sizeof(double) * xSize * (end - beg));
    
    if (end == beg) {
        MPI_Send(task_arr, 0, MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(task_arr, xSize * (end - beg), MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD, &status);
        calc_sin(task_arr, xSize, beg, end);
        MPI_Send(task_arr, xSize * (end - beg - 1), MPI_DOUBLE, ROOT, 0, MPI_COMM_WORLD);
    }
    free(task_arr);
}

void master_calc(double* arr, const uint32_t &xSize, const uint32_t &ySize,
                 const uint32_t &task_sz, const int &size) {

    MPI_Status status;
    uint32_t beg, end;
    
    for (int i = 1; i < size; i++) {
        beg = i * task_sz;
        end = (i + 1) * task_sz + 1;
        if (end > ySize) {
            end = ySize;
        }
        if (beg > ySize) {
            beg = ySize;
        }
        MPI_Send(&(arr[beg * xSize]), xSize * (end - beg), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    
    beg = 0;
    end = ((task_sz + 1) < ySize) ? task_sz + 1 : ySize;

    calc_sin(arr, xSize, beg, end);
    
    for (int i = 1; i < size; i++) {
        beg = i * task_sz;
        end = (i + 1) * task_sz;
        if (end > ySize)
            end = ySize;
        if (beg > ySize)
            beg = ySize;
        MPI_Recv (&(arr[beg * xSize]), xSize * (end - beg), MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
    }
}

void calc(double* arr, uint32_t ySize, uint32_t xSize, int rank, int size)
{

    MPI_Bcast(&ySize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    uint32_t task_sz = (ySize + size) / size;
    
    if (rank == 0 && size > 0) {
        master_calc(arr, xSize, ySize, task_sz, size);
    } else {
        slave_calc(xSize, ySize, rank, task_sz);
    }
}



int main(int argc, char** argv)
{
  int rank = 0, size = 0, buf = 0;
  uint32_t ySize = 0, xSize = 0;
  double* arr = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      buf = 1;
      MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize;
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    arr = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> arr[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (buf != 0)
    {
      return 1;
    }
  }

  calc(arr, ySize, xSize, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete arr;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << arr[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete arr;
  }

  MPI_Finalize();
  return 0;
}
