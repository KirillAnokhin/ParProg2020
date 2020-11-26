#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>
#include <cstring>


double acceleration(double t)
{
	return sin (t);
}

void calc (double* trace, uint32_t traceSize, double t0, double dt, double y0, double y1, int rank, int size)
{
	MPI_Status status;
	MPI_Bcast (&traceSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast (&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast (&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	uint32_t beg = traceSize * rank / size;
	uint32_t end = traceSize * (rank + 1) / size;
	uint32_t len = end - beg;
	double v0, y_beg, v_beg, y_end, v_end, y_prev, v_prev;
  v0 = y_beg = v_beg = y_end = v_end = y_prev = v_prev = 0;

	if (rank == 0)
		y_beg = y0;

	double* task_trace = (double *) calloc (len, sizeof (double));
	double time_task = dt * traceSize / size;
	t0 += time_task * rank;

	// Sighting shot
	task_trace[0] = y_beg;
	task_trace[1] = y_beg + dt * v_beg;
	for (uint32_t i = 2; i < len; i++)
	{
		task_trace[i] = dt * dt * acceleration (t0 + (i - 1) * dt)
					+ 2 * task_trace[i - 1] - task_trace[i - 2];
	}
	y_end = task_trace[len - 1];
	v_end = (task_trace[len - 1] - task_trace[len - 2]) / dt;
	
  if (size != 1)	
	{
		if (rank != 0) 
		{
			MPI_Recv (&y_prev, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
			MPI_Recv (&v_prev, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
			y_beg = y_prev;
			v_beg = v_prev;
			y_end += y_prev + v_prev * time_task;
			v_end += v_prev;
		}

		if (rank != size - 1)
		{
			MPI_Send (&y_end, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
			MPI_Send (&v_end, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
		}
    else 
		{
			MPI_Send (&y_end, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);	
		}
    if (rank == 0){
			MPI_Recv (&y_prev, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &status);
			v0 = (y1 - y_prev) / (dt * traceSize);
		} 
		MPI_Bcast (&v0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		y_beg += v0 * time_task * rank;
		v_beg += v0;
	}
  else 
	{
		v0 = (y1 - y_end) / (dt * traceSize);
		v_beg = v0;
	}

	task_trace[0] = y_beg;
	task_trace[1] = y_beg + dt * v_beg;
	for (uint32_t i = 2; i < len; i++)
	{
		task_trace[i] = dt * dt * acceleration (t0 + (i - 1) * dt)
					+ 2 * task_trace[i - 1] - task_trace[i - 2];
	}

	if (rank == 0)
	{
		memcpy (trace, task_trace, len * sizeof (double));
		for (int i = 1; i < size; i++)
		{
			uint32_t first = traceSize * i / size;
			uint32_t last = traceSize * (i + 1) / size;
			uint32_t len = last - first;
			MPI_Recv (trace + first, len, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
		}
	}
	else
		MPI_Send (task_trace, len, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);	

	free (task_trace);
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  uint32_t traceSize = 0;
  double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
  double* trace = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> t0 >> t1 >> dt >> y0 >> y1;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    traceSize = (t1 - t0)/dt;
    trace = new double[traceSize];

    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(trace, traceSize, t0, dt, y0, y1, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete trace;
      return 1;
    }

    for (uint32_t i = 0; i < traceSize; i++)
    {
      output << " " << trace[i];
    }
    output << std::endl;
    output.close();
    delete trace;
  }

  MPI_Finalize();
  return 0;
}
