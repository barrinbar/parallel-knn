#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <mpi.h>
#include <omp.h>



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


int main(int argc,char *argv[])
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

	int  namelen, numprocs, myid;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc,&argv);

	MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);	

	MPI_Get_processor_name(processor_name,&namelen);
	
	MPI_Status status;
	int x = 7, y = 10, answer = 77777;
	if (myid == 0) {
		MPI_Send(&x, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Recv(&answer, 1, MPI_INT, 1, 0, MPI_COMM_WORLD,&status);
	}
	else {
		MPI_Recv(&y, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		y  = y*3;
		MPI_Send(&y, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);	
	}


	if (myid == 0)
		printf("answer = %d numprocs = %d  myid = %d name = %s\n", 
		answer, numprocs, myid, processor_name);
	
    
	if (myid == 1) {
#pragma omp parallel 
		{
			int tid = omp_get_thread_num();
			printf("myid = %d  tid = %d\n", myid, tid);
		}
	}


	if (myid == 0) {
		// Add vectors in parallel.
		cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		printf("myid = %d {1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
			myid, c[0], c[1], c[2], c[3], c[4]);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}
	}
	MPI_Finalize();
    return 0;
}
