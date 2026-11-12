# Same optimization level for both programs so the comparison is fair.
CC     = gcc
MPICC  = mpicc
CFLAGS = -O2 -Wall

all: mpi_matmul omp_matmul

mpi_matmul: mpi_matmul.c common.h
	$(MPICC) $(CFLAGS) -o mpi_matmul mpi_matmul.c -lm

omp_matmul: omp_matmul.c common.h
	$(CC) $(CFLAGS) -fopenmp -o omp_matmul omp_matmul.c -lm

clean:
	rm -f mpi_matmul omp_matmul

.PHONY: all clean
