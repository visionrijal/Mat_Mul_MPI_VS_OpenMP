/*
 * mpi_matmul.c - matrix multiplication C = A * B with MPI.
 *
 * Decomposition: row-wise (1D).
 *   - Rows of A and C are split into contiguous blocks, one per rank.
 *     If N is not divisible by P, the first (N % P) ranks get one extra row.
 *   - Every rank needs all of B, so B is broadcast.
 *   - Rank 0 computes its own block like every other rank.
 *
 * Communication (all collectives):
 *   MPI_Scatterv  - distribute row blocks of A
 *   MPI_Bcast     - send all of B to every rank
 *   MPI_Gatherv   - collect row blocks of C on rank 0
 *
 * Usage: mpirun -np P ./mpi_matmul N
 */
#include <mpi.h>
#include <unistd.h>
#include "common.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 250;
    if (N <= 0) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s N\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* Process placement evidence (outside the timed region). */
    char host[256];
    gethostname(host, sizeof(host));
    printf("Rank %d of %d running on host %s\n", rank, size, host);
    fflush(stdout);

    /* Work out how many rows each rank gets (counts/offsets in elements). */
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int offset = 0;
    for (int r = 0; r < size; r++) {
        int rows = N / size + (r < N % size ? 1 : 0);
        counts[r] = rows * N;
        displs[r] = offset;
        offset += rows * N;
    }
    int my_rows = counts[rank] / N;

    /* Every rank holds all of B plus its own block of A and C. */
    double *B = alloc_matrix(N, N);
    double *localA = alloc_matrix(my_rows, N);
    double *localC = alloc_matrix(my_rows, N);

    /* Only rank 0 holds the full A and C. */
    double *A = NULL, *C = NULL;
    if (rank == 0) {
        A = alloc_matrix(N, N);
        C = alloc_matrix(N, N);
        init_matrices(A, B, N);   /* initialization is NOT timed */
    }

    /* ---------------- timed region ---------------- */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Scatterv(A, counts, displs, MPI_DOUBLE,
                 localA, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t_comp_start = MPI_Wtime();
    matmul_rows(localA, B, localC, my_rows, N);
    double t_comp = MPI_Wtime() - t_comp_start;

    MPI_Gatherv(localC, counts[rank], MPI_DOUBLE,
                C, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t_total = MPI_Wtime() - t_start;
    /* ---------------- end timed region ------------ */

    /* Slowest rank's compute time; total - compute ~ communication cost. */
    double t_comp_max;
    MPI_Reduce(&t_comp, &t_comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N = %d, P = %d\n", N, size);
        if (N <= VERIFY_MAX_N)
            verify(A, B, C, N);
        else
            printf("  checksum = %.10f (verification skipped for N > %d)\n",
                   checksum(C, N), VERIFY_MAX_N);
        printf("TIMING mpi N=%d P=%d total=%.6f compute=%.6f\n",
               N, size, t_total, t_comp_max);
        free(A);
        free(C);
    }

    free(B);
    free(localA);
    free(localC);
    free(counts);
    free(displs);
    MPI_Finalize();
    return 0;
}
