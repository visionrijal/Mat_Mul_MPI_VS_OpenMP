/*
 * common.h - code shared by the MPI and OpenMP matrix multiplication programs.
 *
 * Matrices are N x N doubles stored row-major in 1D arrays:
 *     element (i, j) of M is M[i*N + j]
 */
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Sizes up to this value are checked against a sequential reference. */
#define VERIFY_MAX_N 512

/* Allocate an N-row matrix (or nrows x N block) and exit on failure. */
static double *alloc_matrix(int nrows, int N)
{
    double *M = malloc((size_t)nrows * N * sizeof(double));
    if (M == NULL) {
        fprintf(stderr, "Out of memory allocating %d x %d matrix\n", nrows, N);
        exit(1);
    }
    return M;
}

/*
 * Deterministic initialization: the values depend only on (i, j), so both
 * programs get exactly the same input on every run and every machine.
 * Values are in roughly [-0.5, 0.5].
 */
static void init_matrices(double *A, double *B, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = ((i * 7 + j * 3) % 11) / 11.0 - 0.5;
            B[i * N + j] = ((i * 5 + j * 13) % 17) / 17.0 - 0.5;
        }
    }
}

/*
 * Sequential kernel: C = A * B for 'nrows' rows of A and C.
 * A and C are nrows x N, B is N x N.
 *
 * Loop order is i-k-j so the innermost loop walks B and C row by row
 * (contiguous memory), which is much more cache friendly than i-j-k.
 * The OpenMP program uses this exact same loop body.
 */
static void matmul_rows(const double *A, const double *B, double *C,
                        int nrows, int N)
{
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.0;

        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
    }
}

/* Sum of all elements; printed so runs can be compared at any size. */
static double checksum(const double *C, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N * N; i++)
        sum += C[i];
    return sum;
}

/*
 * Compare C against a sequential reference for the whole matrix.
 * Prints the max error and a few sample elements. Returns 1 on pass.
 */
static int verify(const double *A, const double *B, const double *C, int N)
{
    double *Cref = alloc_matrix(N, N);
    matmul_rows(A, B, Cref, N, N);

    double max_err = 0.0;
    int ok = 1;
    for (int i = 0; i < N * N; i++) {
        double err = fabs(C[i] - Cref[i]);
        if (err > max_err)
            max_err = err;
        if (err > 1e-9 * fabs(Cref[i]) + 1e-12)
            ok = 0;
    }

    int samples[3] = {0, (N / 2) * N + N / 2, N * N - 1};
    for (int s = 0; s < 3; s++)
        printf("  C[%d] = %.10f   reference = %.10f\n",
               samples[s], C[samples[s]], Cref[samples[s]]);
    printf("  checksum = %.10f   reference checksum = %.10f\n",
           checksum(C, N), checksum(Cref, N));
    printf("  max abs error = %.3e\n", max_err);
    printf("VERIFICATION: %s\n", ok ? "PASSED" : "FAILED");

    free(Cref);
    return ok;
}

#endif
