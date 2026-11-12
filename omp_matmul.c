/*
 * omp_matmul.c - matrix multiplication C = A * B with OpenMP.
 *
 * Work sharing: the outer row loop (i) is split across threads with
 * "parallel for". Each row of C is written by exactly one thread, so there
 * are no race conditions and no locks/atomics are needed.
 *
 * Scheduling uses schedule(runtime), so the policy is chosen with the
 * OMP_SCHEDULE environment variable without recompiling, e.g.
 *     OMP_SCHEDULE=static   OMP_SCHEDULE="dynamic,16"   OMP_SCHEDULE=guided
 *
 * Usage: OMP_NUM_THREADS=T OMP_SCHEDULE=static ./omp_matmul N
 */
#include <omp.h>
#include "common.h"

static const char *schedule_name(omp_sched_t kind)
{
    switch (kind & ~omp_sched_monotonic) {
    case omp_sched_static:  return "static";
    case omp_sched_dynamic: return "dynamic";
    case omp_sched_guided:  return "guided";
    case omp_sched_auto:    return "auto";
    default:                return "other";
    }
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 250;
    if (N <= 0) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    double *A = alloc_matrix(N, N);
    double *B = alloc_matrix(N, N);
    double *C = alloc_matrix(N, N);
    init_matrices(A, B, N);   /* initialization is NOT timed */

    /* Diagnostics (outside the timed region). */
    int nthreads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }
    omp_sched_t kind;
    int chunk;
    omp_get_schedule(&kind, &chunk);
    printf("N = %d, threads = %d, logical CPUs = %d, schedule = %s, chunk = %d\n",
           N, nthreads, omp_get_num_procs(), schedule_name(kind), chunk);

    /* ---------------- timed region ---------------- */
    double t_start = omp_get_wtime();

    /*
     * A, B, C, N are shared (read-only except C, whose rows are disjoint per
     * thread). i is private as the loop variable; j, k, a are declared inside
     * the loop so each thread has its own copy.
     */
    #pragma omp parallel for schedule(runtime) default(none) shared(A, B, C, N)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.0;

        for (int k = 0; k < N; k++) {
            double a = A[i * N + k];
            for (int j = 0; j < N; j++)
                C[i * N + j] += a * B[k * N + j];
        }
    }

    double t_total = omp_get_wtime() - t_start;
    /* ---------------- end timed region ------------ */

    if (N <= VERIFY_MAX_N)
        verify(A, B, C, N);
    else
        printf("  checksum = %.10f (verification skipped for N > %d)\n",
               checksum(C, N), VERIFY_MAX_N);

    if (chunk > 0)
        printf("TIMING omp N=%d T=%d schedule=%s,%d time=%.6f\n",
               N, nthreads, schedule_name(kind), chunk, t_total);
    else
        printf("TIMING omp N=%d T=%d schedule=%s time=%.6f\n",
               N, nthreads, schedule_name(kind), t_total);

    free(A);
    free(B);
    free(C);
    return 0;
}
