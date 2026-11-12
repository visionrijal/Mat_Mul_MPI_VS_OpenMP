# CSC 718 Assignment 3: MPI vs OpenMP Matrix Multiplication

## Files
| File | Purpose |
|---|---|
| `common.h` | Shared code: matrix initialization, sequential kernel, checksum, verification |
| `mpi_matmul.c` | MPI implementation (row-wise decomposition) |
| `omp_matmul.c` | OpenMP implementation (`parallel for` over rows) |
| `Makefile` | Builds both programs |
| `run_mpi.sh` | Runs all MPI experiments (Rushmore cluster) |
| `run_omp.sh` | Runs all OpenMP experiments (16-vCPU VM) |
| `hosts.txt` | MPI hostfile: one slot per Rushmore node |

The sequential reference is `matmul_rows()` in `common.h`. Both programs call it
from `verify()` for small sizes, so no separate reference program is needed.

## Build
```
make            # builds both
make mpi_matmul # MPI only  (mpicc -O2 -Wall -o mpi_matmul mpi_matmul.c -lm)
make omp_matmul # OpenMP only (gcc -O2 -Wall -fopenmp -o omp_matmul omp_matmul.c -lm)
```
Compiler optimization flags: `-O2` for both programs.

## Run
**Matrix size** is the first command-line argument, `N` (default 250).

**MPI** (one process per node):
```
mpirun --hostfile hosts.txt --map-by node -np P ./mpi_matmul N
```
where P = 1, 2 or 4. The `mpi_matmul` binary must be at the same path on every
node (shared filesystem, or copy it with `scp`). The commands above are for
Open MPI. With MPICH, use `mpiexec -f hosts.txt -ppn 1 -n P ./mpi_matmul N`,
and drop the `slots=1` from the hostfile.

**OpenMP**:
```
OMP_NUM_THREADS=T OMP_SCHEDULE=static ./omp_matmul N
```
where T = 1, 2, 4, 8 or 16. Always set `OMP_SCHEDULE`. The loop uses
`schedule(runtime)`, and gcc's default when the variable is unset is `dynamic,1`.
Examples for the scheduling experiment: `OMP_SCHEDULE=dynamic,16`,
`OMP_SCHEDULE=guided`.

**Everything at once:**
```
./run_mpi.sh    # on Rushmore, after editing hosts.txt
./run_omp.sh    # on the OpenMP VM
```
Output goes to `results/`. Each configuration runs 3 times; report the
median or minimum. Lines starting with `TIMING` are easy to `grep`.

## Matrix sizes and input
- Correctness: N = 250. It is deliberately not divisible by 4, to test uneven
  row distribution.
- Performance: N = 2048 and N = 3072.
- Input: `A[i][j] = ((7i + 3j) mod 11)/11 - 0.5` and
  `B[i][j] = ((5i + 13j) mod 17)/17 - 0.5`, all doubles. These are
  deterministic and identical in both programs.

## Verification
For N <= 512, the full parallel result is compared element by element with a
sequential computation (relative tolerance 1e-9). The output shows sample
elements, checksums, the max error, and `PASSED`/`FAILED`. For larger N, only
the checksum is printed. Use it to confirm MPI and OpenMP produce the same
result.

## Timing method
- **MPI** (`MPI_Wtime`): matrices are initialized on rank 0, then
  `MPI_Barrier`, then the timer starts. The timed region covers Scatterv(A),
  Bcast(B), local compute and Gatherv(C), and stops once rank 0 has the full
  C. `compute` is the slowest rank's compute-only time, so
  `total - compute` estimates communication cost.
- **OpenMP** (`omp_get_wtime`): the timer covers only the
  `parallel for` multiplication.
- In both programs, initialization, verification, and hostname or thread
  printing are outside the timed region.

## Design summary
- **MPI**: 1D row-wise decomposition. Each rank gets `N/P` rows of A and C
  (the first `N%P` ranks get one extra row) plus a full copy of B. Rank 0 also
  computes its own block.
- **OpenMP**: the outer row loop is split among threads. A, B, C and N are
  shared, and the loop indices and temporaries are private. Each thread writes
  only its own rows of C, so no synchronization is needed beyond the implicit
  barrier at the end of the loop.
- **Scaling-study schedule**: `static` is used because every row costs the
  same work (N² multiply-adds). Equal contiguous chunks balance the load with
  the lowest scheduling overhead and good locality.
- **Kernel**: both programs use the same i-k-j triple loop (the
  standard O(N³) algorithm, no BLAS). The i-k-j order makes the inner loop
  read memory contiguously.

## Assumptions / limitations
- Square matrices only. N must be small enough that rank 0 can hold A, B and
  C (3 x N² doubles; about 216 MB at N = 3072).
- Rank 0 holds the full matrices and every rank holds all of B. This is
  simple, but uses more memory than a 2D decomposition would.
- The kernel is deliberately not tiled or vectorized by hand, to keep both
  implementations simple and comparable.
