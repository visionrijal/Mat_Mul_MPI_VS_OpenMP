#!/bin/bash
# Run all required MPI experiments on the Rushmore cluster.
# Edit hosts.txt first so it lists the four Rushmore nodes (slots=1 each).
# Output is saved in results/.

HOSTFILE=${HOSTFILE:-hosts.txt}
SMALL=250
SIZES=${SIZES:-"2048 3072"}
REPS=${REPS:-3}
MPIRUN="mpirun --hostfile $HOSTFILE --map-by node"

mkdir -p results

echo "=== Correctness (N=$SMALL) ===" | tee results/mpi_verify.txt
for P in 1 2 4; do
    $MPIRUN -np $P ./mpi_matmul $SMALL | tee -a results/mpi_verify.txt
done

echo "=== Performance ===" | tee results/mpi_perf.txt
for N in $SIZES; do
    for P in 1 2 4; do
        for r in $(seq 1 $REPS); do
            $MPIRUN -np $P ./mpi_matmul $N | tee -a results/mpi_perf.txt
        done
    done
done

echo
echo "Timing summary:"
grep TIMING results/mpi_perf.txt
