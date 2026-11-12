#!/bin/bash
# Run all required OpenMP experiments on the 16-vCPU multicore VM.
# Output is saved in results/.

SMALL=250
SIZES=${SIZES:-"2048 3072"}
REPS=${REPS:-3}
SCHED_N=${SCHED_N:-2048}     # matrix size for the scheduling experiment
SCHED_T=8         # fixed thread count for the scheduling experiment

mkdir -p results

echo "=== CPU evidence ===" | tee results/omp_cpu.txt
nproc | tee -a results/omp_cpu.txt
lscpu | grep -E "^CPU\(s\)|Model name|Thread|Core|Socket" | tee -a results/omp_cpu.txt

echo "=== Correctness (N=$SMALL) ===" | tee results/omp_verify.txt
for T in 1 2 4 8 16; do
    OMP_NUM_THREADS=$T OMP_SCHEDULE=static ./omp_matmul $SMALL | tee -a results/omp_verify.txt
done

echo "=== Scaling (schedule=static) ===" | tee results/omp_perf.txt
for N in $SIZES; do
    for T in 1 2 4 8 16; do
        for r in $(seq 1 $REPS); do
            OMP_NUM_THREADS=$T OMP_SCHEDULE=static ./omp_matmul $N | tee -a results/omp_perf.txt
        done
    done
done

echo "=== Scheduling experiment (N=$SCHED_N, T=$SCHED_T) ===" | tee results/omp_sched.txt
for S in "static" "static,1" "dynamic,1" "dynamic,16" "guided" "guided,16"; do
    for r in $(seq 1 $REPS); do
        OMP_NUM_THREADS=$SCHED_T OMP_SCHEDULE=$S ./omp_matmul $SCHED_N | tee -a results/omp_sched.txt
    done
done

echo
echo "Timing summary:"
grep TIMING results/omp_perf.txt results/omp_sched.txt
