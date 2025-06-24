opt=3
for run in $(seq 0 4); do
    ./run_job.sh $opt $run
done
