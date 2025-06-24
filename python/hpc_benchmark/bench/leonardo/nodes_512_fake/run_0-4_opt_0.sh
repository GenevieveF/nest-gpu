opt=0
for run in $(seq 0 4); do
    ./run_job.sh $opt $run
done
