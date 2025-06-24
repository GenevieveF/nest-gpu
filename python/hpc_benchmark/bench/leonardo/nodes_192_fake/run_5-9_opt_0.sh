opt=0
for run in $(seq 5 9); do
    ./run_job.sh $opt $run
done
