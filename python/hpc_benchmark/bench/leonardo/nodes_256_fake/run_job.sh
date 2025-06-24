#!/bin/bash
nodes=256
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 opt run"
    echo "opt [0-3]: optimization gpu-memory-vs-speed, 3 for no spike recording"
    echo "run [0-9]: run number (changes output folder and random seed)"
else
    cat ../run_sbatch_fake.templ | sed "s/__nodes__/$nodes/" > run_sbatch_fake.sh
    echo "sbatch run_sbatch_fake.sh $1 $2"
    sbatch run_sbatch_fake.sh $1 $2

fi
