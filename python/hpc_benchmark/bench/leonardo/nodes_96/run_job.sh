#!/bin/bash -x
nodes=96
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 opt run"
    echo "opt [0-3]: optimization gpu-memory-vs-speed, 3 for no spike recording"
    echo "run [0-9]: run number (changes output folder and random seed)"
else
    cat ../run_sbatch.templ | sed 's/__nodes__/$nodes/' > run_sbatch.sh
    sbatch run_sbatch.sh
fi
