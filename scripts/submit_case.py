import sys
from subprocess import check_output
from pathlib import Path

header = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --job-name={}
"""

header_gpu="#SBATCH --gpus-per-node=1"

module_loads = """
time_stamp=$(date +%Y-%m-%d-%T)
module load compiler/gnu/9.3
module load devel/cuda/11.0
module load toolkit/oneAPI/mpi/2021.1.1
"""

# Get machine/device info
#CUDA_DEVICE=$(/home/kit/scc/nq7776/code/cuda/deviceQuery/deviceQuery | grep "Device 0" | sed "s/Device 0://g" | sed  "s/[\"\ ]*//g")
omp_num_threads="OMP_NUM_THREADS={}"

cmd="""
source ~/OpenFOAM/OpenFOAM-8/etc/bashrc
cd ~/code/OBR
time python obr_run_cases.py --filter={} --folder {} --results_folder=/home/kit/scc/nq7776/OGL_DATA/{}/{}/{} --report results_{}{}.csv
"""

def write_script(case, executor, omp_threads=0):
    executor_filt = ["OMP", "HIP", "CUDA", "Serial"]
    executor_filt.remove(executor)
    describe_cmd = ["git", "describe", "--abbrev", "--tags", "--always", "--dirty"]

    gko_version = check_output(
            describe_cmd,cwd=Path("~/code/ginkgo/").expanduser()).decode("utf-8").replace("\n","")

    ogl_version = check_output(
            describe_cmd,cwd=Path("~/code/OGL/").expanduser()).decode("utf-8").replace("\n","")

    check_output(["mkdir", "-p", Path("~/OGL/{}/{}/{}".format(case,ogl_version,gko_version)).expanduser()])

    omp = "_" + str(omp_threads) if omp_threads else ""
    fn = "{}_{}{}.sh".format(case, executor, omp)

    with open(fn, "w") as fh:
        fh.write(header.format(fn))
        if executor == "CUDA":
            fh.write(header_gpu)
        fh.write(module_loads)
        if omp_threads:
            fh.write(omp_num_threads.format(omp_threads))
        fh.write(cmd.format(
            ",".join(executor_filt),
            case,
            case,
            ogl_version,
            gko_version,
            executor,
            omp
            ) )
    return fn

if __name__ == "__main__":
    # move_slurm()
    executor= ["OMP", "CUDA", "Serial"]

    for e in executor:
        partition = "gpu_4" if e == "CUDA" else "single"
        if e == "OMP":
            for t in  [1,2,4,5,8,10,16,20,32,40]:
                script = write_script("lidDrivenCavity3D", e, t)
                print(check_output(["sbatch", "-p", partition, script]))
        else:
            script = write_script("lidDrivenCavity3D", e)
            print(check_output(["sbatch", "-p", partition, script]))
