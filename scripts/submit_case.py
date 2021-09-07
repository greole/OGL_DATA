"""
    This script generates and queues job submissions scripts for various super computers
    Additionally the required job metadata data is collected and results are written to the correct OGL_DATA folders.


    Usage:
        submit_case.py [options]

    Options:
        -h --help           Show this screen
        -v --version        Print version and exit
        --queue=<queue>     Which queue manager is used [default: local].
        --executor=<execs>  A list of executor [default: Serial].
"""

from docopt import docopt
import sys
from subprocess import check_output
from pathlib import Path

header = {
    "general": {
        "pre": """
source ~/OpenFOAM/OpenFOAM-8/etc/bashrc
cd {}
        """,
        "base_path": "~/data/code",
        "set_threads": "OMP_NUM_THREADS={}",
        "ogl_location":  "OGL",
        "ogl_data_location":  "OGL_DATA",
        "obr_location":  "OBR",
        "cmd": "time python obr_benchmark_cases.py --filter={} --folder {} --results_folder={} --report results_{}{}.csv",
    },
    "local": {
        "header": "#!/bin/bash\n",
        "module_loads": ""
              },
    "slurm": {
        "header": """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --job-name={}
""",
        "header_gpu": "#SBATCH --gpus-per-node=1",
        "module_loads": """
time_stamp=$(date +%Y-%m-%d-%T)
module load compiler/gnu/9.3
module load devel/cuda/11.0
module load toolkit/oneAPI/mpi/2021.1.1
""",
    },
    "pbs": {
        "header": "#PBS -l  nodes=1:gen9:ppn=2",
    },
          }

executor_avail = ["OMP", "CUDA", "Serial", "HIP", "DPCPP"]

def write_script(queue, case, executor, omp_threads=0):
    from copy import deepcopy
    print(executor)
    executors = deepcopy(executor_avail)
    executors.remove(executor)
    describe_cmd = ["git", "describe", "--abbrev", "--tags", "--always", "--dirty"]

    base_path = Path(header["general"]["base_path"])
    obr_path = base_path / header["general"]["obr_location"]

    gko_version = check_output(
            describe_cmd,cwd=(base_path / "ginkgo").expanduser()).decode("utf-8").replace("\n","")

    ogl_version = check_output(
            describe_cmd,cwd=(base_path / header["general"]["ogl_location"]).expanduser()).decode("utf-8").replace("\n","")

    results_path = Path(header["general"]["base_path"] + "/" +
                                      header["general"]["ogl_data_location"] +
                                      "/{}/{}/{}"
                                      .format(case,
                                              ogl_version,
                                              gko_version)).expanduser()


    # create ogl data directory
    check_output(["mkdir", "-p", results_path])

    omp = "_" + str(omp_threads) if omp_threads else ""
    fn = "{}_{}{}.sh".format(case, executor, omp)

    with open(fn, "w") as fh:
        header_ = header[queue]
        fh.write(header_["header"].format(fn))
        if executor == "CUDA":
            fh.write(header_["header_gpu"])
        fh.write(header_["module_loads"])
        if omp_threads:
            fh.write(header["general"]["set_threads"].format(omp_threads))
        fh.write(header["general"]["pre"].format(obr_path))
        fh.write(header["general"]cmd"].format(
            ",".join(executors),
            case,
            results_path,
            executor,
            omp
            ))
    return fn

if __name__ == "__main__":
    arguments = docopt(__doc__, version="0.1.0")
    executor = arguments["--executor"].split(",")
    queue = arguments["--queue"]


    for e in executor:
        if e == "OMP":
            for t in  [1,2,4,5,8,10,16,20,32,40]:
                script = write_script(queue, "lidDrivenCavity3D", e, t)
                #print(check_output(["sbatch", "-p", partition, script]))
        else:
            script = write_script("lidDrivenCavity3D", e)
            print(check_output(["sbatch", "-p", partition, script]))
            partition = "gpu_4" if e == "CUDA" else "single"
