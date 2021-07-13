#./lidDrivenCavity3D/v0.1-108-gfe9e387/f584ccac99/IntelXeonGold6230CPU@2.10GHz/Test_2021-06-23-09:18:03/50/p-CG-OF-none-reference/lidDrivenCavity3D/log.1

import os
import sys
from pathlib import Path
from subprocess import check_output

base_folder = sys.argv[1]
print(base_folder)

for root, folder, files in os.walk(base_folder):
    for f in files:
        if "log" in f:
            try:
                target = Path(root).parents[3] / "Log"
                case = str(Path(root).parts[-3]) + "_" + str(Path(root).parts[-2]) + "_" + f
                check_output(["mkdir", "-p", target])
                check_output(["cp",  Path(root)/f, target/case])
            except Exception as e:
                pass

for root, folder, files in os.walk(base_folder):
    for f in folder:
        if "Test" in f:
            check_output(["rm", "-r",  Path(root)/f])
