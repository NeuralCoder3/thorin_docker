import os
import sys
import subprocess
import psutil
import optparse
import re

src_folder = "python_src/"

out_folder = "results/torch2/PyTorch_2S"
module_file = src_folder+"modules/PyTorch/PyTorchGMM2S.py"
benchmark_folder = "./gmm/data/10k"

if __name__ == "__main__":
    opt = optparse.OptionParser()
    opt.add_option("-o", "--out", dest="out_name", default=out_folder)
    opt.add_option("-m", "--module", dest="module_name",
                   default=module_file)
    opt.add_option("-d", "--data", dest="data_name", default=benchmark_folder)
    options, arguments = opt.parse_args()
    out_folder = options.out_name
    module_file = options.module_name
    benchmark_folder = options.data_name

if not os.path.exists(module_file):
    print("Module file does not exist")
    exit(1)

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

if not out_folder.endswith("/"):
    out_folder += "/"

if not benchmark_folder.endswith("/"):
    benchmark_folder += "/"


def get_command(input_file):
    module = "GMM"
    runner = src_folder+"runner/main.py"
    return f"python {runner} {module} {module_file} {input_file} {out_folder} 0.5 10 10 10"


if not os.path.exists(benchmark_folder):
    print("Run from the root directory of the project")
    print("Input directory " + benchmark_folder + " does not exist")
    exit(1)


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


input_files = os.listdir(benchmark_folder)
input_files.sort(key=alphanum_key)

for input_file in input_files:
    if not input_file.endswith(".txt"):
        continue
    benchmark_file = os.path.join(benchmark_folder, input_file)
    command = get_command(benchmark_file)
    print(command)
    p = subprocess.Popen(command, shell=True)
    p.wait()
