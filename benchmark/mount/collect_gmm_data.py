import os
import re
import optparse

# go into every sub-directory,
# collect all files with "time" in the name
# extract d,k,name accoring to pattern "gmm_d20_K5_times_TorchScript.txt"
# add all lines from the file together (numbers given as 5.10850400e-02)
# create csv file with columns d,k,name1,name2,..,nameN
pattern = re.compile("gmm_d(\d+)_K(\d+)_times_(\w+).txt")

folder_name = "results/torch2/PyTorch_2S"
out_file = "results/torch2/times.csv"

opt = optparse.OptionParser()
opt.add_option("-f", "--folder", dest="folder_name", default=folder_name)
opt.add_option("-o", "--out", dest="out_name", default=out_file)
options, arguments = opt.parse_args()
folder_name = options.folder_name
out_file = options.out_name


folders = [folder_name]
for folder in folders:
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist")
        exit(1)

times = []
for folder in folders:
    files = [f for f in os.listdir(folder) if "time" in f]
    for file in files:
        # print("file", file)
        m = pattern.match(file)
        if m is None:
            print(f"Could not match {file}")
            continue
        d, k, name = m.groups()
        with open(os.path.join(folder, file), "r") as f:
            lines = f.readlines()
            # print("  ", lines)
            # print("  ", [float(l) for l in lines])
            time = sum([float(l) for l in lines])
            times.append([d, k, name, time])
            # print(f"  {d} {k} {name} {time}")

# create dataframe
names = set()
for t in times:
    names.add(t[2])

data = {}
for d, k, name, time in times:
    point = (d, k)
    if point not in data:
        data[point] = {}
    data[point][name] = time

# for p in data:
#     print(p, data[p])

# sep = ","
sep = "\t"

with open(out_file, "w") as f:
    # f.write("d,k," + ",".join(names) + "\n")
    f.write(sep.join(["d", "k"] + list(names)) + "\n")
    # first sort by d, then by k
    for p in sorted(data, key=lambda x: (int(x[0]), int(x[1]))):
        row = [p[0], p[1]]
        for name in names:
            if name in data[p]:
                # row.append("%.2f" % data[p][name])
                row.append(int(1000*data[p][name]))
            else:
                row.append(0)
        f.write(sep.join([str(x) for x in row]) + "\n")
