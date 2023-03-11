import os
import csv

DATASET = "STANDARD_DATASET"

with open("targets") as f:
    targets = f.read().splitlines()

with open('sycl-bench_sizes', 'w') as f:
    for bench in targets:
        if os.path.isdir(f"../CUDA/{bench}"):
            for filename in os.listdir(f"../CUDA/{bench}"):
                if filename.endswith(".cuh"):
                    with open(f"../CUDA/{bench}/{filename}") as ff:
                        cuh = ff.read().splitlines()
                    
                    args = {}

                    for i in range(len(cuh) - 1, -1, -1):
                        line = cuh[i]
                        if "ifdef" in line and DATASET in line:
                            k = i + 1
                            while True:
                                if "endif" in cuh[k]:
                                    break
                                else:
                                    _, key, val = cuh[k].split(" ")[-3:]
                                    args[key] = val
                                k += 1
                    
                    print(os.path.splitext(filename)[0], args)
                    f.write(f"export SIZE_{os.path.splitext(filename)[0]}={list(args.values())[-1]}\n")
                    f.write(f"export NAME_{os.path.splitext(filename)[0]}={os.path.basename(bench)}\n")
                            
