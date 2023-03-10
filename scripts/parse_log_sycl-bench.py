import os
import csv
import re

with open('results_sycl-bench.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['bench_name', 'SYCL_CPU', 'SYCL_GPU'])
    
    bench_result = []
    for filename in sorted(os.listdir("../sycl-bench/build")):
        
        if filename.endswith(".sycl_cpu.txt"):
            bench_result.append(os.path.splitext(os.path.splitext(filename)[0])[0])

            with open(f"../sycl-bench/build/{filename}") as f:
                match = re.search(r"run-time-mean: (.+) \[s\]", f.read())
                bench_result.append(match.group(1))

        if filename.endswith(".sycl_gpu.txt"):
            with open(f"../sycl-bench/build/{filename}") as f:
                match = re.search(r"run-time-mean: (.+) \[s\]", f.read())
                bench_result.append(match.group(1))
        
            writer.writerow(bench_result)
            bench_result = []
