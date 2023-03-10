import os
import csv
import re

with open('results_sycl-bench.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['bench_name', 'SYCL_CPU', 'SYCL_GPU']
    
    bench_result = header
    last_name = ""
    for filename in sorted(os.listdir("../sycl-bench/build")):
        
        def test_variant(ext, index):
            global last_name, bench_result
            if filename.endswith(ext):
                bench_name = os.path.splitext(os.path.splitext(filename)[0])[0]
                if last_name != bench_name:
                    writer.writerow(bench_result)
                    bench_result = [bench_name, '', '']
                    last_name = bench_name

                with open(f"../sycl-bench/build/{filename}") as f:
                    match = re.search(r"run-time-mean: (.+) \[s\]", f.read())
                    bench_result[index] = match.group(1)

        test_variant(".sycl_cpu.txt", 1)
        test_variant(".sycl_gpu.txt", 2)
    writer.writerow(bench_result)
    