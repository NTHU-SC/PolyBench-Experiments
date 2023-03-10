import os
import csv

with open("targets") as f:
    targets = f.read().splitlines()

with open('results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['bench_name', 'CUDA', 'DPCT_CPU', 'DPCT_GPU'])
    
    for bench in targets:
        if os.path.isdir(f"../CUDA/{bench}"):
            bench_result = []
            print(os.path.basename(f"../CUDA/{bench}"))
            bench_result.append(os.path.basename(f"../CUDA/{bench}"))

            def get_time(variant):
                if os.path.isfile(f"../CUDA/{bench}/output.{variant}.txt"):
                    with open(f"../CUDA/{bench}/output.{variant}.txt") as f:
                        bench_result.append(f.read().splitlines()[-1])
                else:
                    bench_result.append("")
            
            get_time("cuda")
            get_time("dpct_cpu")
            get_time("dpct_gpu")
            
            # write parsed result into csv fule
            writer.writerow(bench_result)
