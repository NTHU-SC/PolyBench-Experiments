BASENAME := ${basename ${CUFILES}}

dont_run_cpu:
	sed -i '/#define RUN_ON_CPU/d' ${CUFILES}

convert:
	intercept-build make cuda
	dpct -p compile_commands.json
	sed -i 's/&deviceProp, GPU_DEVICE/GPU_DEVICE/g' dpct_output/${BASENAME}.dp.cpp

dpct_cpu:
	icpx -fsycl -O3 dpct_output/${BASENAME}.dp.cpp -I${PATH_TO_UTILS} -o ${BASENAME}.dpct_cpu.exe

dpct_gpu:
	clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 \
		-O3 dpct_output/${BASENAME}.dp.cpp -I${PATH_TO_UTILS} -o ${BASENAME}.dpct_gpu.exe 

cuda:
	nvcc -O3 ${CUFILES} -I${PATH_TO_UTILS} -o ${BASENAME}.cuda.exe

clean:
	rm -f *~ *.exe
	rm -rf dpct_output