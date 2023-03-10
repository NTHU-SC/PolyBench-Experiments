/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define POLYBENCH_TIME 1

#include "gesummv.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f



void gesummv(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_1D(tmp,N,n),
		DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n))
{
	int i, j;
	
	for (i = 0; i < _PB_N; i++)
	{
		tmp[i] = 0;
		y[i] = 0;
		for (j = 0; j < _PB_N; j++)
		{
			tmp[i] = A[i][j] * x[j] + tmp[i];
			y[i] = B[i][j] * x[j] + y[i];
		}
		
		y[i] = alpha * tmp[i] + beta * y[i];
	}
}


void init(int n, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), 
	DATA_TYPE POLYBENCH_1D(x,N,n))
{
  	int i, j;

	*alpha = 43532;
	*beta = 12313;

 	for (i = 0; i < n; i++)
    	{
    		x[i] = ((DATA_TYPE) i) / N;
      	
		for (j = 0; j < n; j++) 
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
			B[i][j] = ((DATA_TYPE) i*j) / n;
		}
    }
}


void compareResults(int n, DATA_TYPE POLYBENCH_1D(y,N,n), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,N,n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<n; i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
        dpct::device_info deviceProp;
        dpct::dev_mgr::instance()
            .get_device(GPU_DEVICE)
            .get_device_info(deviceProp);
        printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.get_name());
        /*
        DPCT1093:1: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}


void gesummv_kernel(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* tmp, DATA_TYPE* x, DATA_TYPE* y,
                    sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (i < _PB_N)
	{
		int j;
		for(j = 0; j < _PB_N; j++)
		{	
			tmp[i] += A[i * N + j] * x[j];
			y[i] += B[i * N + j] * x[j];
		}
		y[i] = alpha * tmp[i] + beta  * y[i];
	}
}

void gesummvCuda(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), 
		DATA_TYPE POLYBENCH_1D(tmp,N,n), DATA_TYPE POLYBENCH_1D(x,N,n), DATA_TYPE POLYBENCH_1D(y,N,n),  
		DATA_TYPE POLYBENCH_1D(y_outputFromGpu,N,n))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N * N,
                                             dpct::get_default_queue());
        B_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N * N,
                                             dpct::get_default_queue());
        x_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                             dpct::get_default_queue());
        y_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                             dpct::get_default_queue());
        tmp_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                               dpct::get_default_queue());

        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N);
        dpct::get_default_queue().memcpy(B_gpu, B, sizeof(DATA_TYPE) * N * N);
        dpct::get_default_queue().memcpy(x_gpu, x, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(y_gpu, y, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * N).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid(1, 1, (unsigned int)ceil(((float)N) / ((float)block[2])));

        /* Start timer. */
  	polybench_start_instruments;

        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) {
                    gesummv_kernel(n, alpha, beta, A_gpu, B_gpu, tmp_gpu, x_gpu,
                                   y_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * N)
            .wait();
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int n = N;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,N,n);

	init(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(x));
	
	GPU_argv_init();
	gesummvCuda(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),  
		POLYBENCH_ARRAY(y_outputFromGpu));
	
	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		gesummv(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y));
		
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(n, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);  
	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);

	return 0;
}

#include <polybench.c>