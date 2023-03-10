/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#define POLYBENCH_TIME 1

#include "mvt.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0



void init_array(int n, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		x1[i] = ((DATA_TYPE) i) / N;
		x2[i] = ((DATA_TYPE) i + 1) / N;
		y1[i] = ((DATA_TYPE) i + 3) / N;
		y2[i] = ((DATA_TYPE) i + 4) / N;
		for (j = 0; j < n; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
	}
}



void runMvt(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y1, N, n), DATA_TYPE POLYBENCH_1D(y2, N, n))
{
	int i, j;
	
	for (i=0; i<_PB_N; i++) 
	{
		for (j=0; j<N; j++) 
		{
       		x1[i] = x1[i] + a[i][j] * y1[j];
        	}
    	}
	
	for (i=0; i<_PB_N; i++) 
	{
		for (j=0; j<_PB_N; j++) 
		{
 		      	x2[i] = x2[i] + a[j][i] * y2[j];
      		}
    	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<n; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
        DPCT1093:2: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}


void mvt_kernel1(int n, DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1,
                 sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (i < _PB_N)
	{
		int j;
		for(j=0; j < _PB_N; j++)
		{
			x1[i] += a[i * N + j] * y_1[j];
		}
	}
}


void mvt_kernel2(int n, DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2,
                 sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (i < _PB_N)
	{
		int j;
		for(j=0; j < _PB_N; j++)
		{
			x2[i] += a[j * N + i] * y_2[j];	
		}
	}
}

void mvtCuda(int n, DATA_TYPE POLYBENCH_2D(a, N, N, n, n), DATA_TYPE POLYBENCH_1D(x1, N, n), DATA_TYPE POLYBENCH_1D(x2, N, n), DATA_TYPE POLYBENCH_1D(y_1, N, n), DATA_TYPE POLYBENCH_1D(y_2, N, n), 
			DATA_TYPE POLYBENCH_1D(x1_outputFromGpu, N, n), DATA_TYPE POLYBENCH_1D(x2_outputFromGpu, N, n))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE* a_gpu;
	DATA_TYPE* x1_gpu;
	DATA_TYPE* x2_gpu;
	DATA_TYPE* y_1_gpu;
	DATA_TYPE* y_2_gpu;

        a_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N * N,
                                             dpct::get_default_queue());
        x1_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                              dpct::get_default_queue());
        x2_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                              dpct::get_default_queue());
        y_1_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                               dpct::get_default_queue());
        y_2_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                               dpct::get_default_queue());
        dpct::get_default_queue().memcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N);
        dpct::get_default_queue().memcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid(
            1, 1, (size_t)ceil((float)N / ((float)DIM_THREAD_BLOCK_X)));

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
                    mvt_kernel1(n, a_gpu, x1_gpu, y_1_gpu, item_ct1);
            });
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) {
                    mvt_kernel2(n, a_gpu, x2_gpu, y_2_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue().memcpy(x1_outputFromGpu, x1_gpu,
                                         sizeof(DATA_TYPE) * N);
        dpct::get_default_queue()
            .memcpy(x2_outputFromGpu, x2_gpu, sizeof(DATA_TYPE) * N)
            .wait();

        sycl::free(a_gpu, dpct::get_default_queue());
        sycl::free(x1_gpu, dpct::get_default_queue());
        sycl::free(x2_gpu, dpct::get_default_queue());
        sycl::free(y_1_gpu, dpct::get_default_queue());
        sycl::free(y_2_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(x1,N,n),
		 DATA_TYPE POLYBENCH_1D(x2,N,n))

{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, x1[i]);
    fprintf (stderr, DATA_PRINTF_MODIFIER, x2[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
}


int main(int argc, char *argv[])
{
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(a,DATA_TYPE,N,N,n,n);
	POLYBENCH_1D_ARRAY_DECL(x1,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(x2,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(x1_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(x2_outputFromGpu,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y_1,DATA_TYPE,N,n);
	POLYBENCH_1D_ARRAY_DECL(y_2,DATA_TYPE,N,n);

	init_array(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));
	
	GPU_argv_init();

	mvtCuda(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu));

	#ifdef RUN_ON_CPU
	
		/* Start timer. */
	  	polybench_start_instruments;

		//run the algorithm on the CPU
		runMvt(n, POLYBENCH_ARRAY(a), POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(y_1), POLYBENCH_ARRAY(y_2));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(n, POLYBENCH_ARRAY(x1), POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2), POLYBENCH_ARRAY(x2_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x1_outputFromGpu), POLYBENCH_ARRAY(x2_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(a);
	POLYBENCH_FREE_ARRAY(x1);
	POLYBENCH_FREE_ARRAY(x2);
	POLYBENCH_FREE_ARRAY(x1_outputFromGpu);
	POLYBENCH_FREE_ARRAY(x2_outputFromGpu);
	POLYBENCH_FREE_ARRAY(y_1);
	POLYBENCH_FREE_ARRAY(y_2);

  	return 0;
}

#include <polybench.c>