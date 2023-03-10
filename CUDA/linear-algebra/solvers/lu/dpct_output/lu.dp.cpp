/**
 * lu.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "lu.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0



void lu(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
	for (int k = 0; k < _PB_N; k++)
    {
		for (int j = k + 1; j < _PB_N; j++)
		{
			A[k][j] = A[k][j] / A[k][k];
		}

		for (int i = k + 1; i < _PB_N; i++)
		{
			for (int j = k + 1; j < _PB_N; j++)
			{
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
		}
    }
}


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j + 1) / N;
		}
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(A_cpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(A_cpu[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
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


void lu_kernel1(int n, DATA_TYPE *A, int k, sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if ((j > k) && (j < _PB_N))
	{
		A[k*N + j] = A[k*N + j] / A[k*N + k];
	}
}


void lu_kernel2(int n, DATA_TYPE *A, int k, sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i > k) && (j > k) && (i < _PB_N) && (j < _PB_N))
	{
		A[i*N + j] = A[i*N + j] - A[i*N + k] * A[k*N + j];
	}
}


void luCuda(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,N,N,n,n))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE* AGpu;

        AGpu = (float *)sycl::malloc_device(N * N * sizeof(DATA_TYPE),
                                            dpct::get_default_queue());
        dpct::get_default_queue().memcpy(AGpu, A, N * N * sizeof(DATA_TYPE)).wait();

        sycl::range<3> block1(1, DIM_THREAD_BLOCK_KERNEL_1_Y,
                              DIM_THREAD_BLOCK_KERNEL_1_X);
        sycl::range<3> block2(1, DIM_THREAD_BLOCK_KERNEL_2_Y,
                              DIM_THREAD_BLOCK_KERNEL_2_X);
        sycl::range<3> grid1(1, 1, 1);
        sycl::range<3> grid2(1, 1, 1);

        /* Start timer. */
  	polybench_start_instruments;

	for (int k = 0; k < N; k++)
	{
                grid1[2] = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block1[2])));
                /*
                DPCT1049:0: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid1 * block1, block1),
                    [=](sycl::nd_item<3> item_ct1) {
                            lu_kernel1(n, AGpu, k, item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();

                grid2[2] = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2[2])));
                grid2[1] = (unsigned int)(ceil((float)(N - (k + 1)) / ((float)block2[1])));
                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid2 * block2, block2),
                    [=](sycl::nd_item<3> item_ct1) {
                            lu_kernel2(n, AGpu, k, item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
        }
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(A_outputFromGpu, AGpu, N * N * sizeof(DATA_TYPE))
            .wait();
        sycl::free(AGpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
      if ((i * n + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}
	

int main(int argc, char *argv[])
{
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
  	POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(A));

	GPU_argv_init();
	luCuda(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
	

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		lu(n, POLYBENCH_ARRAY(A));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);

   	return 0;
}

#include <polybench.c>