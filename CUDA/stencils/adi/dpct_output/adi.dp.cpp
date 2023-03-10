/**
 * adi.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "adi.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 2.5

#define GPU_DEVICE 0



void adi(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{
	for (int t = 0; t < _PB_TSTEPS; t++)
    	{
    		for (int i1 = 0; i1 < _PB_N; i1++)
		{
			for (int i2 = 1; i2 < _PB_N; i2++)
			{
				X[i1][i2] = X[i1][i2] - X[i1][(i2-1)] * A[i1][i2] / B[i1][(i2-1)];
				B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][(i2-1)];
			}
		}

	   	for (int i1 = 0; i1 < _PB_N; i1++)
		{
			X[i1][(N-1)] = X[i1][(N-1)] / B[i1][(N-1)];
		}

	   	for (int i1 = 0; i1 < _PB_N; i1++)
		{
			for (int i2 = 0; i2 < _PB_N-2; i2++)
			{
				X[i1][(N-i2-2)] = (X[i1][(N-2-i2)] - X[i1][(N-2-i2-1)] * A[i1][(N-i2-3)]) / B[i1][(N-3-i2)];
			}
		}

	   	for (int i1 = 1; i1 < _PB_N; i1++)
		{
			for (int i2 = 0; i2 < _PB_N; i2++) 
			{
		  		X[i1][i2] = X[i1][i2] - X[(i1-1)][i2] * A[i1][i2] / B[(i1-1)][i2];
		  		B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[(i1-1)][i2];
			}
		}

	   	for (int i2 = 0; i2 < _PB_N; i2++)
		{
			X[(N-1)][i2] = X[(N-1)][i2] / B[(N-1)][i2];
		}

	   	for (int i1 = 0; i1 < _PB_N-2; i1++)
		{
			for (int i2 = 0; i2 < _PB_N; i2++)
			{
		 	 	X[(N-2-i1)][i2] = (X[(N-2-i1)][i2] - X[(N-i1-3)][i2] * A[(N-3-i1)][i2]) / B[(N-2-i1)][i2];
			}
		}
    }
}


void init_array(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n))
{
  	int i, j;

  	for (i = 0; i < n; i++)
	{
    		for (j = 0; j < n; j++)
      		{
			X[i][j] = ((DATA_TYPE) i*(j+1) + 1) / N;
			A[i][j] = ((DATA_TYPE) (i-1)*(j+4) + 2) / N;
			B[i][j] = ((DATA_TYPE) (i+3)*(j+7) + 3) / N;
      		}
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_2D(B_cpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(B_fromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(X_cpu,N,N,n,n), 
			DATA_TYPE POLYBENCH_2D(X_fromGpu,N,N,n,n))
{
	int i, j, fail;
	fail = 0;
	
	// Compare b and x output on cpu and gpu
	for (i=0; i < n; i++) 
	{
		for (j=0; j < n; j++) 
		{
			if (percentDiff(B_cpu[i][j], B_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	for (i=0; i<n; i++) 
	{
		for (j=0; j<n; j++) 
		{
			if (percentDiff(X_cpu[i][j], X_fromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
        printf("setting device %d with name %s\n", GPU_DEVICE,
               deviceProp.get_name());
        /*
        DPCT1093:6: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}


void adi_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X,
                 sycl::nd_item<3> item_ct1)
{
        int i1 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

        if ((i1 < _PB_N))
	{
		for (int i2 = 1; i2 < _PB_N; i2++)
		{
			X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
		}
	}
}


void adi_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X,
                 sycl::nd_item<3> item_ct1)
{
        int i1 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

        if ((i1 < _PB_N))
	{
		X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];
	}
}
	

void adi_kernel3(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X,
                 sycl::nd_item<3> item_ct1)
{
        int i1 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

        if (i1 < _PB_N)
	{
		for (int i2 = 0; i2 < _PB_N-2; i2++)
		{
			X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];
		}
	}
}


void adi_kernel4(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1,
                 sycl::nd_item<3> item_ct1)
{
        int i2 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

        if (i2 < _PB_N)
	{
		X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
		B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
	}
}


void adi_kernel5(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X,
                 sycl::nd_item<3> item_ct1)
{
        int i2 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

        if (i2 < _PB_N)
	{
		X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];
	}
}


void adi_kernel6(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1,
                 sycl::nd_item<3> item_ct1)
{
        int i2 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

        if (i2 < _PB_N)
	{
		X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
	}
}


void adiCuda(int tsteps, int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n), DATA_TYPE POLYBENCH_2D(B,N,N,n,n), DATA_TYPE POLYBENCH_2D(X,N,N,n,n), 
	DATA_TYPE POLYBENCH_2D(B_outputFromGpu,N,N,n,n), DATA_TYPE POLYBENCH_2D(X_outputFromGpu,N,N,n,n))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE* A_gpu;
	DATA_TYPE* B_gpu;
	DATA_TYPE* X_gpu;

        A_gpu = (float *)sycl::malloc_device(N * N * sizeof(DATA_TYPE),
                                             dpct::get_default_queue());
        B_gpu = (float *)sycl::malloc_device(N * N * sizeof(DATA_TYPE),
                                             dpct::get_default_queue());
        X_gpu = (float *)sycl::malloc_device(N * N * sizeof(DATA_TYPE),
                                             dpct::get_default_queue());
        dpct::get_default_queue().memcpy(A_gpu, A, N * N * sizeof(DATA_TYPE));
        dpct::get_default_queue().memcpy(B_gpu, B, N * N * sizeof(DATA_TYPE));
        dpct::get_default_queue().memcpy(X_gpu, X, N * N * sizeof(DATA_TYPE)).wait();

        sycl::range<3> block1(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid1(1, 1, 1);
        grid1[2] = (size_t)(ceil(((float)N) / ((float)block1[2])));

        /* Start timer. */
  	polybench_start_instruments;

	for (int t = 0; t < _PB_TSTEPS; t++)
	{

                /*
                DPCT1049:0: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid1 * block1, block1),
                    [=](sycl::nd_item<3> item_ct1) {
                            adi_kernel1(n, A_gpu, B_gpu, X_gpu, item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid1 * block1, block1),
                    [=](sycl::nd_item<3> item_ct1) {
                            adi_kernel2(n, A_gpu, B_gpu, X_gpu, item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
                /*
                DPCT1049:2: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid1 * block1, block1),
                    [=](sycl::nd_item<3> item_ct1) {
                            adi_kernel3(n, A_gpu, B_gpu, X_gpu, item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();

                for (int i1 = 1; i1 < _PB_N; i1++)
		{
                        /*
                        DPCT1049:4: The work-group size passed to the SYCL
                        kernel may exceed the limit. To get the device limit,
                        query info::device::max_work_group_size. Adjust the
                        work-group size if needed.
                        */
                        dpct::get_default_queue().parallel_for(
                            sycl::nd_range<3>(grid1 * block1, block1),
                            [=](sycl::nd_item<3> item_ct1) {
                                    adi_kernel4(n, A_gpu, B_gpu, X_gpu, i1,
                                                item_ct1);
                            });
                        dpct::get_current_device().queues_wait_and_throw();
                }

                /*
                DPCT1049:3: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid1 * block1, block1),
                    [=](sycl::nd_item<3> item_ct1) {
                            adi_kernel5(n, A_gpu, B_gpu, X_gpu, item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();

                for (int i1 = 0; i1 < _PB_N-2; i1++)
		{
                        /*
                        DPCT1049:5: The work-group size passed to the SYCL
                        kernel may exceed the limit. To get the device limit,
                        query info::device::max_work_group_size. Adjust the
                        work-group size if needed.
                        */
                        dpct::get_default_queue().parallel_for(
                            sycl::nd_range<3>(grid1 * block1, block1),
                            [=](sycl::nd_item<3> item_ct1) {
                                    adi_kernel6(n, A_gpu, B_gpu, X_gpu, i1,
                                                item_ct1);
                            });
                        dpct::get_current_device().queues_wait_and_throw();
                }
	}

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue().memcpy(B_outputFromGpu, B_gpu,
                                         N * N * sizeof(DATA_TYPE));
        dpct::get_default_queue()
            .memcpy(X_outputFromGpu, X_gpu, N * N * sizeof(DATA_TYPE))
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(B_gpu, dpct::get_default_queue());
        sycl::free(X_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(X,N,N,n,n))

{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      fprintf(stderr, DATA_PRINTF_MODIFIER, X[i][j]);
      if ((i * N + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char *argv[])
{
	int tsteps = TSTEPS;
	int n = N;

	GPU_argv_init();

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X,DATA_TYPE,N,N,n,n);
	POLYBENCH_2D_ARRAY_DECL(X_outputFromGpu,DATA_TYPE,N,N,n,n);

	init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));

	adiCuda(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(B_outputFromGpu), 
		POLYBENCH_ARRAY(X_outputFromGpu));
	

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		adi(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(X));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(n, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu), POLYBENCH_ARRAY(X), POLYBENCH_ARRAY(X_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(X_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);
	POLYBENCH_FREE_ARRAY(X);
	POLYBENCH_FREE_ARRAY(X_outputFromGpu);

	return 0;
}

#include <polybench.c>