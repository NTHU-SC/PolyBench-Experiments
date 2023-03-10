/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "2DConvolution.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0



void conv2D(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < _PB_NI - 1; ++i) // 0
	{
		for (j = 1; j < _PB_NJ - 1; ++j) // 1
		{
			B[i][j] = c11 * A[(i - 1)][(j - 1)]  +  c12 * A[(i + 0)][(j - 1)]  +  c13 * A[(i + 1)][(j - 1)]
				+ c21 * A[(i - 1)][(j + 0)]  +  c22 * A[(i + 0)][(j + 0)]  +  c23 * A[(i + 1)][(j + 0)] 
				+ c31 * A[(i - 1)][(j + 1)]  +  c32 * A[(i + 0)][(j + 1)]  +  c33 * A[(i + 1)][(j + 1)];
		}
	}
}



void init(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj))
{
	int i, j;

	for (i = 0; i < ni; ++i)
    	{
		for (j = 0; j < nj; ++j)
		{
			A[i][j] = (float)rand()/RAND_MAX;
        	}
    	}
}


void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj))
{
	int i, j, fail;
	fail = 0;
	
	// Compare outputs from CPU and GPU
	for (i=1; i < (ni-1); i++) 
	{
		for (j=1; j < (nj-1); j++) 
		{
			if (percentDiff(B[i][j], B_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
        DPCT1093:1: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}


void convolution2D_kernel(int ni, int nj, DATA_TYPE *A, DATA_TYPE *B,
                          sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < _PB_NI-1) && (j < _PB_NJ-1) && (i > 0) && (j > 0))
	{
		B[i * NJ + j] =  c11 * A[(i - 1) * NJ + (j - 1)]  + c21 * A[(i - 1) * NJ + (j + 0)] + c31 * A[(i - 1) * NJ + (j + 1)] 
			+ c12 * A[(i + 0) * NJ + (j - 1)]  + c22 * A[(i + 0) * NJ + (j + 0)] +  c32 * A[(i + 0) * NJ + (j + 1)]
			+ c13 * A[(i + 1) * NJ + (j - 1)]  + c23 * A[(i + 1) * NJ + (j + 0)] +  c33 * A[(i + 1) * NJ + (j + 1)];
	}
}


void convolution2DCuda(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj), 
			DATA_TYPE POLYBENCH_2D(B_outputFromGpu, NI, NJ, ni, nj))
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                             dpct::get_default_queue());
        B_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                             dpct::get_default_queue());
        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid(1, (size_t)ceil(((float)NJ) / ((float)block[1])),
                            (size_t)ceil(((float)NI) / ((float)block[2])));

        polybench_start_instruments;

        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item_ct1) {
                    convolution2D_kernel(ni, nj, A_gpu, B_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");

  	polybench_stop_instruments;
  	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * NI * NJ)
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(B_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, B[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size */
	int ni = NI;
	int nj = NJ;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);

	//initialize the arrays
	init(ni, nj, POLYBENCH_ARRAY(A));
	
	GPU_argv_init();

	convolution2DCuda(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#ifdef RUN_ON_CPU
	
	 	/* Start timer. */
	  	polybench_start_instruments;

		conv2D(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, nj, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(B_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
  	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);
	
	return 0;
}

#include <polybench.c>