/**
 * syr2k.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "syr2k.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0



void init_arrays(int ni, int nj,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni))
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nj; j++) 
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
			B[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < ni; j++)
		{
			C[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}
}


void syr2kCpu(int ni, int nj,
		  DATA_TYPE alpha,
		  DATA_TYPE beta,
		  DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
		  DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj),
		  DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni))
{
	int i, j, k;

	/*    C := alpha*A*B' + alpha*B*A' + beta*C */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			C[i][j] *= beta;
		}
	}
	
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NI; j++)
		{
			for (k = 0; k < _PB_NJ; k++)
			{
				C[i][j] += alpha * A[i][k] * B[j][k];
				C[i][j] += alpha * B[i][k] * A[j][k];
			}
		}
	}
}


void compareResults(int ni, DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni))
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<ni; i++)
	{
		for (j=0; j<ni; j++)
		{
			if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{ 
				fail++;
			}
		}
	}
	
	// print results
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


void syr2k_kernel(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c,
                  sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < NI) && (j < NI))
	{
		c[i * NI + j] *= beta;
		
		int k;
		for(k = 0; k < NJ; k++)
		{
			c[i * NI + j] += alpha * a[i * NJ + k] * b[j * NJ + k] + alpha * b[i * NJ + k] * a[j * NJ + k];
		}
	}
}


void syr2kCuda(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj), 
		DATA_TYPE POLYBENCH_2D(C, NI, NI, ni, ni), DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NI, ni, ni)) 
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                             dpct::get_default_queue());
        B_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                             dpct::get_default_queue());
        C_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NI,
                                             dpct::get_default_queue());
        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ);
        dpct::get_default_queue().memcpy(B_gpu, B, sizeof(DATA_TYPE) * NI * NJ);
        dpct::get_default_queue().memcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NI).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid(
            1, (size_t)(ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_Y))),
            (size_t)ceil(((float)NI) / ((float)DIM_THREAD_BLOCK_X)));

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
                    syr2k_kernel(ni, nj, alpha, beta, A_gpu, B_gpu, C_gpu,
                                 item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NI)
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(B_gpu, dpct::get_default_queue());
        sycl::free(C_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, DATA_TYPE POLYBENCH_2D(C,NI,NI,ni,ni))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < ni; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, C[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NI,ni,ni);
	POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu,DATA_TYPE,NI,NI,ni,ni);

	init_arrays(ni, nj, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));
    
	GPU_argv_init();
	
	syr2kCuda(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start time for CPU */
	  	polybench_start_instruments;

		syr2kCpu(ni, nj, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, POLYBENCH_ARRAY(C_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(C_outputFromGpu);

  	return 0;
}

#include <polybench.c>