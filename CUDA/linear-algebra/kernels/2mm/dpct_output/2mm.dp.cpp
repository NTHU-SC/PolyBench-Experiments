/**
 * 2mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "2mm.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0



void init_array(int ni, int nj, int nk, int nl, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), 
		DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj), 
		DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
	int i, j;

	*alpha = 32412;
	*beta = 2123;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NI;
		}
	}

	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}

	for (i = 0; i < nl; i++)
	{
		for (j = 0; j < nj; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / NK;	
		}
	}
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{
			if (percentDiff(D[i][j], D_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
        DPCT1093:2: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}


void mm2_kernel1(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *A, DATA_TYPE *B,
                 sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_NI) && (j < _PB_NJ))
	{ 
		tmp[i * NJ + j] = 0;
		int k;
		for (k = 0; k < _PB_NK; k++)
		{
			tmp[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
		}
	}
}


void mm2_kernel2(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *tmp, DATA_TYPE *C, DATA_TYPE *D,
                 sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_NI) && (j < _PB_NL))
	{ 
		D[i * NL + j] *= beta;
		int k;
		for (k = 0; k < _PB_NJ; k++)
		{
			D[i * NL + j] += tmp[i * NJ + k] * C[k * NL + j];
		}
	}
}


void mm2_cpu(int ni, int nj, int nk, int nl,
		DATA_TYPE alpha,
		DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj),
		DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
	int i, j, k;
	
	/* D := alpha*A*B*C + beta*D */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			tmp[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				tmp[i][j] += alpha * A[i][k] * B[k][j];
			}
		}
	}

	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			D[i][j] *= beta;
			for (k = 0; k < _PB_NJ; ++k)
			{
				D[i][j] += tmp[i][k] * C[k][j];
			}
		}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, D[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


void mm2Cuda(int ni, int nj, int nk, int nl, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(tmp,NI,NJ,ni,nj), 
	DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk), DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj), DATA_TYPE POLYBENCH_2D(C,NL,NJ,nl,nj), 
	DATA_TYPE POLYBENCH_2D(D,NI,NL,ni,nl), DATA_TYPE POLYBENCH_2D(D_outputFromGpu,NI,NL,ni,nl))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE *tmp_gpu;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;

        tmp_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                               dpct::get_default_queue());
        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NK,
                                             dpct::get_default_queue());
        B_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NK * NJ,
                                             dpct::get_default_queue());
        C_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NL * NJ,
                                             dpct::get_default_queue());
        D_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NL,
                                             dpct::get_default_queue());

        dpct::get_default_queue().memcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NI * NJ);
        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK);
        dpct::get_default_queue().memcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ);
        dpct::get_default_queue().memcpy(C_gpu, C, sizeof(DATA_TYPE) * NL * NJ);
        dpct::get_default_queue().memcpy(D_gpu, D, sizeof(DATA_TYPE) * NI * NL).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid1(1, (size_t)ceil(((float)NI) / ((float)block[1])),
                             (size_t)ceil(((float)NJ) / ((float)block[2])));
        sycl::range<3> grid2(1, (size_t)ceil(((float)NI) / ((float)block[1])),
                             (size_t)ceil(((float)NL) / ((float)block[2])));

        /* Start timer. */
  	polybench_start_instruments;

        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid1 * block, block),
            [=](sycl::nd_item<3> item_ct1) {
                    mm2_kernel1(ni, nj, nk, nl, alpha, beta, tmp_gpu, A_gpu,
                                B_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid2 * block, block),
            [=](sycl::nd_item<3> item_ct1) {
                    mm2_kernel2(ni, nj, nk, nl, alpha, beta, tmp_gpu, C_gpu,
                                D_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(D_outputFromGpu, D_gpu, sizeof(DATA_TYPE) * NI * NL)
            .wait();

        sycl::free(tmp_gpu, dpct::get_default_queue());
        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(B_gpu, dpct::get_default_queue());
        sycl::free(C_gpu, dpct::get_default_queue());
        sycl::free(D_gpu, dpct::get_default_queue());
}


int main(int argc, char** argv)
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;

	/* Variable declaration/allocation. */
	DATA_TYPE alpha;
	DATA_TYPE beta;
	POLYBENCH_2D_ARRAY_DECL(tmp,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
	POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NL,NJ,nl,nj);
	POLYBENCH_2D_ARRAY_DECL(D,DATA_TYPE,NI,NL,ni,nl);
	POLYBENCH_2D_ARRAY_DECL(D_outputFromGpu,DATA_TYPE,NI,NL,ni,nl);
	
	/* Initialize array(s). */
  	init_array(ni, nj, nk, nl, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));
	GPU_argv_init();

	mm2Cuda(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), 
		POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		mm2_cpu(ni, nj, nk, nl, alpha, beta, POLYBENCH_ARRAY(tmp), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, nl, POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(D_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(D_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(tmp);
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(D_outputFromGpu);

  	return 0;
}

#include <polybench.c>