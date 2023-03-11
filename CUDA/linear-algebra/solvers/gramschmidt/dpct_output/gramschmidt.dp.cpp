/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gramschmidt.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0



void gramschmidt(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj), DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
	int i,j,k;
	DATA_TYPE nrm;
	for (k = 0; k < _PB_NJ; k++)
	{
		nrm = 0;
		for (i = 0; i < _PB_NI; i++)
		{
			nrm += A[i][k] * A[i][k];
		}
		
		R[k][k] = sqrt(nrm);
		for (i = 0; i < _PB_NI; i++)
		{
			Q[i][k] = A[i][k] / R[k][k];
		}
		
		for (j = k + 1; j < _PB_NJ; j++)
		{
			R[k][j] = 0;
			for (i = 0; i < _PB_NI; i++)
			{
				R[k][j] += Q[i][k] * A[i][j];
			}
			for (i = 0; i < _PB_NI; i++)
			{
				A[i][j] = A[i][j] - Q[i][k] * R[k][j];
			}
		}
	}
}

/* Array initialization. */
void init_array(int ni, int nj,
		DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj),
		DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj))
{
	int i, j;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nj; j++) 
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
			Q[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
		}
	}

	for (i = 0; i < nj; i++)
	{
		for (j = 0; j < nj; j++)
		{
			R[i][j] = ((DATA_TYPE) i*(j+2)) / nj;
		}
	}
}

void compareResults(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,NI,NJ,ni,nj))
{
	int i, j, fail;
	fail = 0;

	for (i=0; i < ni; i++) 
	{
		for (j=0; j < nj; j++) 
		{
			if (percentDiff(A[i][j], A_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
        DPCT1093:3: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
        return;
}


void gramschmidt_kernel1(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k,
                         sycl::nd_item<3> item_ct1)
{
        int tid = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2);

        if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < _PB_NI; i++)
		{
			nrm += a[i * NJ + k] * a[i * NJ + k];
		}
                r[k * NJ + k] = sycl::sqrt(nrm);
        }
}


void gramschmidt_kernel2(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k,
                         sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (i < _PB_NI)
	{	
		q[i * NJ + k] = a[i * NJ + k] / r[k * NJ + k];
	}
}


void gramschmidt_kernel3(int ni, int nj, DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k,
                         sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if ((j > k) && (j < _PB_NJ))
	{
		r[k*NJ + j] = 0.0;

		int i;
		for (i = 0; i < _PB_NI; i++)
		{
			r[k*NJ + j] += q[i*NJ + k] * a[i*NJ + j];
		}
		
		for (i = 0; i < _PB_NI; i++)
		{
			a[i*NJ + j] -= q[i*NJ + k] * r[k*NJ + j];
		}
	}
}


void gramschmidtCuda(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(R,NJ,NJ,nj,nj), DATA_TYPE POLYBENCH_2D(Q,NI,NJ,ni,nj), DATA_TYPE POLYBENCH_2D(A_outputFromGpu,NI,NJ,ni,nj))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> gridKernel1(1, 1, 1);
        sycl::range<3> gridKernel2(
            1, 1, (size_t)ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X)));
        sycl::range<3> gridKernel3(
            1, 1, (size_t)ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X)));

        DATA_TYPE *A_gpu;
	DATA_TYPE *R_gpu;
	DATA_TYPE *Q_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                             dpct::get_default_queue());
        R_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NJ * NJ,
                                             dpct::get_default_queue());
        Q_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                             dpct::get_default_queue());
        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NJ).wait();

        /* Start timer. */
  	polybench_start_instruments;
	int k;
	for (k = 0; k < _PB_NJ; k++)
	{
                /*
                DPCT1049:0: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(gridKernel1 * block, block),
                    [=](sycl::nd_item<3> item_ct1) {
                            gramschmidt_kernel1(ni, nj, A_gpu, R_gpu, Q_gpu, k,
                                                item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(gridKernel2 * block, block),
                    [=](sycl::nd_item<3> item_ct1) {
                            gramschmidt_kernel2(ni, nj, A_gpu, R_gpu, Q_gpu, k,
                                                item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
                /*
                DPCT1049:2: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(gridKernel3 * block, block),
                    [=](sycl::nd_item<3> item_ct1) {
                            gramschmidt_kernel3(ni, nj, A_gpu, R_gpu, Q_gpu, k,
                                                item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
        }
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(A_outputFromGpu, A_gpu, sizeof(DATA_TYPE) * NI * NJ)
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(R_gpu, dpct::get_default_queue());
        sycl::free(Q_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj, DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j]);
	if (i % 20 == 0) fprintf (stderr, "\n");
    }

  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int ni = NI;
	int nj = NJ;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(A_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);
	POLYBENCH_2D_ARRAY_DECL(R,DATA_TYPE,NJ,NJ,nj,nj);
	POLYBENCH_2D_ARRAY_DECL(Q,DATA_TYPE,NI,NJ,ni,nj);
	
	init_array(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));
	
	GPU_argv_init();

	gramschmidtCuda(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q), POLYBENCH_ARRAY(A_outputFromGpu));

	#ifdef RUN_ON_CPU
	
		/* Start timer. */
	  	polybench_start_instruments;

		gramschmidt(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(R), POLYBENCH_ARRAY(Q));

		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(A_outputFromGpu));
	
	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(A_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(A_outputFromGpu);
	POLYBENCH_FREE_ARRAY(R);
	POLYBENCH_FREE_ARRAY(Q);  

    return 0;
}

#include <polybench.c>