/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>

#define POLYBENCH_TIME 1

#include "bicg.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#ifndef M_PI
#define M_PI 3.14159
#endif



void init_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx))
{
	int i, j;
	
	for (i = 0; i < ny; i++)
	{
    		p[i] = i * M_PI;
	}

	for (i = 0; i < nx; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < ny; j++)
		{
      			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
}


void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_1D(s,NY,ny), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
		DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
	int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<nx; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<ny; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
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


//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
void bicg_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s,
                  sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (j < _PB_NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < _PB_NX; i++)
		{
			s[j] += r[i] * A[i * NY + j];
		}
	}	
}


//Distributed (split) from initial loop to allow parallelism
void bicg_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q,
                  sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (i < _PB_NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < _PB_NY; j++)
		{
			q[i] += A[i * NY + j] * p[j];
		}
	}
}


void bicg_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
		DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	int i,j;
	
  	for (i = 0; i < _PB_NY; i++)
	{
		s[i] = 0.0;
	}

	for (i = 0; i < _PB_NX; i++)
	{
		q[i] = 0.0;
		for (j = 0; j < _PB_NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i][j];
	    		q[i] = q[i] + A[i][j] * p[j];
	  	}
	}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx, int ny,
		 DATA_TYPE POLYBENCH_1D(s,NY,ny),
		 DATA_TYPE POLYBENCH_1D(q,NX,nx))

{
  int i;

  for (i = 0; i < ny; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, s[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, q[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}


void bicgCuda(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
	DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx), DATA_TYPE POLYBENCH_1D(s_outputFromGpu,NY,ny), 
	DATA_TYPE POLYBENCH_1D(q_outputFromGpu,NX,nx))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE *A_gpu;
	DATA_TYPE *q_gpu;
	DATA_TYPE *p_gpu;
	DATA_TYPE *r_gpu;
	DATA_TYPE *s_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY,
                                             dpct::get_default_queue());
        r_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX,
                                             dpct::get_default_queue());
        s_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NY,
                                             dpct::get_default_queue());
        p_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NY,
                                             dpct::get_default_queue());
        q_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX,
                                             dpct::get_default_queue());
        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY);
        dpct::get_default_queue().memcpy(r_gpu, r, sizeof(DATA_TYPE) * NX);
        dpct::get_default_queue().memcpy(s_gpu, s, sizeof(DATA_TYPE) * NY);
        dpct::get_default_queue().memcpy(p_gpu, p, sizeof(DATA_TYPE) * NY);
        dpct::get_default_queue().memcpy(q_gpu, q, sizeof(DATA_TYPE) * NX).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid1(1, 1, (size_t)(ceil(((float)NY) / ((float)block[2]))));
        sycl::range<3> grid2(1, 1, (size_t)(ceil(((float)NX) / ((float)block[2]))));

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
                    bicg_kernel1(nx, ny, A_gpu, r_gpu, s_gpu, item_ct1);
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
                    bicg_kernel2(nx, ny, A_gpu, p_gpu, q_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue().memcpy(s_outputFromGpu, s_gpu,
                                         sizeof(DATA_TYPE) * NY);
        dpct::get_default_queue()
            .memcpy(q_outputFromGpu, q_gpu, sizeof(DATA_TYPE) * NX)
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(r_gpu, dpct::get_default_queue());
        sycl::free(s_gpu, dpct::get_default_queue());
        sycl::free(p_gpu, dpct::get_default_queue());
        sycl::free(q_gpu, dpct::get_default_queue());
}


int main(int argc, char** argv)
{
	int nx = NX;
	int ny = NY;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(s,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(p,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(r,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));

	GPU_argv_init();

	bicgCuda(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q), 
		POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		bicg_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(nx, ny, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q), 
			POLYBENCH_ARRAY(q_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu)));
	
	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(r);
	POLYBENCH_FREE_ARRAY(s);
	POLYBENCH_FREE_ARRAY(p);
	POLYBENCH_FREE_ARRAY(q);
	POLYBENCH_FREE_ARRAY(s_outputFromGpu);
	POLYBENCH_FREE_ARRAY(q_outputFromGpu);

  	return 0;
}

#include <polybench.c>