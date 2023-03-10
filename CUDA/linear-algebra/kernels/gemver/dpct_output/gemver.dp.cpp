/**
 * gemver.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "gemver.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0



void gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n), DATA_TYPE POLYBENCH_1D(u1, N, n), DATA_TYPE POLYBENCH_1D(v1, N, n), 
	DATA_TYPE POLYBENCH_1D(u2, N, n), DATA_TYPE POLYBENCH_1D(v2, N, n), DATA_TYPE POLYBENCH_1D(w, N, n), DATA_TYPE POLYBENCH_1D(x, N, n), DATA_TYPE POLYBENCH_1D(y, N, n), 
	DATA_TYPE POLYBENCH_1D(z, N, n))
{
	int i,j;
	
  	for (i = 0; i < _PB_N; i++)
	{
    		for (j = 0; j < _PB_N; j++)
		{
      			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}

  	for (i = 0; i < _PB_N; i++)
	{
    		for (j = 0; j < _PB_N; j++)
		{
      			x[i] = x[i] + beta * A[j][i] * y[j];
		}
	}

  	for (i = 0; i < _PB_N; i++)
	{
    		x[i] = x[i] + z[i];
	}

  	for (i = 0; i < _PB_N; i++)
	{
    		for (j = 0; j < _PB_N; j++)
		{
      			w[i] = w[i] +  alpha * A[i][j] * x[j];
		}
	}
}


void init(int n, DATA_TYPE *alpha,
	DATA_TYPE *beta,
	DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
	DATA_TYPE POLYBENCH_1D(u1,N,n),
	DATA_TYPE POLYBENCH_1D(v1,N,n),
	DATA_TYPE POLYBENCH_1D(u2,N,n),
	DATA_TYPE POLYBENCH_1D(v2,N,n),
	DATA_TYPE POLYBENCH_1D(w,N,n),
	DATA_TYPE POLYBENCH_1D(x,N,n),
	DATA_TYPE POLYBENCH_1D(y,N,n),
	DATA_TYPE POLYBENCH_1D(z,N,n))
{
	int i, j;

	*alpha = 43532;
	*beta = 12313;

  	for (i = 0; i < N; i++)
	{
	    	u1[i] = i;
	    	u2[i] = (i+1)/N/2.0;
	    	v1[i] = (i+1)/N/4.0;
	    	v2[i] = (i+1)/N/6.0;
	    	y[i] = (i+1)/N/8.0;
	    	z[i] = (i+1)/N/9.0;
	    	x[i] = 0.0;
	    	w[i] = 0.0;

    		for (j = 0; j < N; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / N;
		}
	}
}


void compareResults(int n, DATA_TYPE POLYBENCH_1D(w1, N, n), DATA_TYPE POLYBENCH_1D(w2, N, n))
{
	int i, fail;
	fail = 0;
	
	for (i=0; i < N; i++) 
	{
		if (percentDiff(w1[i], w2[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
			fail++;
		}
	}
		
	// Print results
	printf("Number of misses: %d\n", fail);
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
}


void gemver_kernel1(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *v1, DATA_TYPE *v2, DATA_TYPE *u1, DATA_TYPE *u2,
                    sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_N) && (j < _PB_N))
	{
		a[i * N + j] += u1[i] * v1[j] + u2[i] * v2[j];
	}
}


void gemver_kernel2(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z,
                    sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (i < _PB_N)
	{
		int j;
		for(j = 0; j < _PB_N; j++) 
		{
			x[i] += beta * a[j * N + i] * y[j];
		}
		x[i] += z[i];
	}
}


void gemver_kernel3(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *w,
                    sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if ((i >= 0) && (i < _PB_N))
	{
		int j;
		for(j = 0; j < _PB_N; j++)
		{ 
			w[i] += alpha * a[i*N + j] * x[j];
		}
	}
}


void gemverCuda(int n, DATA_TYPE alpha, DATA_TYPE beta,
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		DATA_TYPE POLYBENCH_1D(u1,N,n),
		DATA_TYPE POLYBENCH_1D(v1,N,n),
		DATA_TYPE POLYBENCH_1D(u2,N,n),
		DATA_TYPE POLYBENCH_1D(v2,N,n),
		DATA_TYPE POLYBENCH_1D(w,N,n),
		DATA_TYPE POLYBENCH_1D(w_outputFromGpu,N,n),
		DATA_TYPE POLYBENCH_1D(x,N,n),
		DATA_TYPE POLYBENCH_1D(y,N,n),
		DATA_TYPE POLYBENCH_1D(z,N,n))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *z_gpu;
	DATA_TYPE *v1_gpu;
	DATA_TYPE *v2_gpu;
	DATA_TYPE *u1_gpu;
	DATA_TYPE *u2_gpu;
	DATA_TYPE *w_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N * N,
                                             dpct::get_default_queue());
        x_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                             dpct::get_default_queue());
        y_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                             dpct::get_default_queue());
        z_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                             dpct::get_default_queue());
        w_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                             dpct::get_default_queue());
        v1_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                              dpct::get_default_queue());
        v2_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                              dpct::get_default_queue());
        u1_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                              dpct::get_default_queue());
        u2_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * N,
                                              dpct::get_default_queue());

        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N);
        dpct::get_default_queue().memcpy(x_gpu, x, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(y_gpu, y, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(z_gpu, z, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(w_gpu, w, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(v1_gpu, v1, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(v2_gpu, v2, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(u1_gpu, u1, sizeof(DATA_TYPE) * N);
        dpct::get_default_queue().memcpy(u2_gpu, u2, sizeof(DATA_TYPE) * N).wait();

        sycl::range<3> block1(1, DIM_THREAD_BLOCK_KERNEL_1_Y,
                              DIM_THREAD_BLOCK_KERNEL_1_X);
        sycl::range<3> grid1(
            1, (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_Y)),
            (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)));

        sycl::range<3> block2(1, DIM_THREAD_BLOCK_KERNEL_2_Y,
                              DIM_THREAD_BLOCK_KERNEL_2_X);
        sycl::range<3> grid2(
            1, 1,
            (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));

        sycl::range<3> block3(1, DIM_THREAD_BLOCK_KERNEL_3_Y,
                              DIM_THREAD_BLOCK_KERNEL_3_X);
        sycl::range<3> grid3(
            1, 1,
            (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)));

        /* Start timer. */
  	polybench_start_instruments;

        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid1 * block1, block1),
            [=](sycl::nd_item<3> item_ct1) {
                    gemver_kernel1(n, alpha, beta, A_gpu, v1_gpu, v2_gpu,
                                   u1_gpu, u2_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid2 * block2, block2),
            [=](sycl::nd_item<3> item_ct1) {
                    gemver_kernel2(n, alpha, beta, A_gpu, x_gpu, y_gpu, z_gpu,
                                   item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid3 * block3, block3),
            [=](sycl::nd_item<3> item_ct1) {
                    gemver_kernel3(n, alpha, beta, A_gpu, x_gpu, w_gpu,
                                   item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(w_outputFromGpu, w_gpu, sizeof(DATA_TYPE) * N)
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(x_gpu, dpct::get_default_queue());
        sycl::free(y_gpu, dpct::get_default_queue());
        sycl::free(z_gpu, dpct::get_default_queue());
        sycl::free(w_gpu, dpct::get_default_queue());
        sycl::free(v1_gpu, dpct::get_default_queue());
        sycl::free(v2_gpu, dpct::get_default_queue());
        sycl::free(u1_gpu, dpct::get_default_queue());
        sycl::free(u2_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(w,N,n))
{
  int i;

  for (i = 0; i < n; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, w[i]);
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
	POLYBENCH_1D_ARRAY_DECL(u1,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(v1,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(u2,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(v2,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(w,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(w_outputFromGpu,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,N,n);
  	POLYBENCH_1D_ARRAY_DECL(z,DATA_TYPE,N,n);
  	
	
	init(n, &alpha, &beta,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(u1),
	      POLYBENCH_ARRAY(v1),
	      POLYBENCH_ARRAY(u2),
	      POLYBENCH_ARRAY(v2),
	      POLYBENCH_ARRAY(w),
	      POLYBENCH_ARRAY(x),
	      POLYBENCH_ARRAY(y),
	      POLYBENCH_ARRAY(z));
	
	GPU_argv_init();

	gemverCuda(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), 
		POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

	#ifdef RUN_ON_CPU

	 	/* Start timer. */
	  	polybench_start_instruments;
	
		gemver(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), 
			POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));


		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
 		polybench_print_instruments;
		
		compareResults(n, POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(w_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(w);  
	POLYBENCH_FREE_ARRAY(w_outputFromGpu);  
	POLYBENCH_FREE_ARRAY(x);  
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(z);
	POLYBENCH_FREE_ARRAY(u1);
	POLYBENCH_FREE_ARRAY(u2);
	POLYBENCH_FREE_ARRAY(v1);
	POLYBENCH_FREE_ARRAY(v2);

 	return 0;
}

#include <polybench.c>