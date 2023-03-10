/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "atax.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0


#ifndef M_PI
#define M_PI 3.14159
#endif



void init_array(int nx, int ny, DATA_TYPE POLYBENCH_1D(x,NX,nx), DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny))
{
	int i, j;

	for (i = 0; i < nx; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < ny; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
	}
}


void compareResults(int ny, DATA_TYPE POLYBENCH_1D(z,NY,ny), DATA_TYPE POLYBENCH_1D(z_outputFromGpu,NY,ny))
{
	int i, fail;
	fail = 0;

	for (i=0; i<ny; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
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


void atax_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp,
                  sycl::nd_item<3> item_ct1)
{
        int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (i < _PB_NX)
	{
		tmp[i] = 0;
		int j;
		for(j=0; j < _PB_NY; j++)
		{
			tmp[i] += A[i*NY+j] * x[j];
		}
	}
}

void atax_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp,
                  sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (j < _PB_NY)
	{
		y[j] = 0;
		int i;
		for(i=0; i < _PB_NX; i++)
		{
			y[j] += A[i*NY+j] * tmp[i];
		}
	}
}


void atax_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NY,ny), DATA_TYPE POLYBENCH_1D(y,NY,ny), 
		DATA_TYPE POLYBENCH_1D(tmp,NX,nx))
{
	int i,j;
	
	for (i= 0; i < _PB_NY; i++)
	{
    		y[i] = 0;
	}
  
	for (i = 0; i < _PB_NX; i++)
 	{
      		tmp[i] = 0;

      		for (j = 0; j < _PB_NY; j++)
		{
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
		
      		for (j = 0; j < _PB_NY; j++)
		{
			y[j] = y[j] + A[i][j] * tmp[i];
		}
    }
}


void ataxGpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A, NX, NY,nx,ny), DATA_TYPE POLYBENCH_1D(x,NX,nx), DATA_TYPE POLYBENCH_1D(y,NY,ny), 
		DATA_TYPE POLYBENCH_1D(tmp,NX,nx), DATA_TYPE POLYBENCH_1D(y_outputFromGpu,NY,ny))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY,
                                             dpct::get_default_queue());
        x_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NY,
                                             dpct::get_default_queue());
        y_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NY,
                                             dpct::get_default_queue());
        tmp_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX,
                                               dpct::get_default_queue());

        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY);
        dpct::get_default_queue().memcpy(x_gpu, x, sizeof(DATA_TYPE) * NY);
        dpct::get_default_queue().memcpy(y_gpu, y, sizeof(DATA_TYPE) * NY);
        dpct::get_default_queue().memcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid1(1, 1, (size_t)(ceil(((float)NX) / ((float)block[2]))));
        sycl::range<3> grid2(1, 1, (size_t)(ceil(((float)NY) / ((float)block[2]))));

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
                    atax_kernel1(nx, ny, A_gpu, x_gpu, tmp_gpu, item_ct1);
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
                    atax_kernel2(nx, ny, A_gpu, y_gpu, tmp_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * NX)
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(x_gpu, dpct::get_default_queue());
        sycl::free(y_gpu, dpct::get_default_queue());
        sycl::free(tmp_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx, DATA_TYPE POLYBENCH_1D(y,NX,nx))
{
  int i;

  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, y[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int nx = NX;
	int ny = NY;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(x,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(y_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(tmp,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(A));

	GPU_argv_init();
	ataxGpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp), 
		POLYBENCH_ARRAY(y_outputFromGpu));
	
	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		atax_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(tmp));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ny, POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(y_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ny, POLYBENCH_ARRAY(y_outputFromGpu)));

	#endif //RUN_ON_CPU

	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(x);
	POLYBENCH_FREE_ARRAY(y);
	POLYBENCH_FREE_ARRAY(y_outputFromGpu);
	POLYBENCH_FREE_ARRAY(tmp);

  	return 0;
}

#include <polybench.c>