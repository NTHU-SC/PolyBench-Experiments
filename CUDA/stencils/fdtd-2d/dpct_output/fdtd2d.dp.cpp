/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "fdtd2d.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0



void init_arrays(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
		DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			ex[i][j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i][j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i][j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void runFdtd(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
	DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	int t, i, j;
	
	for (t=0; t < _PB_TMAX; t++)  
	{
		for (j=0; j < _PB_NY; j++)
		{
			ey[0][j] = _fict_[t];
		}
	
		for (i = 1; i < _PB_NX; i++)
		{
       		for (j = 0; j < _PB_NY; j++)
			{
       			ey[i][j] = ey[i][j] - 0.5*(hz[i][j] - hz[(i-1)][j]);
        		}
		}

		for (i = 0; i < _PB_NX; i++)
		{
       		for (j = 1; j < _PB_NY; j++)
			{
				ex[i][j] = ex[i][j] - 0.5*(hz[i][j] - hz[i][(j-1)]);
			}
		}

		for (i = 0; i < _PB_NX-1; i++)
		{
			for (j = 0; j < _PB_NY-1; j++)
			{
				hz[i][j] = hz[i][j] - 0.7*(ex[i][(j+1)] - ex[i][j] + ey[(i+1)][j] - ey[i][j]);
			}
		}
	}
}


void compareResults(int nx, int ny, DATA_TYPE POLYBENCH_2D(hz1,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz2,NX,NY,nx,ny))
{
	int i, j, fail;
	fail = 0;
	
	for (i=0; i < nx; i++) 
	{
		for (j=0; j < ny; j++) 
		{
			if (percentDiff(hz1[i][j], hz2[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
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
}



void fdtd_step1_kernel(int nx, int ny, DATA_TYPE* _fict_, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t,
                       sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_NX) && (j < _PB_NY))
	{
		if (i == 0) 
		{
			ey[i * NY + j] = _fict_[t];
		}
		else
		{ 
			ey[i * NY + j] = ey[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[(i-1) * NY + j]);
		}
	}
}



void fdtd_step2_kernel(int nx, int ny, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t,
                       sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_NX) && (j < _PB_NY) && (j > 0))
	{
		ex[i * NY + j] = ex[i * NY + j] - 0.5f*(hz[i * NY + j] - hz[i * NY + (j-1)]);
	}
}


void fdtd_step3_kernel(int nx, int ny, DATA_TYPE *ex, DATA_TYPE *ey, DATA_TYPE *hz, int t,
                       sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < (_PB_NX-1)) && (j < (_PB_NY-1)))
	{	
		hz[i * NY + j] = hz[i * NY + j] - 0.7f*(ex[i * NY + (j+1)] - ex[i * NY + j] + ey[(i + 1) * NY + j] - ey[i * NY + j]);
	}
}


void fdtdCuda(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
	DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz_outputFromGpu,NX,NY,nx,ny))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE *_fict_gpu;
	DATA_TYPE *ex_gpu;
	DATA_TYPE *ey_gpu;
	DATA_TYPE *hz_gpu;

        _fict_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * TMAX,
                                                 dpct::get_default_queue());
        ex_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY,
                                              dpct::get_default_queue());
        ey_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY,
                                              dpct::get_default_queue());
        hz_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NX * NY,
                                              dpct::get_default_queue());

        dpct::get_default_queue().memcpy(_fict_gpu, _fict_, sizeof(DATA_TYPE) * TMAX);
        dpct::get_default_queue().memcpy(ex_gpu, ex, sizeof(DATA_TYPE) * NX * NY);
        dpct::get_default_queue().memcpy(ey_gpu, ey, sizeof(DATA_TYPE) * NX * NY);
        dpct::get_default_queue()
            .memcpy(hz_gpu, hz, sizeof(DATA_TYPE) * NX * NY)
            .wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid(1, (size_t)ceil(((float)NX) / ((float)block[1])),
                            (size_t)ceil(((float)NY) / ((float)block[2])));

        /* Start timer. */
  	polybench_start_instruments;

	for(int t = 0; t < _PB_TMAX; t++)
	{
                /*
                DPCT1049:0: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1) {
                            fdtd_step1_kernel(nx, ny, _fict_gpu, ex_gpu, ey_gpu,
                                              hz_gpu, t, item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
                /*
                DPCT1049:1: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::get_default_queue().parallel_for(
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1) {
                            fdtd_step2_kernel(nx, ny, ex_gpu, ey_gpu, hz_gpu, t,
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
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1) {
                            fdtd_step3_kernel(nx, ny, ex_gpu, ey_gpu, hz_gpu, t,
                                              item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
        }
	
	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(hz_outputFromGpu, hz_gpu, sizeof(DATA_TYPE) * NX * NY)
            .wait();

        sycl::free(_fict_gpu, dpct::get_default_queue());
        sycl::free(ex_gpu, dpct::get_default_queue());
        sycl::free(ey_gpu, dpct::get_default_queue());
        sycl::free(hz_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
         fprintf(stderr, DATA_PRINTF_MODIFIER, hz[i][j]);
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char *argv[])
{
	int tmax = TMAX;
	int nx = NX;
	int ny = NY;

	POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,TMAX);
	POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz_outputFromGpu,DATA_TYPE,NX,NY,nx,ny);

	init_arrays(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

	GPU_argv_init();
	fdtdCuda(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		runFdtd(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
		
		compareResults(nx, ny, POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(hz_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(_fict_);
	POLYBENCH_FREE_ARRAY(ex);
	POLYBENCH_FREE_ARRAY(ey);
	POLYBENCH_FREE_ARRAY(hz);
	POLYBENCH_FREE_ARRAY(hz_outputFromGpu);

	return 0;
}

#include <polybench.c>