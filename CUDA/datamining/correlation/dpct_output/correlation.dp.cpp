/**
 * correlation.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "correlation.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 1.05

#define GPU_DEVICE 0

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01f
#define EPS 0.005f



void init_arrays(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n))
{
	int i, j;
	
	for (i=0; i < m; i++) 
	{
    		for (j=0; j < n; j++) 
		{
       		data[i][j] = ((DATA_TYPE) i*j)/ M;	
       	}
    	}
}


void correlation(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), DATA_TYPE POLYBENCH_1D(stddev, M, m),
		DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n))
{
	int i, j, j1, j2;	
	
	// Determine mean of column vectors of input data matrix 
  	for (j = 0; j < _PB_M; j++)
   	{
  		mean[j] = 0.0;

   		for (i = 0; i < _PB_N; i++)
		{
			mean[j] += data[i][j];
   		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
   	}

	// Determine standard deviations of column vectors of data matrix. 
  	for (j = 0; j < _PB_M; j++)
   	{
   		stddev[j] = 0.0;
      
		for (i = 0; i < _PB_N; i++)
		{
			stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
		}
		
		stddev[j] /= FLOAT_N;
		stddev[j] = sqrt_of_array_cell(stddev, j);
		stddev[j] = stddev[j] <= EPS ? 1.0 : stddev[j];
	}

 	// Center and reduce the column vectors. 
  	for (i = 0; i < _PB_N; i++)
	{
		for (j = 0; j < _PB_M; j++)
		{
			data[i][j] -= mean[j];
			data[i][j] /= (sqrt(FLOAT_N)*stddev[j]) ;
		}
	}

	// Calculate the m * m correlation matrix. 
  	for (j1 = 0; j1 < _PB_M-1; j1++)
	{	
		symmat[j1][j1] = 1.0;
    
		for (j2 = j1+1; j2 < _PB_M; j2++)
		{
	  		symmat[j1][j2] = 0.0;

	  		for (i = 0; i < _PB_N; i++)
			{
	   			symmat[j1][j2] += (data[i][j1] * data[i][j2]);
			}

	  		symmat[j2][j1] = symmat[j1][j2];
		}
	}
 
	symmat[M-1][M-1] = 1.0;
}


void compareResults(int m, int n, DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < m; i++)
	{
		for (j=0; j < n; j++)
		{
			if (percentDiff(symmat[i][j], symmat_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
        DPCT1093:4: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}

	
void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data,
                 sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (j < _PB_M)
	{
		mean[j] = 0.0;

		int i;
		for(i=0; i < _PB_N; i++)
		{
			mean[j] += data[i*M + j];
		}
		
		mean[j] /= (DATA_TYPE)FLOAT_N;
	}
}


void std_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data,
                sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

        if (j < _PB_M)
	{
		std[j] = 0.0;

		int i;
		for(i = 0; i < _PB_N; i++)
		{
			std[j] += (data[i*M + j] - mean[j]) * (data[i*M + j] - mean[j]);
		}
		std[j] /= (FLOAT_N);
                std[j] = sycl::sqrt(std[j]);
                if(std[j] <= EPS) 
		{
			std[j] = 1.0;
		}
	}
}


void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data,
                   sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_N) && (j < _PB_M))
	{
		data[i*M + j] -= mean[j];
                data[i * M + j] /= (sycl::sqrt(FLOAT_N) * std[j]);
        }
}


void corr_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data,
                 sycl::nd_item<3> item_ct1)
{
        int j1 = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                 item_ct1.get_local_id(2);

        int i, j2;
	if (j1 < (_PB_M-1))
	{
		symmat[j1*M + j1] = 1.0;

		for (j2 = (j1 + 1); j2 < _PB_M; j2++)
		{
			symmat[j1*M + j2] = 0.0;

			for(i = 0; i < _PB_N; i++)
			{
				symmat[j1*M + j2] += data[i*M + j1] * data[i*M + j2];
			}
			symmat[j2*M + j1] = symmat[j1*M + j2];
		}
	}
}


void correlationCuda(int m, int n, DATA_TYPE POLYBENCH_2D(data, M, N, m, n), DATA_TYPE POLYBENCH_1D(mean, M, m), 
			DATA_TYPE POLYBENCH_1D(stddev, M, m), DATA_TYPE POLYBENCH_2D(symmat, M, N, m, n), 
			DATA_TYPE POLYBENCH_2D(symmat_outputFromGpu, M, N, m, n))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE *data_gpu;
	DATA_TYPE *stddev_gpu;
	DATA_TYPE *mean_gpu;
	DATA_TYPE *symmat_gpu;

        data_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * M * N,
                                                dpct::get_default_queue());
        symmat_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * M * N,
                                                  dpct::get_default_queue());
        stddev_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * M,
                                                  dpct::get_default_queue());
        mean_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * M,
                                                dpct::get_default_queue());
        dpct::get_default_queue().memcpy(data_gpu, data, sizeof(DATA_TYPE) * M * N);
        dpct::get_default_queue().memcpy(symmat_gpu, symmat,
                                         sizeof(DATA_TYPE) * M * N);
        dpct::get_default_queue().memcpy(stddev_gpu, stddev, sizeof(DATA_TYPE) * M);
        dpct::get_default_queue().memcpy(mean_gpu, mean, sizeof(DATA_TYPE) * M).wait();

        sycl::range<3> block1(1, DIM_THREAD_BLOCK_KERNEL_1_Y,
                              DIM_THREAD_BLOCK_KERNEL_1_X);
        sycl::range<3> grid1(
            1, 1,
            (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)));

        sycl::range<3> block2(1, DIM_THREAD_BLOCK_KERNEL_2_Y,
                              DIM_THREAD_BLOCK_KERNEL_2_X);
        sycl::range<3> grid2(
            1, 1,
            (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));

        sycl::range<3> block3(1, DIM_THREAD_BLOCK_KERNEL_3_Y,
                              DIM_THREAD_BLOCK_KERNEL_3_X);
        sycl::range<3> grid3(
            1,
            (size_t)(ceil((float)(N)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_Y)),
            (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)));

        sycl::range<3> block4(1, DIM_THREAD_BLOCK_KERNEL_4_Y,
                              DIM_THREAD_BLOCK_KERNEL_4_X);
        sycl::range<3> grid4(
            1, 1,
            (size_t)(ceil((float)(M)) / ((float)DIM_THREAD_BLOCK_KERNEL_4_X)));

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
                    mean_kernel(m, n, mean_gpu, data_gpu, item_ct1);
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
                    std_kernel(m, n, mean_gpu, stddev_gpu, data_gpu, item_ct1);
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
                    reduce_kernel(m, n, mean_gpu, stddev_gpu, data_gpu,
                                  item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid4 * block4, block4),
            [=](sycl::nd_item<3> item_ct1) {
                    corr_kernel(m, n, symmat_gpu, data_gpu, item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;

	DATA_TYPE valueAtSymmatIndexMTimesMPlus1PlusMPoint = 1.0;
        dpct::get_default_queue().memcpy(
            &(symmat_gpu[(M - 1) * M + (M - 1)]),
            &valueAtSymmatIndexMTimesMPlus1PlusMPoint, sizeof(DATA_TYPE));

        dpct::get_default_queue()
            .memcpy(symmat_outputFromGpu, symmat_gpu, sizeof(DATA_TYPE) * M * N)
            .wait();

        sycl::free(data_gpu, dpct::get_default_queue());
        sycl::free(symmat_gpu, dpct::get_default_queue());
        sycl::free(stddev_gpu, dpct::get_default_queue());
        sycl::free(mean_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m,
		 DATA_TYPE POLYBENCH_2D(symmat,M,M,m,m))

{
  int i, j;

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
      if ((i * m + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int m = M;
	int n = N;

	POLYBENCH_2D_ARRAY_DECL(data,DATA_TYPE,M,N,m,n);
  	POLYBENCH_1D_ARRAY_DECL(mean,DATA_TYPE,M,m);
  	POLYBENCH_1D_ARRAY_DECL(stddev,DATA_TYPE,M,m);
	POLYBENCH_2D_ARRAY_DECL(symmat,DATA_TYPE,M,N,m,n);
  	POLYBENCH_2D_ARRAY_DECL(symmat_outputFromGpu,DATA_TYPE,M,N,m,n);
  	
	init_arrays(m, n, POLYBENCH_ARRAY(data));
    
	GPU_argv_init();

	correlationCuda(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat), 
		POLYBENCH_ARRAY(symmat_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		correlation(m, n, POLYBENCH_ARRAY(data), POLYBENCH_ARRAY(mean), POLYBENCH_ARRAY(stddev), POLYBENCH_ARRAY(symmat));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
    
		compareResults(m, n, POLYBENCH_ARRAY(symmat), POLYBENCH_ARRAY(symmat_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(m, POLYBENCH_ARRAY(symmat_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(data);
	POLYBENCH_FREE_ARRAY(mean);
	POLYBENCH_FREE_ARRAY(stddev);
	POLYBENCH_FREE_ARRAY(symmat);
	POLYBENCH_FREE_ARRAY(symmat_outputFromGpu);

  	return 0;
}

#include <polybench.c>