/**
 * doitgen.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "doitgen.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0




/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_doitgenCpu(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np))
{
	int r, q, p, s;

	for (r = 0; r < _PB_NR; r++)
	{
		for (q = 0; q < _PB_NQ; q++)  
		{
			for (p = 0; p < _PB_NP; p++)  
			{
				sum[r][q][p] = 0;
				for (s = 0; s < _PB_NP; s++)
					sum[r][q][p] = sum[r][q][p] + A[r][q][s] * C4[s][p];
			}
			for (p = 0; p < _PB_NR; p++)
				A[r][q][p] = sum[r][q][p];
		}
	}

}



/* Array initialization. */
void init_array(int nr, int nq, int np,
		DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np))
{
	int i, j, k;

	for (i = 0; i < nr; i++)
		for (j = 0; j < nq; j++)
			for (k = 0; k < np; k++)
				A[i][j][k] = ((DATA_TYPE) i*j + k) / np;

	for (i = 0; i < np; i++)
		for (j = 0; j < np; j++)
			C4[i][j] = ((DATA_TYPE) i*j) / np;
}


void compareResults(int nr, int nq, int np, DATA_TYPE POLYBENCH_3D(sum,NR,NQ,NP,nr,nq,np), 
			DATA_TYPE POLYBENCH_3D(sum_outputFromGpu,NR,NQ,NP,nr,nq,np))
{
	int fail = 0;
	
	for (int r = 0; r < nr; r++)
	{
    		for (int q = 0; q < nq; q++)  
		{
      			for (int p = 0; p < np; p++)  
			{
				if (percentDiff(sum[r][q][p], sum_outputFromGpu[r][q][p]) > PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}
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
        DPCT1093:2: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}


void doitgen_kernel1(int nr, int nq, int np, DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r,
                     sycl::nd_item<3> item_ct1)
{
        int p = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int q = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((p < np) && (q < nq))
	{
		sum[r * (nq * np) + q * np + p] = (DATA_TYPE)0.0;
	
		for (int s = 0; s < np; s++)
		{
			sum[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p] + A[r * (nq * np) + q * np + s] * C4[s * np + p];
		}
	}
}

void doitgen_kernel2(int nr, int nq, int np, DATA_TYPE *sum, DATA_TYPE *A, DATA_TYPE *C4, int r,
                     sycl::nd_item<3> item_ct1)
{
        int p = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int q = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((p < np) && (q < nq))
	{
		A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
	}
}

void doitgenCuda(int nr, int nq, int np,
		    DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np),
		    DATA_TYPE POLYBENCH_2D(C4,NP,NP,np,np),
		    DATA_TYPE POLYBENCH_3D(sum_outputFromGpu,NR,NQ,NP,nr,nq,np))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE* AGpu;
	DATA_TYPE* C4Gpu;
	DATA_TYPE* sumGpu;

        AGpu = (float *)sycl::malloc_device(nr * nq * np * sizeof(DATA_TYPE),
                                            dpct::get_default_queue());
        C4Gpu = (float *)sycl::malloc_device(np * np * sizeof(DATA_TYPE),
                                             dpct::get_default_queue());
        sumGpu = (float *)sycl::malloc_device(nr * nq * np * sizeof(DATA_TYPE),
                                              dpct::get_default_queue());

        dpct::get_default_queue().memcpy(AGpu, A, nr * nq * np * sizeof(DATA_TYPE));
        dpct::get_default_queue().memcpy(C4Gpu, C4, np * np * sizeof(DATA_TYPE));
        dpct::get_default_queue()
            .memcpy(sumGpu, sum_outputFromGpu, nr * nq * np * sizeof(DATA_TYPE))
            .wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid(
            1, (unsigned int)ceil(((float)nr) / ((float)block[1])),
            (unsigned int)ceil(((float)np) / ((float)block[2])));

        /* Start timer. */
	polybench_start_instruments;	

	for (int r = 0; r < nr; r++)
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
                            doitgen_kernel1(nr, nq, np, sumGpu, AGpu, C4Gpu, r,
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
                    sycl::nd_range<3>(grid * block, block),
                    [=](sycl::nd_item<3> item_ct1) {
                            doitgen_kernel2(nr, nq, np, sumGpu, AGpu, C4Gpu, r,
                                            item_ct1);
                    });
                dpct::get_current_device().queues_wait_and_throw();
        }

	/* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
	polybench_print_instruments;

        dpct::get_default_queue()
            .memcpy(sum_outputFromGpu, sumGpu, NR * NQ * NP * sizeof(DATA_TYPE))
            .wait();

        sycl::free(AGpu, dpct::get_default_queue());
        sycl::free(C4Gpu, dpct::get_default_queue());
        sycl::free(sumGpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nr, int nq, int np,
		 DATA_TYPE POLYBENCH_3D(A,NR,NQ,NP,nr,nq,np))
{
	int i, j, k;

	for (i = 0; i < nr; i++)
	{
		for (j = 0; j < nq; j++)
		{
			for (k = 0; k < np; k++) 
			{
				fprintf (stderr, DATA_PRINTF_MODIFIER, A[i][j][k]);
				if (i % 20 == 0) fprintf (stderr, "\n");
			}
		}
	}
	fprintf (stderr, "\n");
}
	

int main(int argc, char *argv[])
{
	/* Retrieve problem size. */
	int nr = NR;
	int nq = NQ;
	int np = NP;

	/* Variable declaration/allocation. */
	POLYBENCH_3D_ARRAY_DECL(A,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_3D_ARRAY_DECL(sum,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_3D_ARRAY_DECL(sum_outputFromGpu,DATA_TYPE,NR,NQ,NP,nr,nq,np);
	POLYBENCH_2D_ARRAY_DECL(C4,DATA_TYPE,NP,NP,np,np);

	/* Initialize array(s). */
	init_array (nr, nq, np,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(C4));

	doitgenCuda(nr, nq, np,
		POLYBENCH_ARRAY(A),
		POLYBENCH_ARRAY(C4),
		POLYBENCH_ARRAY(sum_outputFromGpu));


	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		/* Run kernel on CPU */
		kernel_doitgenCpu(nr, nq, np,
		  POLYBENCH_ARRAY(A),
		  POLYBENCH_ARRAY(C4),
		  POLYBENCH_ARRAY(sum));	
		
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
  		polybench_stop_instruments;
	 	polybench_print_instruments;
	
		compareResults(nr, nq, np, POLYBENCH_ARRAY(sum), POLYBENCH_ARRAY(sum_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nr, nq, np, POLYBENCH_ARRAY(sum_outputFromGpu)));

	#endif //RUN_ON_CPU

	/* Garbage collection */
	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(sum);
	POLYBENCH_FREE_ARRAY(sum_outputFromGpu);
	POLYBENCH_FREE_ARRAY(C4);	
    
	return 0;
}

#include <polybench.c>