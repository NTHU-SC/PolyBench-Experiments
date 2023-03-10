/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#include "3mm.dp.hpp"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

#define GPU_DEVICE 0

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05



void init_array(int ni, int nj, int nk, int nl, int nm, DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk), DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj), 
		DATA_TYPE POLYBENCH_2D(C, NJ, NM, nj, nm), DATA_TYPE POLYBENCH_2D(D, NM, NL, nm, nl))
{
	int i, j;

	for (i = 0; i < ni; i++)
	{
		for (j = 0; j < nk; j++)
		{
			A[i][j] = ((DATA_TYPE) i*j) / ni;
		}
	}
  
	for (i = 0; i < nk; i++)
	{
		for (j = 0; j < nj; j++)
		{
			B[i][j] = ((DATA_TYPE) i*(j+1)) / nj;
		}
	}
  
	for (i = 0; i < nj; i++)
	{
		for (j = 0; j < nm; j++)
		{
			C[i][j] = ((DATA_TYPE) i*(j+3)) / nl;
		}
	}
  
	for (i = 0; i < nm; i++)
	{
		for (j = 0; j < nl; j++)
		{
			D[i][j] = ((DATA_TYPE) i*(j+2)) / nk;
		}
	}
}


void compareResults(int ni, int nl, DATA_TYPE POLYBENCH_2D(G, NI, NL, ni, nl), DATA_TYPE POLYBENCH_2D(G_outputFromGpu, NI, NL, ni, nl))
{
	int i,j,fail;
	fail = 0;

	for (i=0; i < ni; i++)
	{
		for (j=0; j < nl; j++)
		{
			if (percentDiff(G[i][j], G_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
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
        DPCT1093:3: The "0" may not be the best XPU device. Adjust the selected
        device if needed.
        */
        dpct::select_device(GPU_DEVICE);
}

	
void mm3_kernel1(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E,
                 sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_NI) && (j < _PB_NJ))
	{
		E[i * NJ + j] = 0;
		int k;
		for(k=0; k < _PB_NK; k++)
		{
			E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}

	
void mm3_kernel2(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F,
                 sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_NJ) && (j < _PB_NL))
	{
		F[i * NL + j] = 0;
		int k;
		for(k=0; k < _PB_NM; k++)
		{
			F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
		}
	}
}

	
void mm3_kernel3(int ni, int nj, int nk, int nl, int nm, DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G,
                 sycl::nd_item<3> item_ct1)
{
        int j = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);
        int i = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                item_ct1.get_local_id(1);

        if ((i < _PB_NI) && (j < _PB_NL))
	{
		G[i * NL + j] = 0;
		int k;
		for(k=0; k < _PB_NJ; k++)
		{
			G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
		}
	}
}


/* Main computational kernel on CPU */
void mm3_cpu(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
	int i, j, k;

	/* E := A*B */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NJ; j++)
		{
			E[i][j] = 0;
			for (k = 0; k < _PB_NK; ++k)
			{
				E[i][j] += A[i][k] * B[k][j];
			}
		}
	}

	/* F := C*D */
	for (i = 0; i < _PB_NJ; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			F[i][j] = 0;
			for (k = 0; k < _PB_NM; ++k)
			{
				F[i][j] += C[i][k] * D[k][j];
			}
		}
	}

	/* G := E*F */
	for (i = 0; i < _PB_NI; i++)
	{
		for (j = 0; j < _PB_NL; j++)
		{
			G[i][j] = 0;
			for (k = 0; k < _PB_NJ; ++k)
			{
				G[i][j] += E[i][k] * F[k][j];
			}
		}
	}
}


void mm3Cuda(int ni, int nj, int nk, int nl, int nm,
		DATA_TYPE POLYBENCH_2D(E,NI,NJ,ni,nj),
		DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(F,NJ,NL,nj,nl),
		DATA_TYPE POLYBENCH_2D(C,NJ,NM,nj,nm),
		DATA_TYPE POLYBENCH_2D(D,NM,NL,nm,nl),
		DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl),
		DATA_TYPE POLYBENCH_2D(G_outputFromGpu,NI,NL,ni,nl))
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
        DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;

        A_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NK,
                                             dpct::get_default_queue());
        B_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NK * NJ,
                                             dpct::get_default_queue());
        C_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NJ * NM,
                                             dpct::get_default_queue());
        D_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NM * NL,
                                             dpct::get_default_queue());
        E_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NJ,
                                             dpct::get_default_queue());
        F_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NJ * NL,
                                             dpct::get_default_queue());
        G_gpu = (float *)sycl::malloc_device(sizeof(DATA_TYPE) * NI * NL,
                                             dpct::get_default_queue());

        dpct::get_default_queue().memcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK);
        dpct::get_default_queue().memcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ);
        dpct::get_default_queue().memcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM);
        dpct::get_default_queue().memcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL);
        dpct::get_default_queue().memcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ);
        dpct::get_default_queue().memcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL);
        dpct::get_default_queue().memcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL).wait();

        sycl::range<3> block(1, DIM_THREAD_BLOCK_Y, DIM_THREAD_BLOCK_X);
        sycl::range<3> grid1(
            1, (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))),
            (size_t)(ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X))));
        sycl::range<3> grid2(
            1, (size_t)(ceil((float)NJ / ((float)DIM_THREAD_BLOCK_Y))),
            (size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))));
        sycl::range<3> grid3(
            1, (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))),
            (size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))));

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
                    mm3_kernel1(ni, nj, nk, nl, nm, A_gpu, B_gpu, E_gpu,
                                item_ct1);
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
                    mm3_kernel2(ni, nj, nk, nl, nm, C_gpu, D_gpu, F_gpu,
                                item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();
        /*
        DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        dpct::get_default_queue().parallel_for(
            sycl::nd_range<3>(grid3 * block, block),
            [=](sycl::nd_item<3> item_ct1) {
                    mm3_kernel3(ni, nj, nk, nl, nm, E_gpu, F_gpu, G_gpu,
                                item_ct1);
            });
        dpct::get_current_device().queues_wait_and_throw();

        /* Stop and print timer. */
	printf("GPU Time in seconds:\n");
  	polybench_stop_instruments;
 	polybench_print_instruments;
        dpct::get_default_queue()
            .memcpy(G_outputFromGpu, G_gpu, sizeof(DATA_TYPE) * NI * NL)
            .wait();

        sycl::free(A_gpu, dpct::get_default_queue());
        sycl::free(B_gpu, dpct::get_default_queue());
        sycl::free(C_gpu, dpct::get_default_queue());
        sycl::free(D_gpu, dpct::get_default_queue());
        sycl::free(E_gpu, dpct::get_default_queue());
        sycl::free(F_gpu, dpct::get_default_queue());
        sycl::free(G_gpu, dpct::get_default_queue());
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nl,
		 DATA_TYPE POLYBENCH_2D(G,NI,NL,ni,nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, G[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char** argv)
{
	int ni = NI;
	int nj = NJ;
	int nk = NK;
	int nl = NL;
	int nm = NM;

	/* Variable declaration/allocation. */
	POLYBENCH_2D_ARRAY_DECL(E, DATA_TYPE, NI, NJ, ni, nj);
	POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
	POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
	POLYBENCH_2D_ARRAY_DECL(F, DATA_TYPE, NJ, NL, nj, nl);
	POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NJ, NM, nj, nm);
	POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NM, NL, nm, nl);
	POLYBENCH_2D_ARRAY_DECL(G, DATA_TYPE, NI, NL, ni, nl);
	POLYBENCH_2D_ARRAY_DECL(G_outputFromGpu, DATA_TYPE, NI, NL, ni, nl);

	init_array(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D));

	GPU_argv_init();

	mm3Cuda(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E), 
		POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		mm3_cpu(ni, nj, nk, nl, nm, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(D), POLYBENCH_ARRAY(E), 
			POLYBENCH_ARRAY(F), POLYBENCH_ARRAY(G));
	
		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;

		compareResults(ni, nl, POLYBENCH_ARRAY(G), POLYBENCH_ARRAY(G_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(G_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(C);
	POLYBENCH_FREE_ARRAY(D);
	POLYBENCH_FREE_ARRAY(E);
	POLYBENCH_FREE_ARRAY(F);
	POLYBENCH_FREE_ARRAY(G);
	POLYBENCH_FREE_ARRAY(G_outputFromGpu);

	return 0;
}

#include <polybench.c>