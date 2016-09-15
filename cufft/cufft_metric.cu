#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//#include <helper_functions.h>
#include <helper_cuda.h>

#include <ctime>
#include <time.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cufft.h>
#include <fstream>

using namespace std;
typedef float2 Complex;

int main()
{
	int test_batch[] = {8, 16, 32, 64, 128, 192, 256};
	//int test_batch[] = {320, 384, 448};
	int test_number = 1000000;
	for (int iter = 13; iter <= 13; ++iter)
	{
		for (int ibatch = 0; ibatch <= 6; ++ibatch)
		{
			int batch = test_batch[ibatch];
			int N = iter;
			int SIZE = N*N;
			
			Complex *fg;
			fg = new Complex[SIZE*batch];
			for (int j = 0; j < batch; ++j)
				for (int i = 0; i < SIZE; i++){
					fg[i + j * SIZE].x = 1; 
					fg[i + j * SIZE].y = 0;
				} 
    
			int mem_size = sizeof(Complex)* SIZE * batch;

			Complex *d_signal;
			checkCudaErrors(cudaMalloc((void **) &(d_signal), mem_size)); 
			checkCudaErrors(cudaMemcpyAsync(d_signal, fg, mem_size, 
				cudaMemcpyHostToDevice));
	
			cudaDeviceSynchronize();
	
			// CUFFT plan
			cufftHandle plan;
			int s[2] = {N, N};
			int inembed[2] = {N, batch};
			cufftPlanMany(&plan,2, s, inembed, batch, 1, inembed, batch, 1, CUFFT_C2C, batch);
	
			// Transform signal and filter
			clock_t start, end;
			start = clock();
			for (int j = 0; j < test_number / batch; ++j) {
				cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, 
					CUFFT_FORWARD);
				cudaDeviceSynchronize();
			}
			end = clock();
			double fft_time = (double)(end - start) / CLOCKS_PER_SEC;
			printf("forwardsize%d:batchsize%d:%.6lf\n", N, batch, fft_time);

			start = clock();
			for (int j = 0; j < test_number / batch; ++j) {
				cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, 
					CUFFT_INVERSE);
				cudaDeviceSynchronize();
			}
			end = clock();
			fft_time = (double)(end - start) / CLOCKS_PER_SEC;
			printf("backwardsize%d:batchsize%d:%.6lf\n", N, batch, fft_time);

			Complex * result;
			result = new Complex[SIZE*batch];
			cudaMemcpyAsync(result, d_signal, sizeof(Complex)*SIZE, 
				cudaMemcpyDeviceToHost);

			delete result;
			delete fg;
			cufftDestroy(plan);
			//cufftDestroy(plan2);
			cudaFree(d_signal);
		}
	}		
}
