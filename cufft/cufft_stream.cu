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
	int test_stream = 8;
	int test_number = 10000;
	for (int iter = 13; iter <= 13; ++iter)
	{
		for (int i_stream = 8; i_stream <= test_stream; ++i_stream)
		{
			int N = iter;
			int SIZE = N*N;
			
			cudaStream_t streams[8];
			for (int i = 0; i < i_stream; ++i)
				cudaStreamCreate(&streams[i]);

			Complex *fg[8];
			for (int i = 0; i < i_stream; ++i)
				fg[i] = new Complex[SIZE];
			for (int j = 0; j < i_stream; ++j)
				for (int i = 0; i < SIZE; i++){
					fg[j][i].x = 1; 
					fg[j][i].y = 0;
				}
    
			int mem_size = sizeof(Complex)* SIZE;

			Complex *d_signal[8];
			for (int i = 0; i < i_stream; ++i) {
				//cout << i << endl;
				checkCudaErrors(cudaMalloc((void **) &(d_signal[i]), mem_size)); 
				checkCudaErrors(cudaMemcpyAsync(d_signal[i], fg[i], mem_size, 
					cudaMemcpyHostToDevice, streams[i]));
			}
	
			cudaDeviceSynchronize();
			for (int i = 0; i < i_stream; ++i) 
				cudaStreamSynchronize(streams[i]);
	
			// CUFFT plan
			cufftHandle plan[8];	
			for (int i = 0; i < i_stream; ++i) {
				cufftPlan2d(&plan[i], N, N, CUFFT_C2C);
				cufftSetStream(plan[i], streams[i]);
			}
	
			// Transform signal and filter
			clock_t start, end;
			start = clock();
			for (int j = 0; j < test_number / i_stream; ++j) {
				for (int i = 0; i < i_stream; ++i) {
					cufftExecC2C(plan[i], (cufftComplex *)d_signal, (cufftComplex *)d_signal, 
						CUFFT_FORWARD);
				}
				cudaDeviceSynchronize();
			}
			end = clock();
			double fft_time = (double)(end - start) / CLOCKS_PER_SEC;
			printf("forwardsize%d:streamsize%d:%.6lf\n", N, i_stream, fft_time);

			start = clock();
			for (int j = 0; j < test_number / i_stream; ++j) {
				for (int i = 0; i < i_stream; ++i) {
					cufftExecC2C(plan[i], (cufftComplex *)d_signal, (cufftComplex *)d_signal, 
						CUFFT_INVERSE);
					cudaDeviceSynchronize();
				}
			}
			end = clock();
			fft_time = (double)(end - start) / CLOCKS_PER_SEC;
			printf("backwardsize%d:streamsize%d:%.6lf\n", N, i_stream, fft_time);

			Complex * result[8];
			for (int i = 0; i < i_stream; i++)
				result[i] = new Complex[SIZE];
			for (int i = 0; i < i_stream; i++)
				cudaMemcpyAsync(result[i], d_signal[i], sizeof(Complex)*SIZE, 
					cudaMemcpyDeviceToHost, streams[i]);

			for (int i = 0; i < i_stream; i++) {
				delete result[i];
				delete fg[i];
			}
			for (int i = 0; i < i_stream; i++)
				cufftDestroy(plan[i]);
			//cufftDestroy(plan2);
			for (int i = 0; i < i_stream; i++)
				cudaFree(d_signal[i]);
			for (int i = 0; i < i_stream; i++)
				d_signal[i] = NULL;
			for (int i = 0; i < i_stream; i++)
				cudaStreamDestroy(streams[i]);
			cudaDeviceReset();
		}
	}		
}
