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
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define DEBUG 0
using namespace std;
typedef float2 Complex;

cufftHandle makeFftPlan(int batch, float * real, float2 * cplx, bool forwardFFT, int size, int dim)
{
	int istride = 1;//batch;
	int ostride = 1;//batch;
	int idist = size * size;//1;
	int odist = size * (size / 2 + 1);//1;
	cufftHandle plan;
	cufftResult errFFT;
	int inSizeArr[2] = {size, size};
	int inembed[2] = {size, size};
	int onembed[2] = {size, size / 2 + 1};
	if (forwardFFT) {
		errFFT = cufftPlanMany(&plan, 
					2,
					inSizeArr,
					inembed,
//					NULL,
					istride,
					idist,
					onembed,
//					NULL,
					ostride,
					odist,
					CUFFT_R2C,
					batch );
	} else {
		errFFT = cufftPlanMany(&plan, 
					2,
					inSizeArr,
					onembed,
//					NULL,
					ostride,
					odist,
					inembed,
//					NULL,
					istride,
					idist,
					CUFFT_C2R,
					batch );	
	}

	if (errFFT != CUFFT_SUCCESS) {
		printf("Fail to produce fft plan.\n");
	}
	
	return plan;
}

struct CudaScaleFunctor {
  const float s;

  CudaScaleFunctor(float _s) : s(_s) {}

  // This forces compilation with nvcc and is the reason why this is a .cu file
  __host__ __device__ float operator()(const float& x) const {
    return s * x;
  }
};

void fft(int batch, float * real, float2 * cplx, bool forwardFFT, cufftHandle* plan, int size, int dim)
{
	cufftHandle localPlan = (!plan) ? makeFftPlan(batch, real, cplx, forwardFFT, size, dim) : *plan;

	cufftResult errFFT;
	if (forwardFFT) {
		errFFT = cufftExecR2C(localPlan, real, cplx);
		if (errFFT != CUFFT_SUCCESS) {
			throw std::bad_alloc();
		}
	} else {
		errFFT = cufftExecC2R(localPlan, cplx, real);
		if (errFFT != CUFFT_SUCCESS) {
			throw std::bad_alloc();
		}

		// the result of ifft must be normalized in convolution applications.
		if (1) {
			int realSize = batch * size * size;
			float val = 1 / (float)(size * size);
			thrust::device_ptr<float> res(real);	
			if (DEBUG)
				std::cout << "transform bad?" << std::endl;
			thrust::transform(res, res + realSize, res, CudaScaleFunctor(val));
			if (DEBUG)
				std::cout << "transform ok" << std::endl;
			//thrust::fill(res, res + realSize - 1, (float)0);
		}
	}
	if (!plan)
		cufftDestroy(localPlan);
}

void loadLenaSep(const char * name, float * input, int size, int dim, int batchi, int big_size)
{
	std::ifstream in(name, ios::in);
	float * temp = new float[big_size * big_size];
	for (int i = 0; i < big_size; ++i)
		for(int j = 0; j < big_size; ++j)
			in >> temp[i * big_size + j];
	in.close();
	for (int i = 0; i < batchi; ++i)
		for (int j = 0; j < batchi; ++j)
			for (int k = 0; k < size; ++k)
				for (int m = 0; m < size; ++m)
					if ((i + k - size / 2) < 0 || (i + k - size / 2) >= big_size
						|| (j + m - size / 2) < 0 || (j + m - size / 2) >= big_size)
						input[(batchi * i + j) * size * size + k * size + m] = 0;
					else
						input[(batchi * i + j) * size * size + k * size + m] = 
							temp[(i + k - size/2) * size + j + m - size/2];
}

void loadLena(const char * name, float * input, int size, int dim, int batch)
{
	std::ifstream in(name, ios::in);
	for (int i = 0; i < size; ++i)
		for(int j = 0; j < size; ++j)
			in >> input[i * size + j];
	in.close();
}

void showRealResult(float * input, int size, const char * name)
{
	float * temp = new float[size*size];
	if (DEBUG)
		std::cout << "OK?" << std::endl;
	cudaMemcpyAsync(temp, input, size * size * sizeof(float), cudaMemcpyDeviceToHost);
	if (DEBUG)
		std::cout << "OK?" << std::endl;
	std::ofstream out(name, ios::out);
	if (DEBUG)
		std::cout << "OK?" << std::endl;
	for (int i = 0; i < size; ++i)
	{
		for(int j = 0; j < size; ++j)
		{
			if (j < size - 1)
				out << temp[i * size + j] << " ";
			else
				out << temp[i * size + j] << std::endl;
		}
	}
	out.close();
	delete temp;
}

void saveLena(const char * name, float * input, int size, int dim, int batch)
{
	std::ofstream out(name, ios::out);
	for (int i = 0; i < size; ++i)
	{
		for(int j = 0; j < size; ++j)
		{
			if (j < size - 1)
				out << input[i * size + j] << " ";
			else
				out << input[i * size + j] << std::endl;
		}
	}
	out.close();
}

void sobelKernel(float * kernel, int kernelSize, int size)
{
	/*
	kernel[0 * size + 0] = 1;
	kernel[0 * size + 1] = 2;
	kernel[0 * size + 2] = 1;
	kernel[1 * size + 0] = 0;
	kernel[1 * size + 1] = 0;
	kernel[1 * size + 2] = 0;
	kernel[2 * size + 0] = -1;
	kernel[2 * size + 1] = -2;
	kernel[2 * size + 2] = -1;
	*/
	kernel[0 * size + 2] = 1;
	kernel[1 * size + 2] = 2;
	kernel[2 * size + 2] = 1;
	kernel[0 * size + 1] = 0;
	kernel[1 * size + 1] = 0;
	kernel[2 * size + 1] = 0;
	kernel[0 * size + 0] = -1;
	kernel[1 * size + 0] = -2;
	kernel[2 * size + 0] = -1;
}

void initInput(float * input, int size, int dim, int batch)
{
	for (int k = 0; k < batch; ++k)
		for (int i = 0; i < size; ++i)
			for (int j = 0; j < size; ++j)
				input[k * size * size + i * size + j] = k + 1 + i + j;
}

void showInput(float * result, int size, int dim, int batch)
{
	for (int k = 0; k < batch; ++k)
	{
		printf("In batch: %d\n", k);
		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < size; ++j)
				printf("%.2lf ", result[k * size * size + i * size + j]);
			printf("\n");
		}
	}
}

void showOutput(float2 * result, int size, int dim, int batch)
{
	for (int k = 0; k < batch; ++k)
	{
		printf("In batch: %d\n", k);
		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < size; ++j)
				printf("%.2lf ", result[k * size * size + i * size + j].x);
			printf("\n");
		}
	}
}

__global__ void multikernel(float2 * A, float2 * B, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	float x = B[index].x * A[index].x - A[index].y * B[index].y;
	float y = B[index].y * A[index].x + B[index].x * A[index].y;
	A[index].x = x;
	A[index].y = y;
}

__global__ void multikernelShared(float2 * A, float2 * B, int size)
{
	__shared__ float2 As[512];
	__shared__ float2 Bs[512];
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	As[threadIdx.x] = A[index];
	Bs[threadIdx.x] = B[index];

	__syncthreads();

	float x = Bs[threadIdx.x].x * As[threadIdx.x].x - 
		As[threadIdx.x].y * Bs[threadIdx.x].y;
	float y = Bs[threadIdx.x].y * As[threadIdx.x].x + 
		Bs[threadIdx.x].x * As[threadIdx.x].y;
	A[index].x = x;
	A[index].y = y;
}

void printFFTResult(float2 * d_output,int size, const char * path)
{
	int size2DC = size * (size / 2 + 1);
	int sizeH = size2DC / size;
	float2 * temp = new float2[size2DC];
	if (DEBUG)
		std::cout << "Segment Fault?" << std::endl;
	cudaMemcpyAsync(temp, d_output, size2DC * sizeof(float2), cudaMemcpyDeviceToHost);
	if (DEBUG)
		std::cout << "Segment Fault?" << std::endl;
	std::ofstream out(path, ios::out);
	for (int i = 0; i < size; ++i)
	{
		for(int j = 0; j < sizeH; ++j)
		{
			if (DEBUG)
				std::cout << "Segment Fault?" << i << j << std::endl;
			if (j < sizeH - 1)
				out << temp[i * sizeH + j].x << " " << temp[i * sizeH + j].y << " ";
			else
				out << temp[i * sizeH + j].x << " " << temp[i * sizeH + j].y << std::endl;
		}
	}
	if (DEBUG)
		std::cout << "Segment Fault?" << std::endl;
	out.close();
	if (DEBUG)
		std::cout << "Segment Fault?" << std::endl;
	delete temp;
}

void fft_convolution()
{
	int batch = 1;
	int size = 512;
	int kernelSize = 3;
	int size2D = size*size;
	int size2DC = size * (size / 2 + 1);
	float * input;
	float * kernel;
	Complex * fftOutput;
	input = new float[batch * size2D];
	fftOutput = new Complex[batch * size2D];
	kernel = new float[size2D];
	int memInput = batch * sizeof(float) * size2D;
	int memOutput = batch * (sizeof(Complex) * size2DC);
	float * d_input;
	float * d_kernel;
	float * d_outputReal;
	Complex * d_output;
	Complex * d_kernelOutput;
	cufftHandle forwardPlan;
	cufftHandle backwardPlan;

	loadLena("lena.f", input, size, 2, batch);
	sobelKernel(kernel, kernelSize, size);

	cudaMalloc((void **) &d_input, memInput);
	cudaMalloc((void **) &d_kernel, memInput);
	cudaMalloc((void **) &d_output, memOutput);
	cudaMalloc((void **) &d_kernelOutput, memOutput);
	cudaMalloc((void **) &d_outputReal, memInput);
	cudaMemcpyAsync(d_input, input, memInput, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_kernel, kernel, memInput, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_output, fftOutput, memOutput, cudaMemcpyHostToDevice);

	forwardPlan = makeFftPlan(batch, d_input, d_output, true, size, 2);
	backwardPlan = makeFftPlan(batch, d_input, d_output, false, size, 2);
	cudaDeviceSynchronize();

	clock_t start, end;
	start = clock();
	for (int i = 0; i < 10000; ++i)
	{
		fft(batch, d_input, d_output, true, &forwardPlan, size, 2);
		fft(1, d_kernel, d_kernelOutput, true, &forwardPlan, size, 2);
		//printFFTResult(d_output, size, "FFTimage.r");
		//printFFTResult(d_kernelOutput, size, "FFTkernel.r");
		//showRealResult(d_input, size, "Realimage.r");
		//showRealResult(d_kernel, size, "Realkernel.r");
		multikernel<<<size, size>>>(d_output, d_kernelOutput, size);
		//multikernelShared<<<size, size>>>(d_output, d_kernelOutput, size);
		fft(batch, d_outputReal, d_output, false, &backwardPlan, size, 2);
		cudaDeviceSynchronize();
	}
	end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Convolution time for 10000 times: %lf\n", time);

	float * result = new float[batch * size2D];
	cudaMemcpyAsync(result, d_outputReal, memInput, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	saveLena("lenar.f", result, size, 2, batch);

	delete result, input, fftOutput, kernel;
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_outputReal);
	cudaFree(d_kernelOutput);
	cudaFree(d_kernel);
}

void fft_batch_convolution()
{
	int batch = 2;
	int size = 512;
	int kernelSize = 3;
	int size2D = size*size;
	int size2DC = size * (size / 2 + 1);
	float * input;
	//float * kernel;
	//Complex * fftOutput;
	input = new float[batch * size2D];
	//fftOutput = new Complex[batch * size2D];
	//kernel = new float[size2D];
	int memInput = batch * sizeof(float) * size2D;
	int memOutput = batch * (sizeof(Complex) * size2DC);
	float * d_input;
	//float * d_kernel;
	float * d_outputReal;
	Complex * d_output;
	cufftHandle forwardPlan;
	cufftHandle backwardPlan;

	loadLena("lena.f", input, size, 2, 1);
	sobelKernel(input + size2D, kernelSize, size);
	
	if (DEBUG)
		std::cout << "d_input bad" << std::endl;
	cudaMalloc((void **) &d_input, memInput);
	//cudaMalloc((void **) &d_kernel, memInput);
	if (DEBUG)
		std::cout << "d_output bad" << std::endl;
	cudaMalloc((void **) &d_output, memOutput);
	//cudaMalloc((void **) &d_kernelOutput, memOutput);
	if (DEBUG)
		std::cout << "d_outputReal bad" << std::endl;
	cudaMalloc((void **) &d_outputReal, memInput);
	if (DEBUG)
		std::cout << "d_outputReal bad" << std::endl;
	cudaMemcpyAsync(d_input, input, memInput, cudaMemcpyHostToDevice);
	if (DEBUG)
		std::cout << "transfer bad" << std::endl;
	//cudaMemcpyAsync(d_kernel, kernel, memInput, cudaMemcpyHostToDevice);
	//cudaMemcpyAsync(d_output, fftOutput, memOutput, cudaMemcpyHostToDevice);

	forwardPlan = makeFftPlan(batch, d_input, d_output, true, size, 2);
	if (DEBUG)
		std::cout << "forward bad" << std::endl;
	backwardPlan = makeFftPlan(1, d_input, d_output, false, size, 2);
	if (DEBUG)
		std::cout << "backward bad" << std::endl;
	cudaDeviceSynchronize();

	clock_t start, end;
	start = clock();
	for (int i = 0; i < 10000; ++i)
	{
		if (DEBUG)
			std::cout << "fft OK?" << std::endl;
		fft(batch, d_input, d_output, true, &forwardPlan, size, 2);
		//printFFTResult(d_output, size, "BFFTimage.r");
		//printFFTResult(d_output + size2DC, size, "BFFTkernel.r");
		//showRealResult(d_input, size, "BRealiamge.r");
		//showRealResult(d_input + size2D, size, "BRealkernel.r");
		if (DEBUG)
			std::cout << "fft bad" << std::endl;
		multikernel<<<size, size>>>(d_output, d_output + size2DC, size);
		//multikernelShared<<<size, size>>>(d_output, d_output + memOutput / batch, size);
		if (DEBUG)
			std::cout << "dot product bad" << std::endl;
		fft(1, d_outputReal, d_output, false, &backwardPlan, size, 2);
		if (DEBUG)
			std::cout << "ifft bad" << std::endl;
		cudaDeviceSynchronize();
	}
	end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Convolution time for 10000 times: %lf\n", time);

	float * result = new float[size2D];
	cudaMemcpyAsync(result, d_outputReal, memInput / batch, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	saveLena("lenar.f", result, size, 2, batch);

	delete result, input; //, fftOutput;
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_outputReal);
}

/*
void fft_block_convolution()
{
	int big_size = 512, size = 512;
	int kernelSize = 3;
	int padKernelSize = 2 * kernelSize - 1;
	int block_size = 5;
	int batchi = ceil((float)(size + 2 * kernelSize - 1) / block_size);
	int batch = batchi * batchi + 1;
	size = block_size;
	int size2D = size*size;
	int size2DC = size * (size / 2 + 1);
	float * input;
	//float * kernel;
	Complex * fftOutput;
	input = new float[batch * size2D];
	//kernel = new float[size2D];
	int memInput = batch * sizeof(float) * size2D;
	int memOutput = batch * (sizeof(Complex) * size2DC);
	float * d_input;
	//float * d_kernel;
	float * d_outputReal;
	Complex * d_output;
	//Complex * d_kernelOutput;
	cufftHandle forwardPlan;
	cufftHandle backwardPlan;

	loadLenaSep("lena.f", input, size, 2, batchi, big_size);
	sobelKernel(input + (batch - 1) * size2D, kernelSize, size);

	cudaMalloc((void **) &d_input, memInput);
	cudaMalloc((void **) &d_output, memOutput);
	cudaMalloc((void **) &d_outputReal, memInput - size2D);
	cudaMemcpyAsync(d_input, input, memInput, cudaMemcpyHostToDevice);

	forwardPlan = makeFftPlan(batch, d_input, d_output, true, size, 2);
	backwardPlan = makeFftPlan(batch-1, d_input, d_output, false, size, 2);
	cudaDeviceSynchronize();

	clock_t start, end;
	start = clock();
	//for (int i = 0; i < 10000; ++i)
	//{
		fft(batch, d_input, d_output, true, &forwardPlan, size, 2);
		//printFFTResult(d_output, size, "FFTimage.r");
		//printFFTResult(d_kernelOutput, size, "FFTkernel.r");
		//showRealResult(d_input, size, "Realimage.r");
		//showRealResult(d_kernel, size, "Realkernel.r");
		multiBlock<<<size, size>>>(d_output, size);
		//multikernelShared<<<size, size>>>(d_output, d_kernelOutput, size);
		fft(batchi-1, d_outputReal, d_output, false, &backwardPlan, size, 2);
		cudaDeviceSynchronize();
	//}
	end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Convolution time for 10000 times: %lf\n", time);

	float * result = new float[batch * size2D];
	cudaMemcpyAsync(result, d_outputReal, memInput, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	saveLena("lenar.f", result, size, 2, batch);

	delete result, input, fftOutput, kernel;
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_outputReal);
	cudaFree(d_kernelOutput);
	cudaFree(d_kernel);
}
*/

__global__ void multiBlock(float2 * input, int size, int batchi)
{
	__shared__ float2 Ks[8][5];
	if (threadIdx.y < size / 2 + 1)
		Ks[threadIdx.x][threadIdx.y] = 
			*(input + size * (size / 2 + 1) * batchi * batchi + threadIdx.x * (size / 2 + 1) + threadIdx.y);
	__syncthreads();

	int limit = pow(ceil((float)batchi / 4), 2);
	for (int i = 0; i < limit; ++i)
	{
		int index = batchi * batchi / 16 * blockIdx.x * size * (size / 2 + 1) + i * size * (size / 2 + 1) + threadIdx.x * (size / 2 + 1) + threadIdx.y;
		float op1x = input[index].x;
		float op1y = input[index].y;
		float x = Ks[threadIdx.x][threadIdx.y].x * op1x - 
			Ks[threadIdx.x][threadIdx.y].y * op1y;
		float y = Ks[threadIdx.x][threadIdx.y].y * op1x + 
			Ks[threadIdx.x][threadIdx.y].x * op1y;
		input[index].x = x;
		input[index].y = y;
	}
}

void kernel_test(int test_origin)
{
	//Block test
	int big_size = 512;
	int size = 8;
	int batchi = ceil((float)big_size / size);
	int batch = batchi * batchi + 1;
	float2 * input;
	cudaMalloc((void**) &input, sizeof(float2) * batch * size * (size / 2 + 1));
	int block_size = ceil((float)batchi / 4);
	dim3 BLOCK_SIZE(size, size / 2 + 1);
	float2 * A, * B;
	cudaMalloc((void**) &A, sizeof(float2) * big_size * (big_size / 2 + 1));
	cudaMalloc((void**) &B, sizeof(float2) * big_size * (big_size / 2 + 1));

	clock_t sc, ec;	
	cudaEvent_t start, stop;
	float elapsedTime;

	if (test_origin & 1)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		sc = clock();

		for (int i = 0; i < 1000; ++i)
			multiBlock<<<16, BLOCK_SIZE>>>(input, size, batchi);

		cudaDeviceSynchronize();
		ec = clock();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cout << "total time " << elapsedTime/1000 << " s" << endl;
		cout << "total time cpu " << (double)(ec - sc) / CLOCKS_PER_SEC << " s" << endl;
	}
	
	if(test_origin & 2)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		sc = clock();

		for (int i = 0; i < 1000; ++i)
			multikernel<<<size, size>>>(A, B, big_size);

		cudaDeviceSynchronize();
		ec = clock();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		cout << "total time " << elapsedTime/1000 << " s" << endl;
		cout << "total time cpu " << (double)(ec - sc) / CLOCKS_PER_SEC << " s" << endl;
	}
}

int main(int argc, char ** argv)
{
	int mode = 0;
	if (argc > 1)
	{
		if (!strcmp(argv[1], "all"))
			mode = 3;
		if (!strcmp(argv[1], "normal"))
			mode = 1;
		if (!strcmp(argv[1], "batch"))
			mode = 2;
		if (!strcmp(argv[1], "kernel1"))
			mode = 4;
		if (!strcmp(argv[1], "kernel2"))
			mode = 8;
		if (!strcmp(argv[1], "kernel_all"))
			mode = 16;
	}
	if (mode & 1)
		fft_convolution();
	if (mode & 2)
		fft_batch_convolution();
	if (mode & 4)
		kernel_test(1);
	if (mode & 8)
		kernel_test(2);
	if (mode & 16)
		kernel_test(3);
	return 0;
}

/*

void fft_convolution()
{
	int batch = 1;
	int size = 512;
	int kernelSize = 3;
	int size2D = size*size;
	float * input;
	float * kernel;
	Complex * fftOutput;
	input = new float[batch * size2D];
	fftOutput = new Complex[batch * size2D];
	kernel = new float[size2D];
	int memInput = batch * sizeof(float) * size2D;
	int memOutput = batch * (sizeof(Complex) * size2D / 2 + 1);
	float * d_input;
	float * d_kernel;
	float * d_outputReal;
	Complex * d_output;
	Complex * d_kernelOutput;
	cufftHandle forwardPlan;
	cufftHandle backwardPlan;

	loadLena("lena.f", input, size, 2, batch);
	sobelKernel(kernel, kernelSize, size);

	cudaMalloc((void **) &d_input, memInput);
	cudaMalloc((void **) &d_kernel, memInput);
	cudaMalloc((void **) &d_output, memOutput);
	cudaMalloc((void **) &d_kernelOutput, memOutput);
	cudaMalloc((void **) &d_outputReal, memInput);
	cudaMemcpyAsync(d_input, input, memInput, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_kernel, kernel, memInput, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_output, fftOutput, memOutput, cudaMemcpyHostToDevice);

	forwardPlan = makeFftPlan(batch, d_input, d_output, true, size, 2);
	backwardPlan = makeFftPlan(batch, d_input, d_output, false, size, 2);
	cudaDeviceSynchronize();

	clock_t start, end;
	start = clock();
	for (int i = 0; i < 10000; ++i)
	{
		fft(batch, d_input, d_output, true, &forwardPlan, size, 2);
		fft(1, d_kernel, d_kernelOutput, true, &forwardPlan, size, 2);
		multikernel<<<size, size>>>(d_output, d_kernelOutput, size);
		fft(batch, d_outputReal, d_output, false, &backwardPlan, size, 2);
		cudaDeviceSynchronize();
	}
	end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Convolution time for 10000 times: %lf\n", time);

	float * result = new float[batch * size2D];
	cudaMemcpyAsync(result, d_outputReal, memInput, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	saveLena("lenar.f", result, size, 2, batch);

	delete result, input, fftOutput, kernel;
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_outputReal);
	cudaFree(d_kernelOutput);
	cudaFree(d_kernel);
}

void fft_block_convolution()
{

}

int main()
{
	fft_convolution();
	return 0;
}

/*
int simpleTest()
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
}*/
