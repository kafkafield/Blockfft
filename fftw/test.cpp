#include <fftw3.h>
#include <iostream>
#include <ctime>

int main()
{
	const int test_number = 10000;
	int size = 7;
	for (int iter = 1; iter <= 512; ++iter)
	{
		size = iter;
		fftw_complex *data = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size);
		fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size * size);
		fftw_plan f = fftw_plan_dft_2d(size, size, data, out, FFTW_FORWARD, FFTW_MEASURE);
		fftw_plan b = fftw_plan_dft_2d(size, size, data, out, FFTW_BACKWARD, FFTW_MEASURE);
		clock_t start, end;
		start = clock();
		for (int i = 0; i < test_number; ++i)
			fftw_execute(f);
		end = clock();
		double time = (double)(end - start) / CLOCKS_PER_SEC;
		std::cout << "Forward" << iter << ":" <<time << std::endl;
		start = clock();
		for (int i = 0; i < test_number; ++i)
			fftw_execute(b);
		end = clock();
		time = (double)(end - start) / CLOCKS_PER_SEC;
		std::cout << "Backward" << iter << ":" <<time << std::endl;
		fftw_destroy_plan(f);
		fftw_destroy_plan(b);
		fftw_free(data);
		fftw_free(out);
	}
	return 0;
}
