/**
 * @file LineMandelCalculator.cc
 * @author Lukáš Wagner <xwagne10@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <string.h>


#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int *)(aligned_alloc(64, height * width * sizeof(int)));
	template_row = (float*)(aligned_alloc(64, width * sizeof(float)));
	template_col = (float*)(aligned_alloc(64, height * sizeof(float)));
	temp_real = (float*)(aligned_alloc(64, height*width*sizeof(float)));
	temp_imag = (float*)(aligned_alloc(64, height*width*sizeof(float)));


	for (int i = 0; i < height; i++)
	{
		template_col[i] = y_start + i * dy; // current imaginary value

		for (int j = 0; j < width; j++){
			data[i*width+j]=-1;
			temp_real[i*width + j] = 0;
			temp_imag[i*width + j] = 0;
		}
	}

	for (int j = 0; j < width; j++)
	{
		template_row[j] = x_start + j * dx; // current real value
	}
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory
	free(data);
	data = NULL;

	free(template_row);
	template_row = NULL;

	free(template_col);
	template_col = NULL;

	free(temp_real);
	temp_real = NULL;

	free(temp_imag);
	temp_imag = NULL;
}


// template <typename T>
// static inline int mandelbrot(T real, T imag, int n_it, int limit)
// {
// 	T zReal = real;
// 	T zImag = imag;

// 	//////////////////////
// 	T r2 = zReal * zReal;
// 	T i2 = zImag * zImag;

// 	if (r2 + i2 > 4.0f)
// 		return n_it;

// 	zImag = 2.0f * zReal * zImag + imag;
// 	zReal = r2 - i2 + real;
// 	//////////////////////

// 	return limit;
// }

int* LineMandelCalculator::calculateMandelbrot()
{
	int* pdata = data;
	float* preal = temp_real;
	float* pimag = temp_imag;
	float* template_row = this->template_row;
	float* template_col = this->template_col;
	int cancel = 0;

	// for every row
	for (int i = 0; i < height; i++)
	{
		// iterate whole rows
		for (int n_it = 0; n_it<limit; n_it++)
		{

			// stop iterating if every cell in row is done
			cancel = 0;
			// #pragma omp simd reduction(|:cancel)
			// for (int j = 0; j < width; j++){
			// 	cancel |= pdata[j];
			// }
			// if (cancel > 0)
			// 	break;


			// for every cell from this row in this iteration
			#pragma omp simd reduction(|:cancel) aligned(preal:64) aligned(pdata:64), aligned(pimag:64), aligned(template_row:64), aligned(template_col:64), safelen(512), simdlen(512)
			for (int j = 0; j < width; j++)
			{
				// load previous iteration of this cell
				float zReal = preal[j];
				float zImag = pimag[j];

				//////////////////////

				//squares
				float r2 = zReal * zReal;
				float i2 = zImag * zImag;

				// (A + Bi)^2 = A^2 + 2ABi - B^2
				pimag[j] = 2.0f * zReal * zImag + template_col[i];
				preal[j] = r2 - i2 + template_row[j];

				//////////////////////

				// result decision, hopefully branchless via masking
				int curr = pdata[j];
				int temp = (r2 + i2 > 4.0f)? n_it : curr;
				int temp2 = (n_it == limit-1)? limit : temp;
				pdata[j] = (curr < 0)? temp2 : curr;
				cancel |= pdata[j];
			}
			if (cancel > 0)
				break;
		}
		// advance to next row
		preal += width;
		pimag += width;
		pdata += width;
	}
	return data;
}
