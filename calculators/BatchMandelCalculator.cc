/**
 * @file BatchMandelCalculator.cc
 * @author Lukáš Wagner <xwagne10@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	// @TODO allocate & prefill memory
	data = (int *)(aligned_alloc(64, height * width * sizeof(int)));
	template_row = (float*)(aligned_alloc(64, width*height * sizeof(float)));
	template_col = (float*)(aligned_alloc(64, height*width * sizeof(float)));
	temp_real = (float*)(aligned_alloc(64, height*width*sizeof(float)));
	temp_imag = (float*)(aligned_alloc(64, height*width*sizeof(float)));


	for (int i = 0; i < height; i++)
	{
		// template_col[i] = y_start + i * dy; // current imaginary value

		for (int j = 0; j < width; j++){
			data[i*width+j]=-1;
			template_col[i*width + j] = y_start + i * dy;
			template_row[i*width + j] = x_start + j * dx;
			temp_real[i*width + j] = 0;
			temp_imag[i*width + j] = 0;
		}
	}

	// for (int j = 0; j < width; j++)
	// {
	// 	template_row[j] = x_start + j * dx; // current real value
	// }
}

BatchMandelCalculator::~BatchMandelCalculator() {
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


int * BatchMandelCalculator::calculateMandelbrot () {
	int* pdata = data;
	float* preal = temp_real;
	float* pimag = temp_imag;
	float* template_row = this->template_row;
	float* template_col = this->template_col;
	int cancel = 0;

	// for every row
	// for (int i = 0; i < height; i++)
	// {
		// split data to batches (to work with same cache)
		int n_batch, bstart = 0, bstop = batch_size;
		int batch_max = width*height/batch_size;
		for (int n_batch = 0; n_batch < batch_max; n_batch++)
		{
			// iterate whole batches
			for (int n_it = 0; n_it<limit; n_it++)
			{
				// stop iterating if every cell in batch is done
				cancel = 0;
				// #pragma omp simd reduction(|:cancel)
				// for (int j = bstart; j < bstop; j++){
				// 	cancel |= pdata[j];
				// }
				// if (cancel > 0)
				// 	break;


				// for every cell from this batch in this iteration
				#pragma omp simd reduction(|:cancel) aligned(preal:64) aligned(pdata:64), aligned(pimag:64), aligned(template_row:64), aligned(template_col:64), safelen(512), simdlen(512)
				for (int j = bstart; j < bstop; j++)
				{
					// load previous iteration of this cell
					float zReal = preal[j];
					float zImag = pimag[j];

					//////////////////////

					//squares
					float r2 = zReal * zReal;
					float i2 = zImag * zImag;

					// (A + Bi)^2 = A^2 + 2ABi - B^2
					pimag[j] = 2.0f * zReal * zImag + template_col[j];
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
			// advance to next batch in this row (next cache)
			bstart += batch_size;
			bstop += batch_size;
		}

		/////////////////////////////////////////////////////////////
		// finish nondivisible batch
		/////////////////////////////////////////////////////////////
		for (int n_it = 0; n_it<limit; n_it++)
		{
			cancel = 0;
			// #pragma omp simd reduction(|:cancel)
			// for (int j = bstart; j < width; j++){
			// 	cancel |= pdata[j];
			// }
			// if (cancel > 0)
			// 	break;

			#pragma omp simd reduction(|:cancel) aligned(preal:64) aligned(pdata:64), aligned(pimag:64), aligned(template_row:64), aligned(template_col:64), safelen(512), simdlen(512)
			for (int j = bstart; j<width*height; j++)
			{
				// load previous iteration of this cell
				float zReal = preal[j];
				float zImag = pimag[j];

				//////////////////////

				//squares
				float r2 = zReal * zReal;
				float i2 = zImag * zImag;

				// (A + Bi)^2 = A^2 + 2ABi - B^2
				pimag[j] = 2.0f * zReal * zImag + template_col[j];
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
		// // advance to next row
		// preal += width;
		// pimag += width;
		// pdata += width;
	// }
	return data;
}