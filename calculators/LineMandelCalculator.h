/**
 * @file LineMandelCalculator.h
 * @author Lukáš Wagner <xwagne10@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#ifndef LINEMANDELCALCULATOR_H
#define LINEMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int* calculateMandelbrot();

private:
    // @TODO add all internal parameters
    int* data;
    float* template_row;
    float* template_col;
    float* temp_real;
    float* temp_imag;
};

#endif
