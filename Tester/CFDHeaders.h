/**
*
* Copyright (C) <2013> <Orlando Chen>
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
* associated documentation files (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial
* portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
* NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
* <Author>      Orlando Chen
* <Date>        Sep 25, 2013
* <File>        CFDHeaders.h
*/


#ifndef _SEAGOSOFT_COMPUTATIONAL_FLUID_DYNAMICS_H_
#define _SEAGOSOFT_COMPUTATIONAL_FLUID_DYNAMICS_H_

#pragma once

#define SIMAREA_WIDTH  256
#define SIMAREA_HEIGHT 256

#include <Eigen\Dense>
#define elif  else if

using Eigen::Vector2d;
using Eigen::Vector2i;


namespace sge 
{
	enum SamplingMode
	{
		samPointClamp = 0x00,   // Pick sampling from a point
		samLinear     = 0x01,   // Pick sampling by using linear method
	};

	enum SelectingMode
	{
		SelectDataFromOrigin  = 0x10,
		SelectDataFromUpdate  = 0x11,
		YieldDataToOrigin     = 0x12,
		YieldDataToUpdate     = 0x13,
	};

	enum SwappingMode
	{
		SwapDataFromOriginToUpdate = 0x20,
		SwapDataFromUpdateToOrigin = 0x21,
	};

	/* Simulator Header of Computational Fluid Dynamics */
	class SGCFD2D
	{
	private:
		// Vector fields of velocity
		Vector2d velocity_origin[SIMAREA_WIDTH+2][SIMAREA_HEIGHT+2];
		Vector2d velocity_update[SIMAREA_WIDTH+2][SIMAREA_HEIGHT+2];
		// Scalar fields of density
		double density_origin[SIMAREA_WIDTH+2][SIMAREA_HEIGHT+2];
		double density_update[SIMAREA_WIDTH+2][SIMAREA_HEIGHT+2];

	public:
		// Sampling method for vector field
		Vector2d *SamplingFromVectorField(SamplingMode pmode, SelectingMode smode, Vector2i *CellIndex);
		// Sampling method for scalar field
		double    SamplingFromScalarField(SamplingMode pmode, SelectingMode smode, Vector2i *CellIndex);

		// Yield value to vector field
		void YieldValueToVectorField(SelectingMode smode, Vector2d *value, Vector2i *CellIndex);
		// Yield value to scalar field
		void YieldValueToScalarField(SelectingMode smode, double    value, Vector2i *CellIndex);

		// Swap value of vector fields
		void SwapVectorField(SwappingMode wmode);
		// Swap value of scalar fields
		void SwapScalarField(SwappingMode wmode);

		// Sampling from last updated vector field
		Vector2d *SamplingFromLastVectorField(SamplingMode pmode, Vector2i *CellIndex);
		// Sampling from last updated scalar field
		double    SamplingFromLastScalarField(SamplingMode pmode, Vector2i *CellIndex);


	private:
		/// internal functions ///

		Vector2d *SamplingVectorLinear(SelectingMode smode, Vector2i *CellIndex);
		double    SamplingScalarLinear(SelectingMode smode, Vector2i *CellIndex);
		Vector2d *SamplingVectorPointer(SelectingMode smode, Vector2i *CellIndex);
		double    SamplingScalarPointer(SelectingMode smode, Vector2i *CellIndex);

		Vector2i *SamplePoint(int u, int v);
	};

};

#endif