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
* <File>        CFD2DSim.h
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

#define CELLSU   SIMAREA_WIDTH / 4
#define CELLSV   SIMAREA_HEIGHT/ 4

#define stepsize  0.01f  // step length (size) during a single time slice

namespace sge 
{
//////////////////////////////////////////////////////////////////////////////////////////
////

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

////
//////////////////////////////////////////////////////////////////////////////////////////
////

	/* Data Structure of Computational Fluid Dynamics */
	class SGCFD2D
	{
	public:
		// Vector fields of velocity
		Vector2d velocity_origin[CELLSU+2][CELLSV+2];
		Vector2d velocity_update[CELLSU+2][CELLSV+2];
		// Scalar fields of density
		double density_origin[CELLSU+2][CELLSV+2];
		double density_update[CELLSU+2][CELLSV+2];

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

		// Update vector field
		void UpdateVectorField(int u, int v, Vector2d *vel_in);
		// Update Scalar field
		void UpdateScalarField(int u, int v, double vel_in);

		// Cells in horizontal direction, including ghost cells
		inline int NumOfHorizontalCells() { return CELLSU + 2; };
		// Cells in vertical direction, including ghost cells
		inline int NumOfVerticalCells() { return CELLSV + 2; };
		// Cells in horizontal direction, excluding ghost cells
		inline int NumOfSimHorizontalCells() { return CELLSU; };
		// Cells in vertical direction, excluding ghost cells
		inline int NumOfSimVerticalCells() { return CELLSV; };

	private:
		/// internal functions ///

		Vector2d *SamplingVectorLinear(SelectingMode smode, Vector2i *CellIndex);
		double    SamplingScalarLinear(SelectingMode smode, Vector2i *CellIndex);
		Vector2d *SamplingVectorPointer(SelectingMode smode, Vector2i *CellIndex);
		double    SamplingScalarPointer(SelectingMode smode, Vector2i *CellIndex);

		Vector2i *SamplePoint(int u, int v);
	};

////
//////////////////////////////////////////////////////////////////////////////////////////
////

	/* Simulator of Computational Fluid Dynamics */
	class SGCFD2DSim
	{
	public:
		// Line solver function for scalar field
		void ScalarLineSolveFunc(double a, double c);
		// Line solver function for vector field
		void VectorLineSolveFunc(double a, double c);

		// Diffuse function for scalar field
		void ScalarDiffuse(double diff, double dt);
		// Diffuse function for vector field
		void VecotrDiffuse(double diff, double dt);

		// Advect function for scalar field
		void ScalarAdvect(double dt);
		// Advect function for vector field
		void VectorAdvect(double dt);

		// Project vector and scalar field
		void Project();
	};

////
//////////////////////////////////////////////////////////////////////////////////////////
};

#endif