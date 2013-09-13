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
* <Date>        Sep 13, 2013
* <File>        Math.h
*/

#ifndef _SEAGOSOFT_MATH_H_
#define _SEAGOSOFT_MATH_H_

#include <Eigen\Dense>

namespace sge
{
	using Eigen::Vector2f;
	using Eigen::Vector3f;
	using Eigen::Vector4f;

	class float2
	{
	public:
		float x, y;

	public:
		float2() : x(0), y(0) {};
		float2(float x_in, float y_in) : 
			x(x_in), y(y_in) {};
	};

	class float3
	{
	public:
		float x, y, z;

	public:
		float3() : x(0), y(0), z(0) {};
		float3(float2 const *data_in, float z_in) :
			x(data_in->x), y(data_in->y), z(z_in) {};
		float3(float x_in, float y_in, float z_in) : 
			x(x_in), y(y_in), z(z_in) {};
	};

	class float4
	{
	public:
		float x, y, z, w;

	public:
		float4() : x(0), y(0), z(0), w(0) {};
		float4(float2 const *data_in, float z_in, float w_in) :
			x(data_in->x), y(data_in->y), z(z_in), w(w_in) {};
		float4(float3 const *data_in, float w_in) :
			x(data_in->x), y(data_in->y), z(data_in->z), w(w_in) {};
		float4(float x_in, float y_in, float z_in, float w_in) : 
			x(x_in), y(y_in), z(z_in), w(w_in) {};
	};

	Vector2f *ToVector(float2 const *in);
	Vector3f *ToVector(float3 const *in);
	Vector4f *ToVector(float4 const *in);
};

#endif