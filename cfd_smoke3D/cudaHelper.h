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
* <First>       Oct 30, 2013
* <Last>		Nov 1, 2013
* <File>        cudaHelper.h
*/

#ifndef __cuda_helper_h_
#define __cuda_helper_h_

#define cudaOpenLogFile(filename) \
	FILE *stream;  \
	stream = fopen(filename, "w"); 

#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stream, "cudaCheckErrors>>> %s (%s at %s:%d)\n", \
				msg, cudaGetErrorString(__err), \
				__FILE__, __LINE__); \
			goto Finished; \
		} \
	} while(0); \

#define cudaCloseLogFile()  \
	fclose(stream);

#define cudaFinished()  Finished:  

#define cudaDevice(gridDim, blockDim) <<<gridDim, blockDim>>>

#define cudaIndex2D(i, j, elements_x) ((j) * (elements_x) + (i))

#define cudaTrans2DTo3D(i, j, k, elements_x) { \
	k = cudaIndex2D(i, j, elements_x) / ((elements_x) * (elements_x)) ; \
	i = i % elements_x; \
	j = j % elements_x; \
	}

#define cudaIndex3D(i, j, k, elements_x) ((k) * elements_x * elements_x + (j) * elements_x + (i))

#endif