/**
* ------------------------------------the MIT License--------------------------------------------------
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
* <Date>        8/25/2013
* <File>        Auxiliaries.h
* <Specification>
*	ԓ�ļ��ȶ��x��һЩ�꣬�@Щ�������o�����߁K�����������^�ļ���Դ���a�ļ��С�
*	Ŀǰ�����������ݣ�
*	- ָ�ጷ��������ᘌ��Ϊ���ָᘣ��B�m�ăȴ���g���Լ�ָᘔ��M��ጷţ�
*	- dll�ļ�����������ӿ����������ں�����׃��ǰ�����@Щ���̖��
*		����¶�ӿڣ��Ա�����������M���L����
*/

#ifndef _SEAGOSOFT_AUXILIARY_H_
#define _SEAGOSOFT_AUXILIARY_H_

#ifdef SAFE_FREE_PTR(ptr)
#undef SAFE_FREE_PTR(ptr)
#endif

#ifdef SAFE_DELT_PTR(ptr)
#undef SAFE_DELT_PTR(ptr)
#endif

#ifdef SAFE_DELT_ARR(ptr)
#undef SAFE_DELT_ARR(ptr)
#endif

#ifdef _DLL
#undef _DLL
#endif

#ifdef pterror(str)
#undef pterror(str)
#endif

#define SAFE_FREE_PTR(ptr)   { if( ptr ) free(ptr);  ptr = NULL; }   // ��ȫ���ͷ���calloc��malloc����������ָ��
#define SAFE_DELT_PTR(ptr)   { if( ptr ) delete ptr; ptr = NULL; }   // ��ȫ���ͷ�һ��ָ��
#define SAFE_DELT_ARR(ptr)   { if( ptr ) delete []ptr; ptr = NULL; } // ��ȫ���ͷ�ָ������

//#define DllImport   __declspec( dllimport )
//#define DllExport   __declspec( dllexport )

#ifdef _In_Dll_File
#define _DLL __declspec(dllexport)    // ����ͷ�ļ���DLL�е�Դ����ģ������ʱ����
#else
#define _DLL __declspec(dllimport)    // ����ͷ�ļ������������е�Դ����ģ������ʱ����  
#endif  

#include <stdio.h>

#define pterror(str) { \
	printf("Error> %s, check your code at line %d of file %s\n", \
	str, __LINE__, __FILE__); exit(1);}

#ifdef SG_OK
#undef SG_OK
#endif

#ifdef SG_FAIL
#undef SG_FAIL
#endif

#ifdef SG_ERROR
#undef SG_ERROR
#endif

#define SG_OK     1
#define SG_FAIL   0
#define SG_ERROR -1

#include <GL\glew.h>
#include <GL\glut.h>

#endif