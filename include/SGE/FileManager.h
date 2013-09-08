/**
* ------------------------------------the MIT License--------------------------------------------------
*
* Copyright (C) year 2013 Orlando Chen <seagochen@gmail.com>
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
* <File>        FileManager.h
* <Specification>
*	�Ⱥ�һ���c�ļ��������P���
*	Ŀǰ���еĹ����У�
*	- ���ı��ļ����xȡ������
*	- �������������ı��ļ��У�
*/

#ifndef _SEAGOSOFT_FILE_MANAGER_H_
#define _SEAGOSOFT_FILE_MANAGER_H_

#include <string>
#include <sstream>

#ifdef _In_Dll_File
#include "Auxiliaries.h"
#endif

namespace sge
{
	/* �x���ļ��ĸ�ʽ */
	enum SGFILEFORMAT
	{
		SG_FILE_FORMAT_XML = 0x0a,
		SG_FILE_FORMAT_TXT = 0x0b,
		SG_FILE_FORMAT_XF  = 0x0c,
	};

	/* �x�񄓽����ļ��Ĵ��_��ʽ */
	enum SGFILEOPENTYPE
	{
		SG_FILE_OPEN_DEFAULT,  // �򿪲�����ȫ�µ��ĵ���ʽ
		SG_FILE_OPEN_APPEND,   // �򿪲�������������е��ĵ���ʽ
	};

	class FileManager
	{
	public:
		// Load data from external file
		_DLL std::string GetDataFromFile(const char *filename);
		
		// Save data to external file
		_DLL void SetDataToFile(
			std::string data, const char *filename, SGFILEOPENTYPE mode);
	};
}

#endif