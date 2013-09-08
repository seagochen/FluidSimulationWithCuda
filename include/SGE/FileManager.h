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
*	內含一個與文件操作相關的類
*	目前擁有的功能有：
*	- 從文本文件中讀取數據；
*	- 將數據保存至文本文件中；
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
	/* 選擇文件的格式 */
	enum SGFILEFORMAT
	{
		SG_FILE_FORMAT_XML = 0x0a,
		SG_FILE_FORMAT_TXT = 0x0b,
		SG_FILE_FORMAT_XF  = 0x0c,
	};

	/* 選擇創建的文件的打開方式 */
	enum SGFILEOPENTYPE
	{
		SG_FILE_OPEN_DEFAULT,  // 打开并创建全新的文档方式
		SG_FILE_OPEN_APPEND,   // 打开并添加数据至已有的文档方式
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