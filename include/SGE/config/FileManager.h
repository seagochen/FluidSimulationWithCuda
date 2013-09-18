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
* <Date>        Sep 12, 2013
* <File>        FileManager.h
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
	enum SGFILEFORMAT
	{
		SG_FILE_FORMAT_XML = 0x0a,
		SG_FILE_FORMAT_TXT = 0x0b,
		SG_FILE_FORMAT_XF  = 0x0c,
	};

	enum SGFILEOPENMODE
	{
		SG_FILE_OPEN_DEFAULT,  // Create or overwrite a file
		SG_FILE_OPEN_APPEND,   // Create or append text to file
	};

	class FileManager
	{
	public:
		// Load data from external file
		_DLL SGRUNTIMEMSG GetDataFromFile(const char *filename, const char *str_out);
		
		// Save data to external file, using string
		_DLL SGRUNTIMEMSG SetDataToFile(std::string str_in, const char *filename, SGFILEOPENMODE mode);

		// Save data to external file, using const char *
		_DLL SGRUNTIMEMSG SetDataToFile(const char *str_in, const char *filename, SGFILEOPENMODE mode);
	};
}

#endif