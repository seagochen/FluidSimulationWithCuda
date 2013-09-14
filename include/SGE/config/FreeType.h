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
* <Origin>      Sven Olsen
* <Year>        2003
* <Modified>    Orlando Chen
* <Date>        Sep 12, 2013
* <File>        FreeType.h
*/

/**
* A quick and simple opengl font library that uses GNU freetype2, 
* written and distributed as part of a tutorial for nehe.gamedev.net. 
* Sven Olsen, 2003
*/

#ifndef _SEAGOSOFT_FREETYPE_H_
#define _SEAGOSOFT_FREETYPE_H_

#include <windows.h>

// Some STL headers
#include <vector>
#include <string>

#ifdef _In_Dll_File
#include <GL\glew.h>
#include <GL\glut.h>
#include "Auxiliaries.h"
#endif

namespace sge {

using std::vector;
using std::string;

	class FreeType {
	
	public:
		// The init function will create a font of the height h from the file fname.
		_DLL bool Init(const char * filename, unsigned int font_height);
		
		// Free all the resources assosiated with the font.
		_DLL void Clean(void);
		
		// The flagship function of the library - this thing will print out text at 
		// window coordinates x,y, using the font ft_font. The current modelview matrix 
		// will also be applied to the text. 
		_DLL void PrintText(const FreeType &freetype_in, float x, float y, const char *fmt, ...);
		
		// Call this function will enable FreeType
		_DLL void EnableFreeType(void);

		// Call this function to disable FreeType
		_DLL void DisableFreeType(void);

		// FreeType will bind a lot of texture IDs, so in order to avoid the texture ID user created
		// has be occupied, you may need to call this function checks whether the current texture ID
		// is available
		_DLL bool IsTextureIDAvailable(unsigned textureID);
};

};

#endif