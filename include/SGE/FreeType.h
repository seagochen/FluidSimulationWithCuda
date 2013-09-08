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


#ifndef _SEAGOSOFT_FREETYPE_H_
#define _SEAGOSOFT_FREETYPE_H_

#include <windows.h>
//Some STL headers
#include <vector>
#include <string>

#ifdef _In_Dll_File
//OpenGL Headers 
#include <GL\glew.h>
#include <GL\glut.h>
//Using the STL exception library increases the
//chances that someone else using our code will corretly
//catch any exceptions that we throw.
//#include <stdexcept>
#include <SGE\Auxiliaries.h>

#endif

//MSVC will spit out all sorts of useless warnings if
//you create vectors of strings, this pragma gets rid of them.
//#pragma warning(disable: 4786) 

///Wrap everything in a namespace, that we can use common
///function names like "print" without worrying about
///overlapping with anyone else's code.

namespace sge {

//Inside of this namespace, give ourselves the ability
//to write just "vector" instead of "std::vector"
using std::vector;

//Ditto for string.
using std::string;

//This holds all of the information related to any
//freetype font that we want to create.  
class FreeType {
private:
	float h;			///< Holds the height of the font.
	GLuint * textures;	///< Holds the texture id's 
	GLuint list_base;	///< Holds the first display list id

public:
	//The init function will create a font of the height h from the file fname.
	_DLL void Init(const char * filename, unsigned int font_height);

	//Free all the resources assosiated with the font.
	_DLL void Clean();

	//The flagship function of the library - this thing will print
	//out text at window coordinates x,y, using the font ft_font.
	//The current modelview matrix will also be applied to the text. 
	_DLL void PrintText(const FreeType &freetype_in, float x, float y, const char *fmt, ...);
};

};

#endif