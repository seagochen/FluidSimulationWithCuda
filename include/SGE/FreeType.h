#ifndef _SEAGOSOFT_FREETYPE_H_
#define _SEAGOSOFT_FREETYPE_H_

//OpenGL Headers 
#include <windows.h>
#include <GL\glew.h>
#include <GL\glut.h>

//Some STL headers
#include <vector>
#include <string>

//Using the STL exception library increases the
//chances that someone else using our code will corretly
//catch any exceptions that we throw.
//#include <stdexcept>
#include <SGE\Auxiliaries.h>

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