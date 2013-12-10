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
* <First>       Dec 4, 2013
* <Last>		Dec 4, 2013
* <File>        GLSLHelper.cpp
*/

#ifndef __glsl_helper_cpp_
#define __glsl_helper_cpp_

#include <GL\glew.h>
#include <GL\freeglut.h>

#include <iostream>
#include <fstream>

#include "GLSLHelper.h"

using namespace std;

void helperCheckOpenGLStatus ( const char* file, const int line )
{
    GLenum error;

    error = glGetError();
    if ( error != GL_NO_ERROR )
    {
		cout << "glError in file " << file
			<< "@line " << line << gluErrorString ( error ) << endl;

		exit(EXIT_FAILURE);
    }
}


// Initialize the vertices buffer object
void helperInitVerticesBufferObj ( GLuint *cluster )
{
	// How agent cube looks like by specified the coordinate positions of vertices
    GLfloat vertices[24] = 
	{
		0.0, 0.0, 0.0,
		0.0, 0.0, 1.0,
		0.0, 1.0, 0.0,
		0.0, 1.0, 1.0,
		1.0, 0.0, 0.0,
		1.0, 0.0, 1.0,
		1.0, 1.0, 0.0,
		1.0, 1.0, 1.0
	};
	
	// Drawing six faces of agent cube with triangles by counter clockwise
	// <Front> 1 5 7 3
	// <Back> 0 2 6 4
	// <Left> 0 1 3 2
	// <Right> 7 5 4 6
	// <Up> 2 3 7 6
	// <Down> 1 0 4 5
    GLuint indices[36] = 
	{
		1,5,7,
		7,3,1,
		0,2,6,
		6,4,0,
		0,1,3,
		3,2,0,
		7,5,4,
		4,6,7,
		2,3,7,
		7,6,2,
		1,0,4,
		4,5,1
	};

	// Generate the buffer indices
    GLuint GenBufferList[2];
    glGenBuffers ( 2, GenBufferList );
    GLuint ArrayBufferData  = GenBufferList [ 0 ];
    GLuint ElementArrayData = GenBufferList [ 1 ];

	/*
	* void glBindBuffer(GLenum target, GLuint buffer);
	* void glBufferData(GLenum target, GLsizeiptr size, const GLvoid * data, GLenum usage);
	*
	* ----------------------------------------------------------------------------------------------------------------------------
	*
	* glBindBuffer binds a buffer object to the specified buffer binding point.
	* Calling glBindBuffer with target set to one of the accepted symbolic constants and 
	* buffer set to the name of a buffer object binds that buffer object name to the target.
	* If no buffer object with name buffer exists, one is created with that name.
	* When a buffer object is bound to a target, the previous binding for that target is automatically broken.
	*
	* Buffer object names are unsigned integers. The value zero is reserved, but there is no default
	* buffer object for each buffer object target. Instead, buffer set to zero effectively unbinds any buffer 
	* object previously bound, and restores client memory usage for that buffer object target (if supported for that target).
	* Buffer object names and the corresponding buffer object contents are local to the shared object space of the
	* current GL rendering context; two rendering contexts share buffer object names only if they explicitly 
	* enable sharing between contexts through the appropriate GL windows interfaces functions.
	*
	* glGenBuffers must be used to generate a set of unused buffer object names.
	* 
	* ----------------------------------------------------------------------------------------------------------------------------
	*
	* glBufferData creates a new data store for the buffer object currently bound to target.
	* Any pre-existing data store is deleted. The new data store is created with the specified size in bytes and usage.
	* If data is not NULL, the data store is initialized with data from this pointer. 
	* In its initial state, the new data store is not mapped, it has a NULL mapped pointer, and its mapped access is GL_READ_WRITE.
	*
	* usage is a hint to the GL implementation as to how a buffer object's data store will be accessed.
	* This enables the GL implementation to make more intelligent decisions that may significantly impact buffer object performance.
	* It does not, however, constrain the actual usage of the data store. 
	* usage can be broken down into two parts: 
	* first, the frequency of access (modification and usage),
	* and second, the nature of that access.
	*/
	// Bind display array list, vertices list used here that indicates the coordinate position of vertices
	glBindBuffer ( GL_ARRAY_BUFFER, ArrayBufferData );
	glBufferData ( GL_ARRAY_BUFFER, 24 * sizeof(GLfloat), vertices, GL_STATIC_DRAW );
    
	// Bind element array list, indices used here that indicates the triangles drawing sequence
    glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, ElementArrayData );
    glBufferData ( GL_ELEMENT_ARRAY_BUFFER, 36 * sizeof(GLuint), indices, GL_STATIC_DRAW );
	
	// After that, we use a cluster for keeping the information of GenBufferList
	/*
	* void glGenVertexArrays(GLsizei n, GLuint *arrays);
	*
	* glGenVertexArrays returns n vertex array object names in arrays.
	* There is no guarantee that the names form a contiguous set of integers;
	* however, it is guaranteed that none of the returned names was in use immediately before the call to glGenVertexArrays.
	*
	* Vertex array object names returned by a call to glGenVertexArrays are not returned by subsequent calls,
	* unless they are first deleted with glDeleteVertexArrays.
	*
	* The names returned in arrays are marked as used, for the purposes of glGenVertexArrays only, 
	* but they acquire state and type only when they are first bound.
	*/
    glGenVertexArrays ( 1, cluster );
    glBindVertexArray ( *cluster );

	/*
	* void glEnableVertexAttribArray(GLuint index);
	* void glDisableVertexAttribArray(GLuint index);
	*
	* glEnableVertexAttribArray enables the generic vertex attribute array specified by index.
	* 
	* glDisableVertexAttribArray disables the generic vertex attribute array specified by index.
	* By default, all client-side capabilities are disabled, including all generic vertex attribute arrays. 
	* If enabled, the values in the generic vertex attribute array will be accessed and used for rendering 
	* when calls are made to vertex array commands such as 
	* glDrawArrays, glDrawElements, glDrawRangeElements, glMultiDrawElements, or glMultiDrawArrays.
	*/
    glEnableVertexAttribArray ( 0 ); // Enable ArrayBufferData
    glEnableVertexAttribArray ( 1 ); // Enable ElementArrayData

    // the vertex location is the same as the vertex color
    glBindBuffer ( GL_ARRAY_BUFFER, ArrayBufferData );
	/*
	* void glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid * pointer);
	* void glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const GLvoid * pointer);
	* void glVertexAttribLPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const GLvoid * pointer);
	*
	* glVertexAttribPointer, glVertexAttribIPointer and glVertexAttribLPointer 
	* specify the location and data format of the array of generic vertex attributes at index index to use when rendering.
	* size specifies the number of components per attribute and must be 1, 2, 3, 4, or GL_BGRA. 
	* type specifies the data type of each component,
	* and stride specifies the byte stride from one attribute to the next,
	* allowing vertices and attributes to be packed into a single array or stored in separate arrays.
	*/
    glVertexAttribPointer ( 0, 3, GL_FLOAT, GL_FALSE, 0, (GLfloat *)NULL );
    glVertexAttribPointer ( 1, 3, GL_FLOAT, GL_FALSE, 0, (GLfloat *)NULL );
    glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, ElementArrayData );
};


GLboolean helperCompilingStatusCheck ( GLuint shader )
{
    GLint err;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &err);
    if (GL_FALSE == err)
    {
		GLint logLen;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
		
		if (logLen > 0)
		{
			char* log = (char *)malloc(logLen);
			GLsizei written;
			glGetShaderInfoLog(shader, logLen, &written, log);
			cerr << "Shader log: " << log << endl;
			free(log);
		}
    }
    return err;
};


GLuint helperInitShaderObj ( const char *srcfile, GLenum shaderType )
{
    ifstream inFile ( srcfile, ifstream::in );
    
	// No file resource exists?
    if ( !inFile )
    {
		cerr << "Error openning file: " << srcfile << endl;
		exit(EXIT_FAILURE);
    }
    
    const int MAX_CNT = 10000;
    
	GLchar *shaderCode = (GLchar *) calloc ( MAX_CNT, sizeof(GLchar) );

    inFile.read(shaderCode, MAX_CNT);
    
	// End of file?
	if ( inFile.eof() )
    {
		size_t bytecnt = inFile.gcount();
		*(shaderCode + bytecnt) = '\0';
	}
    else if ( inFile.fail() )
    {
		cout << srcfile << "read failed " << endl;
    }
    else
    {
		cout << srcfile << "is too large" << endl;
    }

    // create the shader Object
    GLuint shader = glCreateShader(shaderType);
    if ( 0 == shader )
    {
		cerr << "Error creating vertex shader." << endl;
    }

    const GLchar* codeArray[] = { shaderCode };
    glShaderSource ( shader, 1, codeArray, NULL );
    
	free ( shaderCode );

    // Compile the shader
	glCompileShader ( shader );
	if ( GL_FALSE == helperCompilingStatusCheck ( shader ) )
    {
		cerr << "shader compilation failed" << endl;
    }

    return shader;
};


void helperInitPairShadersObj ( GLuint *VertHandler, const char *szVertSource, GLuint *FragHandler, const char *szFragSource )
{
	// To create vertex shader object
	*VertHandler = helperInitShaderObj ( szVertSource, GL_VERTEX_SHADER );
	// To create fragment shader object
    *FragHandler = helperInitShaderObj ( szFragSource, GL_FRAGMENT_SHADER );
};


GLint helperCheckShaderLinkStatus ( GLuint pgmHandle )
{
	GLint status;
    glGetProgramiv ( pgmHandle, GL_LINK_STATUS, &status );
    if ( GL_FALSE == status )
    {
		GLint logLen;
		glGetProgramiv ( pgmHandle, GL_INFO_LOG_LENGTH, &logLen );
		
		if ( logLen > 0 )
		{
			GLchar * log = (GLchar *) malloc(logLen);
			GLsizei written;
			glGetProgramInfoLog ( pgmHandle, logLen, &written, log );
			cerr << "Program log: " << log << endl;
		}
    }

    return status;
};


void helperLinkShader ( GLuint shaderPgm, GLuint newVertHandle, GLuint newFragHandle )
{
    const GLsizei maxCount = 2;
    GLsizei count;
    GLuint shaders [ maxCount ];
    glGetAttachedShaders ( shaderPgm, maxCount, &count, shaders );
    helperCheckOpenGLStatus ( __FILE__, __LINE__ );

    for ( int i = 0; i < count; i++ ) { glDetachShader ( shaderPgm, shaders [ i ] ); }
    
	glAttachShader ( shaderPgm, newVertHandle );
    glAttachShader ( shaderPgm, newFragHandle );
    helperCheckOpenGLStatus ( __FILE__, __LINE__ );
    
	glLinkProgram ( shaderPgm );
    
	if ( GL_FALSE == helperCheckShaderLinkStatus ( shaderPgm ) )
    {
		cerr << "Failed to relink shader program!" << endl;
		exit(EXIT_FAILURE);
	}
    helperCheckOpenGLStatus ( __FILE__, __LINE__ );
};


void helperCreateShaderProgram ( GLuint *progHander )
{
	*progHander = glCreateProgram ( );
	if ( 0 == *progHander )
    {
		cerr << "Error create shader program" << endl;
		exit ( EXIT_FAILURE );
	}
};

#endif