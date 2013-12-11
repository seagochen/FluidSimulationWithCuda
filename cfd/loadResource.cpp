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
* <First>       Dec 12, 2013
* <Last>		Dec 12, 2013
* <File>        loadResource.cpp
*/

#ifndef __load_resource_cpp_
#define __load_resource_cpp_

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLT/glm/glm.hpp>
#include <GLT/glm/gtc/matrix_transform.hpp>
#include <GLT/glm/gtx/transform2.hpp>
#include <GLT/glm/gtc/type_ptr.hpp>

#include "cfdHeader.h"
#include "GLSLHelper.h"

using namespace std;


GLuint initTFF1DTex ( const char* filename )
{
    // Read in the user defined data of transfer function
    ifstream inFile ( filename, ifstream::in );
	if ( !inFile )
    {
		cerr << "Error openning file: " << filename << endl;
		exit(EXIT_FAILURE);
    }
    
    const int MAX_CNT = 10000;
    GLubyte *tff = (GLubyte *) calloc(MAX_CNT, sizeof(GLubyte));
    inFile.read ( reinterpret_cast<char *>(tff), MAX_CNT );

	// File is end?
    if (inFile.eof())
    {
		size_t bytecnt = inFile.gcount();
		*(tff + bytecnt) = '\0';
		cout << "bytecnt " << bytecnt << endl;
	}
    else if(inFile.fail())
    {
		cout << filename << "read failed " << endl;
    }
    else
    {
		cout << filename << "is too large" << endl;
    }
	
	GLuint tff1DTex;
    glGenTextures(1, &tff1DTex);
    glBindTexture(GL_TEXTURE_1D, tff1DTex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, tff);
    
	free(tff);    
    
	return tff1DTex;
};


GLuint initFace2DTex ( GLuint bfTexWidth, GLuint bfTexHeight )
{
    GLuint backFace2DTex;
    glGenTextures(1, &backFace2DTex);
    glBindTexture(GL_TEXTURE_2D, backFace2DTex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, bfTexWidth, bfTexHeight, 0, GL_RGBA, GL_FLOAT, NULL);

    // Create a depth buffer for framebuffer
    GLuint depthBuffer;
    glGenRenderbuffers(1, &depthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, bfTexWidth, bfTexHeight);

    // Attach the texture and the depth buffer to the framebuffer
    glGenFramebuffers(1, &m_frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_frameBuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, backFace2DTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
	
	// Check Framebuffer status
	if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE )
    {
		cout << "framebuffer is not complete" << endl;
		exit(EXIT_FAILURE);
    }
    glEnable(GL_DEPTH_TEST);    


	return backFace2DTex;
};


GLuint initVol3DTex ( const char* filename, GLuint w, GLuint h, GLuint d )
{
    FILE *fp;
    size_t size = w * h * d;
    GLubyte *data = new GLubyte[size];			  // 8bit
 
	if ( !(fp = fopen(filename, "rb")) )
    {
        cout << "Error: opening .raw file failed" << endl;
        exit ( EXIT_FAILURE );
    }
    else
    {
        cout << "OK: open .raw file successed" << endl;
    }
    if ( fread(data, sizeof(char), size, fp)!= size) 
    {
        cout << "Error: read .raw file failed" << endl;
        exit ( 1 );
    }
    else
    {
        cout << "OK: read .raw file successed" << endl;
    }
    fclose ( fp );

	// Generate 3D textuer
    glGenTextures(1, &m_volTexObj);
    glBindTexture(GL_TEXTURE_3D, m_volTexObj);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    
	// pixel transfer happens here from client to OpenGL server
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    //glTexImage3D (GL_TEXTURE_3D, 0, GL_INTENSITY, w, h, d, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);

    delete []data;
    cout << "volume texture created" << endl;

    return m_volTexObj;
};


void initShader ( void )
{
	// To create backface shader for first pass
	helperInitPairShadersObj ( 
		&m_bfVertHandle, "shader/backface.vert", 
		&m_bfFragHandle, "shader/backface.frag" );

	// To create raycasting shader for second pass
	helperInitPairShadersObj ( 
		&m_rcVertHandle, "shader/raycasting.vert", 
		&m_rcFragHandle, "shader/raycasting.frag" );

	// Link shader program
	helperCreateShaderProgram ( &m_programHandle );
};

#endif