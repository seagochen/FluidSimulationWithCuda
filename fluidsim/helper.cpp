#include "main.h"


bool CheckHandleError ( int nShaderObjs, ... )
{
	if ( nShaderObjs < 1 )
	{
		cout << "call this function must specified the number of shader objects first and then pass the value" << endl;
		return false;
	}
	
	va_list list; int i = 1; bool fin = true;
	va_start ( list, nShaderObjs );
	{
		for ( ; i <= nShaderObjs; i++ )
		{
			GLuint value = va_arg ( list, GLuint );
			if ( value == 0 )
			{
				cout << "Error> the No." << i << " handle is null" << endl;
				fin = false;
			}
		}
		cout << "handle checker is finished" << endl;
	}
	va_end ( list );

	return fin;
};


void CreateShaders 
	( Shader *shader_out, GLuint *prog_out, GLuint *bfVert_out, GLuint *bfFrag_out, GLuint *rcVert_out, GLuint *rcFrag_out )
{
	// Create shader helper
	shader_out = new Shader();

	// Create shader objects from source
	shader_out->CreateShaderObj ( ".\\shader\\backface.vert", SG_VERTEX, bfVert_out );
	shader_out->CreateShaderObj ( ".\\shader\\backface.frag", SG_FRAGMENT, bfFrag_out );
	shader_out->CreateShaderObj ( ".\\shader\\raycasting.vert", SG_VERTEX, rcVert_out );
	shader_out->CreateShaderObj ( ".\\shader\\raycasting.frag", SG_FRAGMENT, rcFrag_out );

	// Check error
	if ( !CheckHandleError ( 4, *bfVert_out, *bfFrag_out, *rcVert_out, *rcFrag_out ) )
	{
		cout << "create shaders object failed" << endl;
		exit (1);
	}
	
	// Create shader program object
	shader_out->CreateProgmObj ( prog_out );

	// Check error
	if ( !CheckHandleError ( 1, *prog_out) )
	{
		cout << "create program object failed" << endl;
		exit (1);
	}
}


// Sets 1-D texture for transfer function
GLuint Create1DTransFunc ( void )
{
	// Define the transfer function
	GLubyte *tff = (GLubyte*) malloc ( sizeof(GLubyte) * 256 * 4 );
	for ( int i = 0; i < 256; i++ )
	{
		tff [ i * 4 + 0 ] = i;
		tff [ i * 4 + 1 ] = i;
		tff [ i * 4 + 2 ] = i;
		tff [ i * 4 + 3 ] = 1;
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

// Sets 2-D texture for backface
GLuint Create2DBackFace ( void )
{
    GLuint backFace2DTex;
    glGenTextures(1, &backFace2DTex);
    glBindTexture(GL_TEXTURE_2D, backFace2DTex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, BACKFACE_SIZE_X, BACKFACE_SIZE_X, 0, GL_RGBA, GL_FLOAT, NULL);

	cout << "2D backface created" << endl;

	return backFace2DTex;
};

// Sets 3-D texture for volumetric data 
GLuint Create3DVolumetric ( void )
{
    FILE *fp;
    size_t size = 256 * 256 * 225; // width * length * depth
    GLubyte *data = new GLubyte[size];
 
	if ( !(fp = fopen(".\\res\\head256.raw", "rb")) )
    {
        cout << "Error: opening .raw file failed" << endl;
        exit ( 1 );
    }

    if ( fread(data, sizeof(char), size, fp)!= size) 
    {
        cout << "Error: read .raw file failed" << endl;
        exit ( 1 );
    }

    fclose ( fp );

	// Generate 3D textuer
	GLuint volTex;
    glGenTextures(1, &volTex);
    glBindTexture(GL_TEXTURE_3D, volTex);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    
	// pixel transfer happens here from client to OpenGL server
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, 256, 256, 225, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, data);

    delete []data;

    cout << "3D volume texture created" << endl;

    return volTex;
};


GLuint CreateFrameBuffer ( GLuint texObj )
{
    // Create a depth buffer for framebuffer
    GLuint depthBuffer;
    glGenRenderbuffers(1, &depthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, BACKFACE_SIZE_X, BACKFACE_SIZE_X);

    // Attach the texture and the depth buffer to the framebuffer
	GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texObj, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
	
	// Check Framebuffer status
	if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE )
    {
		cout << "framebuffer is not complete" << endl;
		exit(EXIT_FAILURE);
    }
    glEnable(GL_DEPTH_TEST);    

	cout << "framebuffer created" << endl;
	
	return framebuffer;
};


#include <GLM\glm.hpp>
#include <GLM\gtc\matrix_transform.hpp>
#include <GLM\gtx\transform2.hpp>
#include <GLM\gtc\type_ptr.hpp>


void RenderingFace ( GLenum cullFace, GLfloat angle, GLuint program, GLuint cluster )
{
	using namespace glm;
	
	// Clear background color and depth buffer
    glClearColor ( 0.2f,0.2f,0.2f,1.0f );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    
	//  Set projection and lookat matrix
    mat4 projection = perspective ( 60.0f, (GLfloat)BACKFACE_SIZE_X/BACKFACE_SIZE_X, 0.1f, 400.f );
    mat4 view = lookAt (
		vec3(0.0f, 0.0f, 2.0f),
		vec3(0.0f, 0.0f, 0.0f), 
		vec3(0.0f, 1.0f, 0.0f));

	// Set model view matrix
    mat4 model = mat4(1.0f);
	model = model * rotate ( (float)angle, vec3(0.0f, 1.0f, 0.0f) );
    
	// Rotate and translate the view matrix, let object seems to "stand up"
	// Because, original volumetric data is "lying down" on ground.
	model = model * rotate ( 90.0f, vec3(1.0f, 0.0f, 0.0f) );
	model = model * translate ( vec3(-0.5f, -0.5f, -0.5f) ); 
    
	// Finally, we focus on setting the Model View Projection Matrix (MVP matrix)
	// Notice that the matrix multiplication order: reverse order of transform
    mat4 mvp = projection * view * model;

	// Returns an integer that represents the location of a specific uniform variable within a shader program
    GLuint mvpIdx = glGetUniformLocation ( program, "MVP" );
    
	if ( mvpIdx >= 0 )
    {
    	glUniformMatrix4fv ( mvpIdx, 1, GL_FALSE, &mvp[0][0] );
    }
    else
    {
    	cerr << "can't get the MVP" << endl;
    }
	    
	// Draw agent box
	glEnable ( GL_CULL_FACE );
	glCullFace ( cullFace );
	glBindVertexArray ( cluster );
	glDrawElements ( GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint *)NULL );
	glDisable ( GL_CULL_FACE );
}


void SetVolumeInfoUinforms ( GLuint program, GLuint Tex1DTrans, GLuint Tex2DBF, GLuint Tex3DVol, 
							GLfloat width, GLfloat height, GLfloat stepsize )
{
	// Set the uniform of screen size
    GLint screenSizeLoc = glGetUniformLocation ( program, "ScreenSize" );
    if ( screenSizeLoc >= 0 )
    {
		// Incoming two value, width and height
		glUniform2f ( screenSizeLoc, width, height );
    }
    else
    {
		cout << "ScreenSize is not bind to the uniform" << endl;
    }

	// Set the step length
    GLint stepSizeLoc = glGetUniformLocation ( program, "StepSize" );
	if ( stepSizeLoc >= 0 )
    {
		// Incoming one value, the step size
		glUniform1f ( stepSizeLoc, stepsize );
    }
    else
    {
		cout << "StepSize is not bind to the uniform" << endl;
    }
    
	// Set the transfer function
	GLint transferFuncLoc = glGetUniformLocation ( program, "TransferFunc" );
    if ( transferFuncLoc >= 0 )
	{
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_1D, Tex1DTrans );
		glUniform1i ( transferFuncLoc, 0 );
    }
    else
    {
		cout << "TransferFunc is not bind to the uniform" << endl;
    }

	// Set the back face as exit point for ray casting
	GLint backFaceLoc = glGetUniformLocation ( program, "ExitPoints" );
	if ( backFaceLoc >= 0 )
    {
		glActiveTexture ( GL_TEXTURE1 );
		glBindTexture(GL_TEXTURE_2D, Tex2DBF);
		glUniform1i(backFaceLoc, 1);
    }
    else
    {
		cout << "ExitPoints is not bind to the uniform" << endl;
    }

	// Set the uniform to hold the data of volumetric data
	GLint volumeLoc = glGetUniformLocation(program, "VolumeTex");
	if (volumeLoc >= 0)
    {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_3D, Tex3DVol);
		glUniform1i(volumeLoc, 2);
    }
    else
    {
		cout << "VolumeTex is not bind to the uniform" << endl;
    }    
};



GLuint InitVerticesBufferObj ( void )
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
	GLuint cluster;
    glGenVertexArrays ( 1, &cluster );
    glBindVertexArray ( cluster );

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

	cout << "agent object finished" << endl;

	return cluster;
};
