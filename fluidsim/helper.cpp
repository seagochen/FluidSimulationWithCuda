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
	shader_out->CreateShaderObj ( ".\\Shader\\backface.vert", SG_VERTEX, bfVert_out );
	shader_out->CreateShaderObj ( ".\\Shader\\backface.frag", SG_FRAGMENT, bfFrag_out );
	shader_out->CreateShaderObj ( ".\\Shader\\raycasting.vert", SG_VERTEX, rcVert_out );
	shader_out->CreateShaderObj ( ".\\Shader\\raycasting.frag", SG_FRAGMENT, rcFrag_out );

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
};


// Sets 3-D texture for volumetric data 
GLuint Create3DVolumFunc ( void )
{
};