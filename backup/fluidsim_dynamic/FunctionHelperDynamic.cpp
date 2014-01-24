/**
* <Author>      Orlando Chen
* <First>       Jan 07, 2014
* <Last>		Jan 09, 2014
* <File>        FunctionHelperDynamic.cpp
*/

#include "FunctionHelperDynamic.h"

#define and &&
#define or  ||
#define eqt ==

std::string string_fmt ( const std::string fmt_str, ... )
{
	/* reserve 2 times as much as the length of the fmt_str */
    int final_n, n = fmt_str.size() * 2; 
    std::string str;
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while ( true )
	{
		/* wrap the plain char array into the unique_ptr */
        formatted.reset ( new char[n] ); 
        strcpy ( &formatted[0], fmt_str.c_str() );
        va_start ( ap, fmt_str );
        final_n = vsnprintf ( &formatted[0], n, fmt_str.c_str(), ap );
        va_end ( ap );
        if ( final_n < 0 or final_n >= n )
            n += abs( final_n - n + 1 );
        else
            break;
    }
    return std::string ( formatted.get() );
};

void cudaCheckErrors ( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if (__err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString(__err) );
		printf ( "%s \n", msg );
	}
};