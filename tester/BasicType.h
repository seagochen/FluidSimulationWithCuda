/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Mar 05, 2014
* <Last Time>     Mar 05, 2014
* <File Name>     BasicType.h
*/

#ifndef __basic_type_h__
#define __basic_type_h__

#define eqt    ==
#define not_eq !=
#define not    !
#define and    &&
#define or     ||
#define xand   &
#define xor    |

struct int3
{
	int x, y, z;
};

#define cudaIndex3D(i,j,k,elements_x) ((k)*elements_x*elements_x+(j)*elements_x+(i))

#endif