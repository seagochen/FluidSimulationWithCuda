/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Mar 23, 2014
* <Last Time>     Mar 25, 2014
* <File Name>     ExtFunctions.h
*/

#ifndef __ext_functions_h__
#define __ext_functions_h__

#include <vector>
#include "FunctionHelper.h"

using sge::FunctionHelper;
using std::vector;

extern bool CreateCompNodesForDevice
	( vector<double*> *vectDens, vector<double*> *vectVelU, vector<double*> *vectVelV, 
	vector<double*> *vectVelW, vector<double*> *vectObst, 
	FunctionHelper *helper, size_t size, size_t nodes );

extern bool CreateCompNodesForDevice( vector<double*> *vectBuf, FunctionHelper *helper, size_t size, size_t nodes );

extern bool CreateCompNodesForHost
	( vector<double*> *vectDens, vector<double*> *vectVelU, vector<double*> *vectVelV, vector<double*> *vectVelW,
	vector<double*> *vectObst, FunctionHelper *helper, size_t size, size_t nodes );

#endif