#include "CFD2DSim.h"

using namespace sge;


#define density   cfd2D.density_update[i][j]
#define densityL  cfd2D.density_update[i-1][j]
#define densityR  cfd2D.density_update[i+1][j]
#define densityU  cfd2D.density_update[i][j+1]
#define densityD  cfd2D.density_update[i][j-1]
#define density0  cfd2D.density_origin[i][j]
#define density0L cfd2D.density_origin[i-1][j]
#define density0R cfd2D.density_origin[i+1][j]
#define density0U cfd2D.density_origin[i][j+1]
#define density0D cfd2D.density_origin[i][j-1]

void SGCFD2DSim::ScalarLineSolveFunc(double a, double c)
{
	for (int i=1; i <= CELLSU; i++)
	{
		for (int j=1; j <= CELLSV; j++)
		{
			density = (density0 + a * (densityL + densityR + densityU + densityD)) / c;
		}
	}
};


#define velocity   cfd2D.velocity_update[i][j]
#define velocityL  cfd2D.velocity_update[i-1][j]
#define velocityR  cfd2D.velocity_update[i+1][j]
#define velocityU  cfd2D.velocity_update[i][j+1]
#define velocityD  cfd2D.velocity_update[i][j-1]
#define velocity0  cfd2D.velocity_origin[i][j]
#define velocity0L cfd2D.velocity_origin[i-1][j]
#define velocity0R cfd2D.velocity_origin[i+1][j]
#define velocity0U cfd2D.velocity_origin[i][j+1]
#define velocity0D cfd2D.velocity_origin[i][j-1]

void SGCFD2DSim::VectorLineSolveFunc(double a, double c)
{
	for (int i=0; i <= CELLSU; i++)
	{
		for (int j=0; j <= CELLSV; j++)
		{
			velocity = (velocity0 + a * (velocityL + velocityR + velocityU + velocityD)) / c;
		}
	}
};


void SGCFD2DSim::ScalarDiffuse(double diff, double dt)
{
	double a = dt * diff * CELLSU * CELLSV;
	ScalarLineSolveFunc(a, 1+4*a);
};


void SGCFD2DSim::VecotrDiffuse(double diff, double dt)
{
	double a = dt * diff * CELLSU * CELLSV;
	VectorLineSolveFunc(a, 1+4*a);
};