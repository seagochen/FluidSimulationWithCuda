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
	// TODO
	for (int i=1; i <= CELLSU; i++)
	{
		for (int j=1; j <= CELLSV; j++)
		{
			density = (density0 + a * (densityL + densityR + densityU + densityD)) / c;
		}
	}
};

#undef density
#undef densityL
#undef densityR
#undef densityU
#undef densityD
#undef density0
#undef density0L
#undef density0R
#undef density0U
#undef density0D



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
	// TODO
	for (int i=0; i <= CELLSU; i++)
	{
		for (int j=0; j <= CELLSV; j++)
		{
			velocity = (velocity0 + a * (velocityL + velocityR + velocityU + velocityD)) / c;
		}
	}
};

#undef velocity
#undef velocityL
#undef velocityR
#undef velocityU
#undef velocityD
#undef velocity0
#undef velocity0L
#undef velocity0R
#undef velocity0U
#undef velocity0D



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


#define decode(ptr) int ups, vps; Vector2i px = *ptr; ups = U(px); vps = V(px);
#define sampleS(i, j, value) Vector2i temps(i, j); value = cfd2D.SamplingFromScalarField(SamplingMode::samPointClamp, SelectingMode::SelectDataFromOrigin, &temps);
#define sampleV(i, j, value) Vector2i tempv(i, j); value = *cfd2D.SamplingFromVectorField(SamplingMode::samPointClamp, SelectingMode::SelectDataFromOrigin, &tempv);
#define abs(n) (n>0)?n:-n;
#define ceil(n) (n-(int)n>=0.5f)?(int)n+1:(int)n;
#define floor(n) (int)n;

// Advect function for scalar field
void SGCFD2DSim::ScalarAdvect(double dt)
{
	double    density;
	Vector2d  velocity;

	for (int i=0; i < CELLSU+2; i++)
	{
		for (int j=0; j < CELLSV+2; j++)
		{
			sampleS(i, j, density);
			sampleV(i, j, velocity);
			velocity *= stepsize;
			double u = abs(velocity[0]) + i;
			double v = abs(velocity[1]) + j;
			ceil(u); 
			ceil(v);
			cfd2D.UpdateScalarField(u, v, density);
		}
	}
};


// Advect function for vector field
void SGCFD2DSim::VectorAdvect(double dt)
{
	Vector2d  velocity;

	for (int i=0; i < CELLSU+2; i++)
	{
		for (int j=0; j < CELLSV+2; j++)
		{
			sampleV(i, j, velocity);
			velocity *= stepsize;
			double u = abs(velocity[0]) + i;
			double v = abs(velocity[1]) + j;
			ceil(u); 
			ceil(v);
			cfd2D.UpdateVectorField(u, v, &velocity);
		}
	}
};