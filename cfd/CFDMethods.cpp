#include "CFDMethods.h"

using namespace sge;

#define OUT_OF_BOUNDARY -1
#define IN_BOUNDARY      1


int CheckBoundary(int u, int v)
{
	if (u<SIMAREA_WIDTH && u>=0 && v<SIMAREA_HEIGHT && v>=0)
		return IN_BOUNDARY;
	else return OUT_OF_BOUNDARY;
};


Vector2d *Velocity2D::SamplePoint(int u, int v)
{
	if (CheckBoundary(u, v) == IN_BOUNDARY)
	{
		Vector2d temp(velocity_u[u], velocity_v[v]);
		return &temp;
	}
	else 
	{
		Vector2d temp(0.f, 0.f);
		return &temp;
	}
};


Vector2d *Velocity2D::Sample(int Sampling, Vector2d *CellIndex)
{
	Vector2d temp = *CellIndex;
	int u = temp[0];
	int v = temp[1];

	if (Sampling == samPointClamp)
	{
		return SamplePoint(u, v);
	}
	
	if (Sampling == samLinear)
	{
		Vector2d up    = *SamplePoint(u, v+1);
		Vector2d down  = *SamplePoint(u, v-1);
		Vector2d left  = *SamplePoint(u-1, v);
		Vector2d right = *SamplePoint(u+1, v);

		Vector2d output = (up + down + left + right) / 4.f;
		return &output;
	}
};

Vector2d *advect(double timestep, FLUIDSIM *in, Velocity2D *velocity)
{
	Vector2d OldPos = in->CenterCell;
	Vector2d CellVelocity = *(velocity->Sample(samPointClamp, &OldPos));
	Vector2d NewPos = CellVelocity * timestep - OldPos;
	
	return velocity->Sample(samLinear, &NewPos);
};