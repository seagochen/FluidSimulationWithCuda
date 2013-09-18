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


Vector2d *Texture2D::SamplePoint(int index_u, int index_v)
{
	if (CheckBoundary(index_u, index_v) == IN_BOUNDARY)
	{
		Vector2d temp(u[index_u], v[index_v]);
		return &temp;
	}
	else 
	{
		Vector2d temp(0.f, 0.f);
		return &temp;
	}
};


double Texture2D::SampleData(int Sampling, Vector2d *CellIndex)
{
	Vector2d temp = *CellIndex;
	int u = temp[0];
	int v = temp[1];

	if (Sampling == samPointClamp)
	{
		return s[u][v];
	}
};


Vector2d *Texture2D::Sample(int Sampling, Vector2d *CellIndex)
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


Vector2d *Advect(double timestep, FLUIDSIM *in, Texture2D *velocity)
{
	Vector2d CellVelocity = *(velocity->Sample(samPointClamp, &(in->CenterCell)));
	Vector2d NewPos = CellVelocity * timestep - in->CenterCell;
	
	return velocity->Sample(samLinear, &NewPos);
};


double Divergence(FLUIDSIM *in, Texture2D *velocity)
{
	// Get velocity values from neighboring cells
	Vector2d fieldL = *(velocity->Sample(samPointClamp, &(in->LeftCell)));
	Vector2d fieldR = *(velocity->Sample(samPointClamp, &(in->RightCell)));
	Vector2d fieldB = *(velocity->Sample(samPointClamp, &(in->DownCell)));
	Vector2d fieldU = *(velocity->Sample(samPointClamp, &(in->UpCell)));

	// Compute the velocity's divergence using central differences
	double divergence = 0.5 * ((fieldR[0] - fieldL[0]) + (fieldU[1] - fieldB[1]));

	return divergence;
};


double Jacobi(FLUIDSIM *in, Texture2D *pressure, Texture2D *divergence)
{
	// Get the divergence at the current cell
	double dC = divergence->SampleData(samPointClamp, &(in->CellIndex));

	// Get pressure values from neighboring cells
	double pL = pressure->SampleData(samPointClamp, &(in->LeftCell));
	double pR = pressure->SampleData(samPointClamp, &(in->RightCell));
	double pU = pressure->SampleData(samPointClamp, &(in->UpCell));
	double pB = pressure->SampleData(samPointClamp, &(in->DownCell));

	// Compute the new pressure value for the center cell
	return (pL + pR + pU + pB - dC) / 4.f;
};

Vector2d *Project(FLUIDSIM *in, Texture2D *pressure, Texture2D *velocity)
{
	// Compute the gradient of pressure at the current cell by taking central differences 
	// of neighboring pressure values
	double pL = pressure->SampleData(samPointClamp, &(in->LeftCell));
	double pR = pressure->SampleData(samPointClamp, &(in->RightCell));
	double pU = pressure->SampleData(samPointClamp, &(in->UpCell));
	double pB = pressure->SampleData(samPointClamp, &(in->DownCell));

	Vector2d gradP(pR - pL, pU - pB);
	gradP *= 0.5f;

	// Project the velocity onto its divergence-free component by subtracting the gradient of pressure
	Vector2d vOld = *velocity->Sample(samPointClamp, &(in->CellIndex));
	Vector2d vNew = vOld - gradP;

	return &vNew;
};