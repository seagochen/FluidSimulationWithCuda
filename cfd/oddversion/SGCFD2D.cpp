#include "CFD2DSim.h"

using namespace sge;

#define MAXSIZEU  SIMAREA_WIDTH  + 2
#define MAXSIZEV  SIMAREA_HEIGHT + 2

#define U(px) px[0]
#define V(px) px[1]
#define W(px) px[2]

#define decode(ptr) int ups, vps; Vector2i px = *ptr; ups = U(px); vps = V(px);


Vector2i *SGCFD2D::SamplePoint(int u, int v)
{
	// Check if out of boundary
	if ( u >= MAXSIZEU ) { u = MAXSIZEU - 1; }
	elif ( u < 0 ) { u = 0; }

	if ( v >= MAXSIZEV ) { v = MAXSIZEV - 1; }
	elif ( v < 0 ) { v = 0; }

	// Return the result
	Vector2i temp(u, v);
	return &temp;
};


double SGCFD2D::SamplingScalarLinear(SelectingMode smode, Vector2i *CellIndex)
{
	// Decode the cell index
	decode(CellIndex);
	

	/// horizontal direction is u
	/// vertical direction is v
	/// Thus, left is u-1, right is u+1, up is v+1, down is v-1
	Vector2i l  = *SamplePoint(ups-1, vps);   // left
	Vector2i r  = *SamplePoint(ups+1, vps);   // right
	Vector2i u  = *SamplePoint(ups, vps+1);   // up
	Vector2i d  = *SamplePoint(ups, vps-1);   // down


	// Sampling data from scalar field
	if (smode == SelectingMode::SelectDataFromOrigin)
	{
		double lv = density_origin[U(l)][V(l)];
		double rv = density_origin[U(r)][V(r)];
		double uv = density_origin[U(u)][V(u)];
		double dv = density_origin[U(d)][V(d)];

		double val = (lv + rv + uv + dv) / 4.f;

		return val;
	}
	elif (smode == SelectingMode::SelectDataFromUpdate)
	{
		double lv = density_update[U(l)][V(l)];
		double rv = density_update[U(r)][V(r)];
		double uv = density_update[U(u)][V(u)];
		double dv = density_update[U(d)][V(d)];

		double val = (lv + rv + uv + dv) / 4.f;

		return val;
	}

	return 0.f;
};


Vector2d *SGCFD2D::SamplingVectorLinear(SelectingMode smode, Vector2i *CellIndex)
{
	// Decode the cell index
	decode(CellIndex);
	

	/// horizontal direction is u
	/// vertical direction is v
	/// Thus, left is u-1, right is u+1, up is v+1, down is v-1
	Vector2i l  = *SamplePoint(ups-1, vps);   // left
	Vector2i r  = *SamplePoint(ups+1, vps);   // right
	Vector2i u  = *SamplePoint(ups, vps+1);   // up
	Vector2i d  = *SamplePoint(ups, vps-1);   // down


	if (smode == SelectingMode::SelectDataFromOrigin)
	{
		Vector2d lv = velocity_origin[U(l)][V(l)];
		Vector2d rv = velocity_origin[U(r)][V(r)];
		Vector2d uv = velocity_origin[U(u)][V(u)];
		Vector2d dv = velocity_origin[U(d)][V(d)];

		Vector2d val = (lv + rv + uv + dv) / 4.f;

		return &val;
	}
	elif (smode == SelectingMode::SelectDataFromUpdate)
	{
		Vector2d lv = velocity_update[U(l)][V(l)];
		Vector2d rv = velocity_update[U(r)][V(r)];
		Vector2d uv = velocity_update[U(u)][V(u)];
		Vector2d dv = velocity_update[U(d)][V(d)];

		Vector2d val = (lv + rv + uv + dv) / 4.f;

		return &val;
	}

	Vector2d temp(0, 0);
	return &temp;
};


Vector2d *SGCFD2D::SamplingVectorPointer(SelectingMode smode, Vector2i *CellIndex)
{
	// Decode the cell index
	decode(CellIndex);

	if (smode == SelectingMode::SelectDataFromOrigin)
	{
		Vector2d val = velocity_origin[ups][vps];
		return &val;
	}
	elif (smode == SelectingMode::SelectDataFromUpdate)
	{
		Vector2d val = velocity_update[ups][vps];
		return &val;
	}

	Vector2d temp(0, 0);
	return &temp;
};


double SGCFD2D::SamplingScalarPointer(SelectingMode smode, Vector2i *CellIndex)
{
	// Decode the cell index
	decode(CellIndex);

	if (smode == SelectingMode::SelectDataFromOrigin)
	{
		double val = density_origin[ups][vps];
		return val;
	}
	elif (smode == SelectingMode::SelectDataFromUpdate)
	{
		double val = density_update[ups][vps];
		return val;
	}

	return 0.f;
};


// Sampling method for vector field
Vector2d *SGCFD2D::SamplingFromVectorField(SamplingMode pmode, SelectingMode smode, Vector2i *CellIndex)
{
	// Sampling method
	if (pmode == SamplingMode::samLinear)
	{
		return SamplingVectorLinear(smode, CellIndex);
	}
	elif (pmode == SamplingMode::samPointClamp)
	{
		return SamplingVectorPointer(smode, CellIndex);
	}

	Vector2d temp(0, 0);
	return &temp;
};


// Sampling method for scalar field
double SGCFD2D::SamplingFromScalarField(SamplingMode pmode, SelectingMode smode, Vector2i *CellIndex)
{
	// Sampling method
	if ( pmode == SamplingMode::samLinear )
	{
		return SamplingScalarLinear(smode, CellIndex);
	}
	elif ( pmode == SamplingMode::samPointClamp )
	{
		return SamplingScalarPointer(smode, CellIndex);
	}

	return 0.f;
};


// Yield value to vector field
void SGCFD2D::YieldValueToVectorField(SelectingMode smode, Vector2d *value, Vector2i *CellIndex)
{
	// Decode the cell index
	decode(CellIndex);

	if (smode == SelectingMode::YieldDataToOrigin)
	{
		velocity_origin[ups][vps] = *value;
	}
	elif (smode == SelectingMode::YieldDataToUpdate)
	{
		velocity_update[ups][vps] = *value;
	}
};


// Yield value to scalar field
void SGCFD2D::YieldValueToScalarField(SelectingMode smode, double value, Vector2i *CellIndex)
{
	// Decode the cell index
	decode(CellIndex);
	
	if (smode == SelectingMode::YieldDataToOrigin)
	{
		density_origin[ups][vps] = value;
	}
	elif (smode == SelectingMode::YieldDataToUpdate)
	{
		density_update[ups][vps] = value;
	}
};


// Swap value of vector fields
void SGCFD2D::SwapVectorField(SwappingMode wmode)
{
	if (wmode == SwappingMode::SwapDataFromOriginToUpdate)
	{
		for (int i=0; i < CELLSU + 2; i++)
		{
			for (int j=0; j < CELLSV + 2; j++)
			{
				Vector2d temp = velocity_update[i][j];
				velocity_update[i][j] = velocity_origin[i][j];
				velocity_origin[i][j] = temp;
			}
		}
	}
	elif (wmode == SwappingMode::SwapDataFromUpdateToOrigin)
	{
		for (int i=0; i < CELLSU + 2; i++)
		{
			for (int j=0; j < CELLSV + 2; j++)
			{
				Vector2d temp = velocity_origin[i][j];
				velocity_origin[i][j] = velocity_update[i][j];
				velocity_update[i][j] = temp;
			}
		}
	}
};


// Swap value of scalar fields
void SGCFD2D::SwapScalarField(SwappingMode wmode)
{
	if (wmode == SwappingMode::SwapDataFromOriginToUpdate)
	{
		for (int i=0; i < CELLSU + 2; i++)
		{
			for (int j=0; j < CELLSV + 2; j++)
			{
				double temp = density_update[i][j];
				density_update[i][j] = density_origin[i][j];
				density_origin[i][j] = temp;
			}
		}
	}
	elif (wmode == SwappingMode::SwapDataFromUpdateToOrigin)
	{
		for (int i=0; i < CELLSU + 2; i++)
		{
			for (int j=0; j < CELLSV + 2; j++)
			{
				double temp = density_origin[i][j];
				density_origin[i][j] = density_update[i][j];
				density_update[i][j] = temp;
			}
		}
	}
};


// Sampling from last updated vector field
Vector2d *SGCFD2D::SamplingFromLastVectorField(SamplingMode pmode, Vector2i *CellIndex)
{
	return SamplingFromVectorField(pmode, SelectingMode::SelectDataFromUpdate, CellIndex);
};


// Sampling from last updated scalar field
double SGCFD2D::SamplingFromLastScalarField(SamplingMode pmode, Vector2i *CellIndex)
{
	return SamplingFromScalarField(pmode, SelectingMode::SelectDataFromUpdate, CellIndex);
};


// Update vector field
void SGCFD2D::UpdateVectorField(int u, int v, Vector2d *vel_in)
{
	Vector2i temp(u, v);
	YieldValueToVectorField(SelectingMode::YieldDataToUpdate, vel_in, &temp);
};


// Update Scalar field
void SGCFD2D::UpdateScalarField(int u, int v, double vel_in)
{
	Vector2i temp(u, v);
	YieldValueToScalarField(SelectingMode::YieldDataToUpdate, vel_in, &temp);	
};