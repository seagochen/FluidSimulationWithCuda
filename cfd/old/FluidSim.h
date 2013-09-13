#ifndef _SEAGOSOFT_FLUID_SIM_H_
#define _SEAGOSOFT_FLUID_SIM_H_

class float3
{
public:
	float x, y, z;

	float3(float x_in, float y_in, float z_in) :
		x(x_in), y(y_in), z(z_in) {};

	float3() : x(0), y(0), z(0) {};
};


struct FluidSim
{
	// Index of the current grid cell (i, j, k in [0, gridSize] range)
	float3 cellIndex;

	// Texture coordinates (x, y, z in [0, 1] range) for the current grid cell and its immediate neighbors
	float3 CENTERCELL;
	float3 LEFTCELL;
	float3 RIGHTCELL;
	float3 BOTTOMCELL;
	float3 TOPCELL;
	float3 BACKCELL;
	float3 FRONTCELL;
};

#endif