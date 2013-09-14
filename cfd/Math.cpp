#include "Math.h"

using namespace sge;

Vector2f *ToVector(float2 const *in)
{
	Vector2f temp(in->x, in->y);
	return &temp;
};

Vector3f *ToVector(float3 const *in)
{
	Vector3f temp(in->x, in->y, in->z);
	return &temp;
};

Vector4f *ToVector(float4 const *in)
{
	Vector4f temp(in->x, in->y, in->z, in->w);
	return &temp;
};