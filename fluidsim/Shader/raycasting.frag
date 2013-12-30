#version 400

in vec3 raystart;
uniform sampler1D transfer;
uniform sampler2D stopface;
uniform sampler3D volumetric;
uniform float     stride;
uniform vec2      screensize;
out     vec4      pixel;


const vec4  bgColor = vec4 ( 1.f, 1.f, 1.f, 0.f );

void main ()
{
	// We will use the following method to calculate the intersection of ray from far to near 
	// with the screen.
	vec3 raystop = texture ( stopface, gl_FragCoord.st / screensize ).xyz;

	// Before starting ray-casting, we need check the starting point whether is someplace on
	// background. If it is, discarding the process.
	if ( raystop == raystart )
		discard;

	// Ray-casting will start soon, but before that, we need to do some preparation work.
	// We still need the length of radiation from starting to ending, and the direction.
	// After that, do ray-casting need to increasing the sampling depth gradually, therefore
	// we also need to normalize the ray, and set the step size manually.
	// Thus...
	vec3 direction = raystop - raystart;
	float rayLength = length ( direction );
	vec3  dtDir = normalize ( direction ) * stride;
	float dtLen = length ( dtDir );

	// Declare some variables, such as voxel coordination, accumulated color and length
	vec3  voxelCoord  = raystart;
	vec4  accumColor  = vec4 ( 0.f );
	float accumLength = 0.f;

	// Still needs some mark
	float intensityMark = 0.f;
	vec4  colorMark = vec4 ( 0.f );

	// Start ray-casting
	// Circulation as much as possible, in order to interpolate the samples as much as possible.
	for ( int i = 0; i < 1600; i++ ) 
	{
		// Sampled the volumtric at somewhere, and draw back the intensity
		intensityMark = texture ( volumetric, voxelCoord ).x;

		// Look up the color info related to intensity in transfer function
		colorMark = texture ( transfer, intensityMark );

		// Modulate the value by front to back integration
		if ( colorMark.a > 0.f )
		{
			colorMark.a     = 1.f - pow ( 1.f - colorMark.a, stride * 100.f );
			accumColor.rgb += ( 1.f - accumColor.a ) * colorMark.rgb * colorMark.a;
			accumColor.a   += colorMark.a;
		}

		// Increase the depth
		voxelCoord += dtDir;
		accumLength += dtLen;

		if ( accumLength >= rayLength )
		{
			accumColor.rgb = accumColor.rgb * accumColor.a + ( 1 - accumColor.a ) * bgColor.rgb;	
			break;
		}
		if ( accumColor.a > 1.0 )
		{
			accumColor.a = 1.0;
			break;
		}
	}

	// Yield the final color to pixel
	pixel = accumColor;
}