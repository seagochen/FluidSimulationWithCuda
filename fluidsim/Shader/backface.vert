// for raycasting
#version 400

//layout(location = 0) in vec3 VerPos;
//layout(location = 1) in vec3 VerClr;

in vec3 VerPos;

uniform mat4 MVP;

out vec3 Color;


void main()
{
//    Color = VerClr;
	Color = VerPos;
    gl_Position = MVP * vec4(VerPos, 1.0);
}
