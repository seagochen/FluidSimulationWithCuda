#version 400

in vec3 fragColor;
out vec4 pixel;


void main()
{
	pixel = vec4 ( fragColor, 1.0 );
}
