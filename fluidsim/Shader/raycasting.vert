#version 400

in vec3 vertices;
uniform mat4 mvp;
out vec3 raystart;

void main()
{
	raystart = vertices;
    gl_Position = mvp * vec4 ( vertices, 1.0 );
}
