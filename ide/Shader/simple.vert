// for raycasting
#version 400

uniform float t;  // Time
uniform mat4 MVP; // Combined modelview and projection matrices
in vec4 pos;      // Particle position
in vec4 vel;      // Particle velocity

// Gravitational acceleration
const vec4 g = vec4 (0.0, -9.8, 0.0);


void main()
{
	vec4 position = pos;

	/// <latex> P^{'} = P_{o} + t^{2}\cdot g </latex>
	position += t * vel + t * t * g;

	// Copy data into device register
	gl_Position = MVP * position;
}
