#version 400

in vec3 VerPos;

uniform mat4 MVP;

out vec3 EntryPoint;
out vec4 ExitPointCoord;


void main()
{
//    EntryPoint = VerClr;
	EntryPoint = VerPos;
    gl_Position = MVP * vec4(VerPos,1.0);
    // ExitPointCoord 输入到fragment shader 的过程中经过rasterization， interpolation, assembly primitive
    ExitPointCoord = gl_Position;  
}
