#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <float.h>
#include <GL/glut.h>
#include "MySolver3D.h"

#define INDEX(i,j,k) ((i)+(N+2)*(j)+(N+2)*(N+2)*k)
#define SWAP(x0, x) {float *temp = x0; x0 = x; x = temp;}

int window_width=600;
int window_height=600;

int velocity_or_density=0;

GLfloat xRot;
GLfloat yRot;

GLfloat xTrans;
GLfloat yTrans;
GLfloat zTrans;

int N;
float dt;
float diff;
float visc;
float force;
float source;

float *velocity_u;
float *velocity_v;
float *velocity_w;
float *velocity_u_prev;
float *velocity_v_prev;
float *velocity_w_prev;

float *density;
float *density_prev;

float alpha=0.05;

int add_density;
int add_velocity_x=0;
int add_velocity_y=0;
int add_velocity_z=0;
