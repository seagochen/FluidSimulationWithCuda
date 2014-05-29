#include "main3D.h"

void init()
{
    xRot=30;
    yRot=-40;

    xTrans=0;
    yTrans=0;
    zTrans=-1.0;

    N=32;
    dt=0.1f;
    diff=0.0f;
    visc=0.0f;
    force=90.0f;
    source=800.0f;

    int count;
    int size=(N+2)*(N+2)*(N+2);

    velocity_u=(float *)malloc(size*sizeof(float));
    velocity_v=(float *)malloc(size*sizeof(float));
    velocity_w=(float *)malloc(size*sizeof(float));

    velocity_u_prev=(float *)malloc(size*sizeof(float));
    velocity_v_prev=(float *)malloc(size*sizeof(float));
    velocity_w_prev=(float *)malloc(size*sizeof(float));

    density=(float *)malloc(size*sizeof(float));   
    density_prev=(float *)malloc(size*sizeof(float));

    for(count=0; count<size; count++)
    {
        velocity_u[count]=0;
        velocity_v[count]=0;
        velocity_w[count]=0;

        velocity_u_prev[count]=0;
        velocity_v_prev[count]=0;
        velocity_w_prev[count]=0;

        density[count]=0;   
        density_prev[count]=0;
    }
}

void get_source(float *density, float *velocity_u, float *velocity_v, float *velocity_w)
{
    int count_x;
    int count_y;
    int count_z;
    int count;
    int size=(N+2)*(N+2)*(N+2);

    for(count=0; count<size; count++)
    {
        velocity_u[count]=0;
        velocity_v[count]=0;
        velocity_w[count]=0;
        density[count]=0;
    }

    if(add_velocity_x == 1)
    {
        count_x=2;
        count_y=N/2;
        count_z=N/2;

        velocity_u[INDEX(count_x, count_y, count_z)]=force*10;
        add_velocity_x=0;
    }   

    if(add_velocity_y == 1)
    {
        count_x=N/2;
        count_y=2;
        count_z=N/2;

        velocity_v[INDEX(count_x, count_y, count_z)]=force*10;
        add_velocity_y=0;
    }   

    if(add_velocity_z == 1)
    {
        count_x=N/2;
        count_y=N/2;
        count_z=2;

        velocity_w[INDEX(count_x, count_y, count_z)]=force*10;    
        add_velocity_z=0;
    }   

    if(add_density)
    {
        count_x=N/2,
        count_y=N/4*3;
        count_z=N/2;

        density[INDEX(count_x, count_y, count_z)]=source;
        add_density=0;
    }

    for(count_x=1; count_x<=N; count_x++)
    {
        for(count_y=1; count_y<=N; count_y++)
        {
            for(count_z=1; count_z<=N; count_z++)
            {
                if(density[INDEX(count_x, count_y, count_z)] > 0)
                {
                    velocity_v[INDEX(count_x, count_y, count_z)]=velocity_v[INDEX(count_x, count_y, count_z)]-density[INDEX(count_x, count_y, count_z)]*(0.2f*(count_y+1)/N);
                }
            }
        }
    }
   
    return;
}

void draw_box()
{
    glLineWidth(1.0f);

    glBegin(GL_LINES);   
        glColor3f(1.0f, 0.0f, 0.0f);

        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(1.3f, 0.0f, 0.0f);

        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 1.3f, 0.0f);

        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 1.3f);

        glVertex3f(1.3f, 0.0f, 0.0f);
        glVertex3f(1.3f, 1.3f, 0.0f);

        glVertex3f(1.3f, 1.3f, 0.0f);
        glVertex3f(0.0f, 1.3f, 0.0f);

        glVertex3f(0.0f, 1.3f, 1.3f);
        glVertex3f(0.0f, 0.0f, 1.3f);

        glVertex3f(0.0f, 1.3f, 1.3f);
        glVertex3f(0.0f, 1.3f, 0.0f);

        glVertex3f(1.3f, 0.0f, 0.0f);
        glVertex3f(1.3f, 0.0f, 1.3f);

        glVertex3f(0.0f, 0.0f, 1.3f);
        glVertex3f(1.3f, 0.0f, 1.3f);

        glVertex3f(1.3f, 1.3f, 0.0f);
        glVertex3f(1.3f, 1.3f, 1.3f);

        glVertex3f(1.3f, 1.3f, 1.3f);
        glVertex3f(1.3f, 0.0f, 1.3f);

        glVertex3f(0.0f, 1.3f, 1.3f);
        glVertex3f(1.3f, 1.3f, 1.3f);
    glEnd();
}

void draw_density()
{
    int count_x;
    int count_y;
    int count_z;

    float pos_x;
    float pos_y;
    float pos_z;

    float h;
   
    float d000;
    float d010;
    float d100;
    float d110;
    float d001;
    float d011;
    float d101;
    float d111;
   
    h=1.3f/N;

    glBegin(GL_QUADS);
        for(count_x=0; count_x<=N; count_x++)
        {
            pos_x=(count_x-0.5f)*h;

            for(count_y=0; count_y<=N; count_y++)
            {
                pos_y=(count_y-0.5f)*h;

                for(count_z=0; count_z<=N; count_z++)
                {
                    pos_z=(count_z-0.5f)*h;

                    d000=density[INDEX(count_x, count_y, count_z)];
                    d010=density[INDEX(count_x, count_y+1, count_z)];
                    d100=density[INDEX(count_x+1, count_y, count_z)];
                    d110=density[INDEX(count_x+1, count_y+1, count_z)];

                    d001=density[INDEX(count_x, count_y, count_z+1)];
                    d011=density[INDEX(count_x, count_y+1, count_z+1)];
                    d101=density[INDEX(count_x+1, count_y, count_z+1)];
                    d111=density[INDEX(count_x+1, count_y+1, count_z+1)];                   

                    glColor4f(d111, d111, d111, alpha);
                    glVertex3f(pos_x+h, pos_y+h, pos_z+h);

                    glColor4f(d011, d011, d011, alpha);
                    glVertex3f(pos_x, pos_y+h, pos_z+h);

                    glColor4f(d001, d001, d001, alpha);
                    glVertex3f(pos_x, pos_y, pos_z+h);

                    glColor4f(d101, d101, d101, alpha);
                    glVertex3f(pos_x+h, pos_y, pos_z+h);

                    glColor4f(d110, d110, d110, alpha);
                    glVertex3f(pos_x+h, pos_y+h, pos_z);

                    glColor4f(d111, d111, d111, alpha);
                    glVertex3f(pos_x+h, pos_y+h, pos_z+h);

                    glColor4f(d101, d101, d101, alpha);
                    glVertex3f(pos_x+h, pos_y, pos_z+h);

                    glColor4f(d100, d100, d100, alpha);
                    glVertex3f(pos_x+h, pos_y, pos_z);

                    glColor4f(d010, d010, d010, alpha);
                    glVertex3f(pos_x, pos_y+h, pos_z);

                    glColor4f(d110, d110, d110, alpha);
                    glVertex3f(pos_x+h, pos_y+h, pos_z);

                    glColor4f(d100, d100, d100, alpha);
                    glVertex3f(pos_x+h, pos_y, pos_z);

                    glColor4f(d000, d000, d000, alpha); 
                    glVertex3f(pos_x, pos_y, pos_z);

                    glColor4f(d011, d011, d011, alpha);
                    glVertex3f(pos_x, pos_y+h, pos_z+h);

                    glColor4f(d010, d010, d010, alpha); 
                    glVertex3f(pos_x, pos_y+h, pos_z);

                    glColor4f(d000, d000, d000, alpha);
                    glVertex3f(pos_x, pos_y, pos_z);

                    glColor4f(d001, d001, d001, alpha);
                    glVertex3f(pos_x, pos_y, pos_z+h);

                    glColor4f(d100, d100, d100, alpha);
                    glVertex3f(pos_x+h, pos_y, pos_z);

                    glColor4f(d000, d000, d000, alpha);
                    glVertex3f(pos_x, pos_y, pos_z);

                    glColor4f(d001, d001, d001, alpha);
                    glVertex3f(pos_x, pos_y, pos_z+h);

                    glColor4f(d101, d101, d101, alpha);
                    glVertex3f(pos_x+h, pos_y, pos_z+h);

                    glColor4f(d110, d110, d110, alpha);
                    glVertex3f(pos_x+h, pos_y+h, pos_z);

                    glColor4f(d010, d010, d010, alpha);
                    glVertex3f(pos_x, pos_y+h, pos_z);

                    glColor4f(d011, d011, d011, alpha);
                    glVertex3f(pos_x, pos_y+h, pos_z+h);

                    glColor4f(d111, d111, d111, alpha);
                    glVertex3f(pos_x+h, pos_y+h, pos_z+h);               
                }
            }
        }
    glEnd ();
}

void draw_velocity()
{
    int count_x;
    int count_y;
    int count_z;

    float pos_x;
    float pos_y;
    float pos_z;

    float h;
    h=1.3f/N;

    glColor3f(1.0, 1.0, 1.0);

    for(count_x=0; count_x<=N; count_x++)
    {
        pos_x=(count_x-0.5f)*h;

        for(count_y=0; count_y<=N; count_y++)
        {
            pos_y=(count_y-0.5f)*h;

            for(count_z=0; count_z<=N; count_z++)
            {
                pos_z=(count_z-0.5f)*h;

                glBegin(GL_LINES);
                    glVertex3f(pos_x, pos_y, pos_z);
                    glVertex3f(pos_x+velocity_u[INDEX(count_x, count_y, count_z)]/2, pos_y+velocity_v[INDEX(count_x, count_y, count_z)]/2, pos_z+velocity_w[INDEX(count_x, count_y, count_z)]/2);
                glEnd();
            }
        }
    }
}

void display_func()
{
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0);

    glClear(GL_COLOR_BUFFER_BIT);

    glPushMatrix();
    glTranslatef(xTrans, yTrans, zTrans);
   
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    if(velocity_or_density == 0)
    {
        draw_density();
    }

    if(velocity_or_density == 1)
    {
        draw_velocity();
    }

    draw_box();

    glPopMatrix();

    glutSwapBuffers();
}

void reshape_func(GLint width, GLint height)
{
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(45.0, (float)width/height, 0.001, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);
}

void SpecialKeys(int key, int x, int y)
{
    if(key == GLUT_KEY_UP)
    {
        xRot -= 5.0f;

        xRot=(GLfloat)((const int)15);
        yRot=(GLfloat)((const int)15);
    }

    if(key == GLUT_KEY_DOWN)
    {
        xRot += 5.0f;

        xRot=(GLfloat)((const int)15);
        yRot=(GLfloat)((const int)15);
    }

    if(key == GLUT_KEY_LEFT)
    {
        yRot += 5.0f;

        xRot=(GLfloat)((const int)15);
        yRot=(GLfloat)((const int)15);
    }

    if(key == GLUT_KEY_RIGHT)
    {
        yRot -= 5.0f;

        xRot=(GLfloat)((const int)15);
        yRot=(GLfloat)((const int)15);
    }
               
    glutPostRedisplay();
}

void processKeyBoard(unsigned char key, int x, int y)
{
    if(key == 'i' || key == 'I')
    {   
        add_density=1;
    }
    if(key == 'x' || key == 'X')
    {
        add_velocity_x=1;
    }   

    if(key == 'y' || key == 'Y')
    {
        add_velocity_y=1;
    }

    if(key == 'z' || key == 'Z')
    {
        add_velocity_z=1;
    }

    if(key == 'w' || key == 'W')
    {
        zTrans=zTrans+0.1;
    }

    if(key == 's' || key == 'S')
    {
        zTrans=zTrans-0.1;
    }

    if(key == 'a' || key == 'A')
    {
        xTrans=xTrans-0.1;
    }

    if(key == 'd' || key == 'D')
    {
        xTrans=xTrans+0.1;
    }

    if(key == 'q' || key == 'Q')
    {
        yTrans=yTrans+0.1;
    }

    if(key == 'e' || key == 'E')
    {
        yTrans=yTrans-0.1;
    }

    if(key == 'c' || key == 'C')
    {
        int count;
        int size=(N+2)*(N+2)*(N+2);
        for(count=0; count<size; count++)
        {
            velocity_u[count]=0;
            velocity_v[count]=0;
            velocity_w[count]=0;

            velocity_u_prev[count]=0;
            velocity_v_prev[count]=0;
            velocity_w_prev[count]=0;

            density[count]=0;   
            density_prev[count]=0;
        }
    }

    if(key == 'v' || key == 'V')
    {
        velocity_or_density=1-velocity_or_density;
    }
}

void idle_func()
{
    get_source( density_prev, velocity_u_prev, velocity_v_prev, velocity_w_prev );
    get_velocity(N, velocity_u, velocity_v, velocity_w, velocity_u_prev, velocity_v_prev,velocity_w_prev, visc, dt );
    get_density(N, density, density_prev, velocity_u, velocity_v, velocity_w, diff, dt );

    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    init();

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowPosition(0, 0);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Stable Fluid 3D");

    glutDisplayFunc(display_func);
    glutReshapeFunc(reshape_func);
    glutSpecialFunc(SpecialKeys);
    glutKeyboardFunc(processKeyBoard);
    glutIdleFunc(idle_func);

    glutMainLoop();

    return 0;
}
