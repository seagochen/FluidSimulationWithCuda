#pragma comment(lib, "freeglut.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "sgeutil.lib")

#include "Header.h"
#include "Visualization.h"

#define DISPLAY_WIDTH  800
#define DISPLAY_HEIGHT 600

using namespace seago;
using namespace sge;

Visualization visual;
Visualization::window win;
MainActivity *myact;

extern DWORD simulation (LPVOID lpdwThreadParam );

// ֮��������д������Ϊ�����߳�ʱ������ĺ�����ַ�����и����
// ������ַ����һ���£�������Ҫ��������һ�����������Ӷ��ܿ������Ĵ���
void init(){ visual.sgInit(win.width, win.height); };
void display(){ visual.sgDisplay(); };
void reshape(int width, int height) { visual.sgResizeScreen(width, height); };
//void keyboard(unsigned char key, int mousePositionX, int mousePositionY){visual.sgKeyboard(key, mousePositionX, mousePositionY);};
void keyboard(sge::SGKEYS keys) { visual.sgKeyboard(keys); };
void mouse(int button,int state,int x,int y) { visual.sgMouse(button, state, x, y); }
void motion(int x, int y) { visual.sgMotion(x,y); };


void setWindow()
{
	// Set window values
	win.title = "Fluid Simulation Program, v2.00.00 alpha";
	win.field_of_view_angle = 45;
	win.z_near = 1.0f;
	win.z_far = 500.0f;
	win.width = DISPLAY_WIDTH;
	win.height = DISPLAY_HEIGHT;

	visual.sgSetWindowParam(&win);
}


/* ����������ͬ�����̣߳�һ��������㲢����volume data����һ���������FPS */
void subThreads()
{
	DWORD dwThreadId;
	
	// ��������������߳�
	if ( CreateThread(NULL, //Choose default security
		0, //Default stack size
		(LPTHREAD_START_ROUTINE)&simulation, //Routine to execute
		0, //Thread parameter
		0, //Immediately run the thread
		&dwThreadId ) == NULL) //Thread Id	
		{
			printf("Error in line %d: Couldn't creat sub-thread. Aborting.\n", __LINE__);
		}
}


int main()
{
	glewInit();
	
	// Initialize window parameters
	setWindow();

	// Creates sub threads
	subThreads();
	
	// Set window size
	myact = new MainActivity(win.width, win.height);
	

	// Register functions
	myact->RegisterCreateFunc(init);
	myact->RegisterDisplayFunc(display);
	myact->RegisterKeyDownFunc(keyboard);
	// TODO
	// Register mouse function

	// Setup MFC window
	myact->SetupRoutine(win.title.c_str());

	// Mainloop
	myact->MainLoop();

	SAFE_DELT_PTR(myact);
	return 0;
}