#include "Header.h"

using namespace sge;

extern MainActivity *myact;

LRESULT CALLBACK MainWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	myact->WindowProc(hWnd, uMsg, wParam, lParam);
	return DefWindowProc(hWnd, uMsg, wParam, lParam);
};


#define WINDOW_WIDTH   800	
#define WINDOW_HEIGHT  600
#define WIN_PIXEL_BITS 32

int MainActivity::SetupRoutine(const LPCSTR title)
{
	WNDCLASSEX windowClass;     // window class
	DWORD      dwExStyle;       // Window Extended Style
	DWORD      dwStyle;         // Window Style
	
	HINSTANCE hInstance = GetModuleHandle(NULL);
	
	// fill out the window class structure
	windowClass.cbSize          = sizeof(WNDCLASSEX);
	windowClass.style           = CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc     = MainWindowProc;
	windowClass.cbClsExtra      = 0;
	windowClass.cbWndExtra      = 0;
	windowClass.hInstance       = hInstance;
	windowClass.hIcon           = LoadIcon(NULL, IDI_APPLICATION);  // default icon
	windowClass.hCursor         = LoadCursor(NULL, IDC_ARROW);      // default arrow
	windowClass.hbrBackground   = NULL;                             // don't need background
	windowClass.lpszMenuName    = NULL;                             // no menu
	windowClass.lpszClassName   = "GLClass";
	windowClass.hIconSm         = LoadIcon(NULL, IDI_WINLOGO);      // windows logo small icon

	// register the windows class
	if (!RegisterClassEx(&windowClass))
		return SG_FAIL;
	
	// fullscreen?
	if ( isFullScreen() )
	{
		// device mode
		DEVMODE dmScreenSettings;
		memset(&dmScreenSettings,0,sizeof(dmScreenSettings));
		dmScreenSettings.dmSize = sizeof(dmScreenSettings); 
		// screen width
		dmScreenSettings.dmPelsWidth = WINDOW_WIDTH;
		// screen height
		dmScreenSettings.dmPelsHeight = WINDOW_HEIGHT;
		// bits per pixel
		dmScreenSettings.dmBitsPerPel = WIN_PIXEL_BITS;
		dmScreenSettings.dmFields=DM_BITSPERPEL|DM_PELSWIDTH|DM_PELSHEIGHT;

		if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN) != DISP_CHANGE_SUCCESSFUL)
		{
			// setting display mode failed, switch to windowed
			MessageBox(NULL, "Display mode failed", NULL, MB_OK);
			setFullScreen( FALSE );
		}
	}

	// Are We Still In Fullscreen Mode?
	if ( isFullScreen() )
	{
		// Window Extended Style
		dwExStyle=WS_EX_APPWINDOW;
		// Windows Style
		dwStyle=WS_POPUP;
		// Hide Mouse Pointer
		ShowCursor(FALSE);
	}
	else
	{
		// Window Extended Style
		dwExStyle=WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
		// Windows Style
		dwStyle=WS_OVERLAPPEDWINDOW;
	}
	
	// Adjust Window To True Requested Size
	RECT *rect = getWindowRect();
	AdjustWindowRectEx(rect, dwStyle, FALSE, dwExStyle);
	
	// class registered, so now create our window
	HWND hwnd = CreateWindowEx(NULL,       // extended style
		"GLClass",                    // class name
		title,                        // app name
		dwStyle | WS_CLIPCHILDREN |
		WS_CLIPSIBLINGS,
		0, 0,                         // x,y coordinate
		rect->right - rect->left,
		rect->bottom - rect->top,     // width, height
		NULL,                         // handle to parent
		NULL,                         // handle to menu
		hInstance,                    // application instance
		NULL);                        // no extra params
	
	setHWND( hwnd );
	setHDC( GetDC(hwnd) );
	setHINSTANCE( hInstance );
	
	return SG_OK;
};


int32 MainActivity::WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{	
	HDC hDC   = getHDC();
	HGLRC hRC = getHGLRC();

	// dispatch messages
	switch (uMsg)
	{   
	case WM_CREATE:     // window creation
		OnCreate(hWnd, &hDC, &hRC);
		SG_FuncsHolder.hCreateFunc();
		break;
	
	case WM_DESTROY:    // destroy window
		OnDestroy(hDC, hRC);
		break;
	
	case WM_QUIT:       // quit window
		OnDestroy(hDC, hRC);
		break;

	case WM_CLOSE:      // close window
		OnDestroy(hDC, hRC);
		break;

	case WM_SIZE:
		OnReshapeWindow(lParam, SG_FuncsHolder.hReshapeFunc);
		break;
		
	case WM_ACTIVATEAPP: // activate app
		break;
		
	case WM_PAINT:       // paint
		OnPaint(hWnd);
		break;
		
	case WM_LBUTTONDOWN: // left mouse button down
		break;
		
	case WM_RBUTTONDOWN: // right mouse button down
		break;
		
	case WM_MOUSEMOVE:   // mouse movement
		break;
	
	case WM_LBUTTONUP:   // left mouse button up
		break;
		
	case WM_RBUTTONUP:   // right mouse button up
		break;

	case WM_KEYUP:       // key up
		OnKeyUp(NULL);
		break;
		
	case WM_KEYDOWN:     // key down
		switch(wParam)
		{
		case VK_ESCAPE:
			PostQuitMessage(0);
			break;

		default: break;
		}
		
		OnKeyDown(NULL);
		break;
	
	default:
		break;
	}
	
	return SG_OK;
};


int32 MainActivity::MainLoop()
{	
	HWND hwnd = getHWND();
	// check if window creation failed (hwnd would equal NULL)
	if ( hwnd == NULL )
		pterror("create window failed");
	
	// Initialize OpenGL and display the window
	ShowWindow(hwnd, SW_SHOW);          // display the window
	UpdateWindow(hwnd);                 // update the window
	
	// Enter the message loop
	MSG msg; 
	HDC hDC = getHDC();

	ZeroMemory(&msg, sizeof(MSG));
	
	while( msg.message != WM_QUIT )
	{
		if ( PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) )
		{
			TranslateMessage (&msg);
			DispatchMessage (&msg);
		}
		else
		{
			OnDisplay(SG_FuncsHolder.hDisplayFunc);
			SwapBuffers(hDC);
		}
	}
	
	
	if ( isFullScreen() )
	{
		ChangeDisplaySettings(NULL,0);          // If So Switch Back To The Desktop
		ShowCursor(TRUE);                       // Show Mouse Pointer
	}
	return SG_OK;
};