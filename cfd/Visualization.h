#ifndef _SEAGO_VISUALIZATION_H_
#define _SEAGO_VISUALIZATION_H_

#include "Header.h"

#define OK      0
#define FAIL    -1

#define BYTES_PER_TEXEL 3

namespace seago
{	
	typedef class glDisplay
	{
	private:
		typedef struct _mouse
		{
			int preX, preY;
			bool isLeftHold, isRightHold;
		}Mouse;

		typedef struct _fps
		{
			// Variables used to calculate frames per second:
			DWORD dwFrames;
			DWORD dwCurrentTime;
			DWORD dwLastUpdateTime;
			DWORD dwElapsedTime;
		} FPS; // Frames per second

	public:

		typedef struct _windowSets
		{
			std::string   title;
			float  field_of_view_angle;
			float  z_near, z_far;
			int width, height;
		} window;
		
		typedef struct _volumeData
		{
			unsigned char *data;
			unsigned textureID;
			int width, height, depth;
		} volume;

		typedef struct _viewMatrix
		{
			float rotateX, rotateY;
			float nearZ;
			float eyeX, eyeY, eyeZ;
			float radius; // 球坐标，默认设置为2.f
		} view;
		
	private:
		_mouse      m_mouse;
		_fps        m_fps;
		_volumeData m_volume;
		_windowSets m_window;
		_viewMatrix m_view;
		
	private:
		// 声明与3D纹理操作有关的函数，而这些特性只在更高（通常为3.0及以上）的版本中得到支持
		// 在Windows平台中，只实现了对OpenGL1.1的支持，因此为了使用这些高级特性，所以还需要
		// 获取函数的扩展地址，这些操作只在loadFunc中实现。
		PFNGLTEXIMAGE3DPROC    glTexImage3D;
		PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D;

	public:
		// 当窗口大小发生改动时
		void sgResizeScreen(int width, int height);
		// 显示函数
		void sgDisplay(void);
		// Keyboard function
		//void sgKeyboard(unsigned char key, int mousePositionX, int mousePositionY);
		void sgKeyboard(sge::SGKEYS);
		// Mouse function
		void sgMouse(int button,int state,int x,int y);
		// 加载Volume
		int sgLoadVolumeData(_volumeData const *data_in);
		// 设置窗口参数
		void sgSetWindowParam(_windowSets const *win_in);
		// 获取窗口参数
		inline _windowSets *sgGetWindowParam( ){ return &m_window; };
		// 初始化
		void sgInit(int width, int height);
		// 计算offset, 2-D
		int TEXEL2(int s, int t) { return  BYTES_PER_TEXEL * (s * m_volume.width + t); };
		// 计算offset, 3-D
		int TEXEL3(int s, int t,  int r) { return  TEXEL2(s, t) + LAYER(r); };
		// 鼠标运动状态
		void sgMotion(int x,int y);

	private:
		// 获取有关OpenGL函数扩展地址，这些操作只需要在Windows平台中
		int loadFunc(void);
		// 创建并绑定3D纹理
		int bindTexutre(void);
		// 绘制代理几何
		void agentBox(void);
		// 初始化FPS有关操作及数据
		void initFPS(void), initViewMatrix(void);
		// 计算layer
		int LAYER(int r) {return m_volume.width * m_volume.height * r * BYTES_PER_TEXEL;};
		// FPS function
		void sgCountFPS(void);
	} Visualization;
};

#endif