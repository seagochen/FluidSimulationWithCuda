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
			float radius; // �����꣬Ĭ������Ϊ2.f
		} view;
		
	private:
		_mouse      m_mouse;
		_fps        m_fps;
		_volumeData m_volume;
		_windowSets m_window;
		_viewMatrix m_view;
		
	private:
		// ������3D��������йصĺ���������Щ����ֻ�ڸ��ߣ�ͨ��Ϊ3.0�����ϣ��İ汾�еõ�֧��
		// ��Windowsƽ̨�У�ֻʵ���˶�OpenGL1.1��֧�֣����Ϊ��ʹ����Щ�߼����ԣ����Ի���Ҫ
		// ��ȡ��������չ��ַ����Щ����ֻ��loadFunc��ʵ�֡�
		PFNGLTEXIMAGE3DPROC    glTexImage3D;
		PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D;

	public:
		// �����ڴ�С�����Ķ�ʱ
		void sgResizeScreen(int width, int height);
		// ��ʾ����
		void sgDisplay(void);
		// Keyboard function
		//void sgKeyboard(unsigned char key, int mousePositionX, int mousePositionY);
		void sgKeyboard(sge::SGKEYS);
		// Mouse function
		void sgMouse(int button,int state,int x,int y);
		// ����Volume
		int sgLoadVolumeData(_volumeData const *data_in);
		// ���ô��ڲ���
		void sgSetWindowParam(_windowSets const *win_in);
		// ��ȡ���ڲ���
		inline _windowSets *sgGetWindowParam( ){ return &m_window; };
		// ��ʼ��
		void sgInit(int width, int height);
		// ����offset, 2-D
		int TEXEL2(int s, int t) { return  BYTES_PER_TEXEL * (s * m_volume.width + t); };
		// ����offset, 3-D
		int TEXEL3(int s, int t,  int r) { return  TEXEL2(s, t) + LAYER(r); };
		// ����˶�״̬
		void sgMotion(int x,int y);

	private:
		// ��ȡ�й�OpenGL������չ��ַ����Щ����ֻ��Ҫ��Windowsƽ̨��
		int loadFunc(void);
		// ��������3D����
		int bindTexutre(void);
		// ���ƴ�����
		void agentBox(void);
		// ��ʼ��FPS�йز���������
		void initFPS(void), initViewMatrix(void);
		// ����layer
		int LAYER(int r) {return m_volume.width * m_volume.height * r * BYTES_PER_TEXEL;};
		// FPS function
		void sgCountFPS(void);
	} Visualization;
};

#endif