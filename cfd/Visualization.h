#ifndef _SEAGO_VISUALIZATION_H_
#define _SEAGO_VISUALIZATION_H_
#pragma once

#include "Headers.h"

#define BYTES_PER_TEXEL 4

namespace sge
{
	struct _mouse
	{
		GLuint pre_cursor_x, pre_cursor_y;
		GLuint cur_cursor_x, cur_cursor_y;
		GLboolean left_button_pressed;
		GLboolean right_button_pressed;
	};

	struct _fps
	{
		// Variables used to calculate frames per second:
		DWORD dwFrames;
		DWORD dwCurrentTime;
		DWORD dwLastUpdateTime;
		DWORD dwElapsedTime;
		UINT  FPS;
	}; // Frames per second

	struct _volumeData
	{
		GLubyte *data;
		GLuint textureID;
		GLuint width, height, depth;
	};

	struct _viewMatrix
	{
		GLfloat field_of_view_angle;
		GLfloat z_near, z_far;
		GLfloat rotate_of_x;
		GLfloat rotate_of_y;
		GLfloat rotate_of_z;
		GLfloat eye_at_x, eye_at_y, eye_at_z;
		GLfloat look_at_x, look_at_y, look_at_z;
		GLfloat dx_up_x, dx_up_y, dx_up_z;
		GLfloat radius_of_view_matrix;
		GLfloat z_forward;
	};
		

	class Visualization
	{
	public:
		Visualization(MainActivity *hActivity);
		~Visualization();

	public:
		// Call this function when window resized
		static void ResizeScreen(GLuint width, GLuint height);
		// Display image
		static void Display(GLvoid);
		// Keyboard function
		static void Keyboard(SG_KEYS keys, SG_KEY_STATUS status);
		// Mouse function
		static void Mouse(SG_MOUSE mouse, GLuint x_pos, GLuint y_pos);
		// Load volume data from external source
		int LoadVolumeData(_volumeData const *data_in);
		// Save volume data to external source
		int SaveVolumeData(_volumeData *data_out);
		// Setup view matrix
		int SetWindowParam(_viewMatrix const *view_matrix_in);
		// Get view matrix
		int GetWindowParam(_viewMatrix *view_matrix_out);
		// Initialize class
		void Init(GLuint width, GLuint height);
		// Offset, 2-D
		GLuint TEXEL2(GLuint s, GLuint t);
		// Offset, 3-D
		GLuint TEXEL3(GLuint s, GLuint t,  GLuint r);

		void GetActivityHandler(MainActivity *hActivity);


	private:
		// 创建并绑定3D纹理
		void Bind3DTexutre(void);
		// 绘制代理几何
		static void CreateAgentBox(void);
		// 初始化FPS有关操作及数据
		void InitFPS(void), InitViewMatrix(void), InitFont(void);
		// 计算layer
		GLuint LAYER(GLuint r);
		// FPS function
		static void CountFPS(void);

	};

};

#endif