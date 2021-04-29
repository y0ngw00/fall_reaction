
#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <memory>
#include <iostream>

#include "openglrecorder.h"

#include "Camera.h"

class GLUTWindow
{
public:
	GLUTWindow();
	~GLUTWindow();

	virtual void GLInitWindow(const char* _name);
	
	static GLUTWindow* current();
	static void DisplayEvent();
	static void KeyboardEvent(unsigned char key,int x,int y);
	static void SKeyboardEvent(int key,int x,int y);
	static void MouseEvent(int button, int state, int x, int y);
	static void MotionEvent(int x, int y);
	static void ReshapeEvent(int w, int h);
	static void IdleEvent();
	static void TimerEvent(int value);

	static void glRecordInitialize();


	static std::vector<GLUTWindow*> mWindows;
	static std::vector<int> mWinIDs;
	
protected:
	virtual void GLinitEnvironment();
	virtual void DrawGround()=0;
	virtual void display()=0;
	virtual void motion(int mx, int my)=0; 
	virtual void mouse(int button, int state, int mx, int my)=0;
	virtual void keyboard(unsigned char key, int mx, int my)=0;
	virtual void skeyboard(int key, int x, int y)=0;
	virtual void Timer(int value)=0;
	virtual void reshape(int w, int h)=0;
protected:

	Camera* 		mCamera;

	int 	mDisplayTimeout;

	int     drag_mouse_r;
	int     drag_mouse_l;
	int     last_mouse_x, last_mouse_y;

	int mw,mh;


	int mx;
	int my;
	int frame_no;
};



