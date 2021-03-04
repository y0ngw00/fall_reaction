#ifndef __MOTION_WIDGET_H__
#define __MOTION_WIDGET_H__
#include <GL/glew.h>
#include <GL/glut.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <memory>
#include <iostream>
#include <chrono>
#include <algorithm>
#include "GLUTWindow.h"
#include "SkeletonBuilder.h"
#include "Functions.h"
#include "GLfunctions.h"
#include "DART_interface.h"
#include "BVH.h"
#include "Character.h"
#include "Controller.h"
#include "ReferenceManager.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class MainInterface : public GLUTWindow
{
public:
	MainInterface();
	MainInterface(std::string bvh, std::string ppo);

	void DrawGround();
	void display();
	void motion(int mx, int my); 
	void mouse(int button, int state, int mx, int my);
	void keyboard(unsigned char key, int mx, int my);
	void skeyboard(int key, int x, int y);
	void reshape(int w, int h);
	void Timer(int value);
	void Reset();

	void SetFrame(int n);
	void DrawSkeletons();

 	void initNetworkSetting(std::string ppo);
 	void UpdateMotion(std::vector<Eigen::VectorXd> motion, const char* type);
 	void RunPPO();

	
protected:

	Camera* 		mCamera;
	//BVH* 			current_bvh;
	DPhy::ReferenceManager*			mReferenceManager;
	DPhy::Controller* 				mController;

	std::string character_path;

	dart::dynamics::SkeletonPtr 	mSkel;
	dart::dynamics::SkeletonPtr 	mSkel_sim;

	dart::dynamics::SkeletonPtr 	mObj_1;
	dart::dynamics::SkeletonPtr 	mObj_2;

	int     drag_mouse_r;
	int     drag_mouse_l;
	int     last_mouse_x, last_mouse_y;

	int mw,mh;
	double phase;
	std::vector<double>				mTiming; 
	std::vector<Eigen::VectorXd> mMotion_bvh;
	std::vector<Eigen::VectorXd> mMotion_sim;
	std::vector<Eigen::VectorXd> mMotion_reg;

	int mx;
	int my;
	int frame_no;
	int mDisplayTimeout;
	std::chrono::steady_clock::time_point begin;

	bool on_animation;
	int speed_type;
	int motion_type;

	int mCurFrame;
	int mTotalFrame;

	//p::object 						mRegression;
	py::object 						mPPO;



	bool render_bvh=false;
	bool render_sim=false;

	
};


#endif