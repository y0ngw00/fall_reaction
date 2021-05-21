#include "Camera.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>

Camera::
Camera()
	:fovy(40.0),lookAt(Eigen::Vector3d(0,0.8,0)),eye(Eigen::Vector3d(0,3,15)),up(Eigen::Vector3d(0,1,0))
{

}

void 
Camera::
Rotate(int mx, int my, int prev_x, int prev_y)
{
	GLint w = glutGet(GLUT_WINDOW_WIDTH);
	GLint h = glutGet(GLUT_WINDOW_HEIGHT);

	double rad = std::min(w, h) / 2.0;
	double dx = (double)mx - (double)prev_x;
	double dy = (double)my - (double)prev_y;
	double angleY = atan2(dx * 0.5, rad);
	double angleX = atan2(dy * 0.5, rad);

	Eigen::Vector3d n = this->lookAt - this->eye;

	Eigen::Vector3d axisX = Eigen::Vector3d::UnitY().cross(n.normalized());
	n = Eigen::Quaterniond(Eigen::AngleAxisd(-angleY, Eigen::Vector3d::UnitY()))._transformVector(n);
	n = Eigen::Quaterniond(Eigen::AngleAxisd(angleX, axisX))._transformVector(n);
	
	this->eye = this->lookAt - n;

	glutPostRedisplay();
}

void 
Camera::
viewupdate()
{
	gluLookAt(eye.x(), eye.y(), eye.z(),
		lookAt.x(), lookAt.y(), lookAt.z(),
		up.x(), up.y(), up.z());
}


void 
Camera::
Translate(int mx, int my, int prev_x, int prev_y)
{
	Eigen::Vector3d delta((double)mx - (double)prev_x, (double)my - (double)prev_y, 0);
	Eigen::Vector3d yvec = lookAt - eye;
	yvec[1]=0;

	double scale = yvec.norm()/1000.;
	yvec.normalize();
	Eigen::Vector3d xvec = -yvec.cross(this->up);
	xvec.normalize();

	delta = delta[0]*xvec*scale + delta[1]*yvec*scale;

	lookAt += delta; eye += delta;

	glutPostRedisplay();
}

void
Camera::
SetLookAt(Eigen::Vector3d& new_lookAt)
{
	this->lookAt = new_lookAt;
	this->eye = new_lookAt;
	eye[2] += 2;
}

void
Camera::
SetEye(Eigen::Vector3d& new_eye)
{
	
	this->eye = this->lookAt + new_eye;
	glutPostRedisplay();
}


void Camera::Zoom(int degree){

	double delta = degree * 0.1;
	Eigen::Vector3d vec = lookAt - eye;
	double scale = vec.norm();
	scale = std::max(scale - delta,1.0);
	Eigen::Vector3d vd =(scale-delta) * (lookAt - eye).normalized();

	this->eye = this->lookAt - vd;

	glutPostRedisplay();
}
