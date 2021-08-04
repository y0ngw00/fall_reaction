#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

class Camera
{
public:
	Camera();
	//Camera(int w, int h);
	
	void Rotate(int mx, int my, int prev_x, int prev_y);
	void viewupdate();
	void Translate(int mx, int my, int prev_x, int prev_y);
	void SetLookAt(Eigen::Vector3d& new_lookAt);
	void Zoom(int degree);


protected:
	Eigen::Vector3d lookAt;
	Eigen::Vector3d eye;
	Eigen::Vector3d up;
	double fovy;


};


