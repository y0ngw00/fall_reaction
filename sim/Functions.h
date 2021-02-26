#ifndef __DEEP_PHYSICS_FUNCTIONS_H__
#define __DEEP_PHYSICS_FUNCTIONS_H__
#include "dart/dart.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace DPhy
{

	py::array_t<float> toNumPyArray(const std::vector<float>& val);
	py::array_t<double> toNumPyArray(const std::vector<double>& val);
	py::array_t<double> toNumPyArray(const std::vector<Eigen::VectorXd>& val);
	py::array_t<double> toNumPyArray(const std::vector<Eigen::MatrixXd>& val);
	py::array_t<double> toNumPyArray(const std::vector<std::vector<double>>& val);
	py::array_t<double> toNumPyArray(const std::vector<std::vector<double>>& val);
	py::array_t<double> toNumPyArray(const std::vector<bool>& val);
	py::array_t<double> toNumPyArray(const Eigen::VectorXd& vec);
	py::array_t<double> toNumPyArray(const Eigen::MatrixXd& matrix);
	py::array_t<double> toNumPyArray(const Eigen::Isometry3d& T);
	Eigen::VectorXd toEigenVector(py::array_t<double>& array);
	Eigen::VectorXd toEigenVector(py::array_t<double>& array, int n);
	std::vector<Eigen::VectorXd> toEigenVectorVector(py::array_t<double>& array);
	Eigen::MatrixXd toEigenMatrix(py::array_t<double>& array);
	Eigen::MatrixXd toEigenMatrix(py::array_t<double>& array, int n, int m);
	std::vector<bool> toStdVector(const py::list& list);

	// Utilities
	std::vector<double> split_to_double(const std::string& input, int num);
    std::vector<double> split_to_double(const std::string& input);
	Eigen::Vector3d string_to_vector3d(const std::string& input);
	Eigen::VectorXd string_to_vectorXd(const std::string& input, int n);
    Eigen::VectorXd string_to_vectorXd(const std::string& input);
	Eigen::Matrix3d string_to_matrix3d(const std::string& input);
	std::string vectorXd_to_string(const Eigen::VectorXd& vec);

	double exp_of_squared(const Eigen::VectorXd& vec,double sigma = 1.0);
	double exp_of_squared(const Eigen::Vector3d& vec,double sigma = 1.0);
	double exp_of_squared(const Eigen::MatrixXd& mat,double sigma = 1.0);
	std::pair<int, double> maxCoeff(const Eigen::VectorXd& in);

	double RadianClamp(double input);
	std::vector<dart::dynamics::BodyNode*> GetChildren(const dart::dynamics::SkeletonPtr& skel, const dart::dynamics::BodyNode* parent);

	Eigen::Quaterniond DARTPositionToQuaternion(Eigen::Vector3d in);
	Eigen::Vector3d QuaternionToDARTPosition(const Eigen::Quaterniond& in);
	void QuaternionNormalize(Eigen::Quaterniond& in);
	Eigen::VectorXd BlendPosition(Eigen::VectorXd v_target, Eigen::VectorXd v_source, double weight, bool blend_rootpos=true);
	Eigen::VectorXd BlendVelocity(Eigen::VectorXd target_a, Eigen::VectorXd target_b, double weight);
	Eigen::Vector3d NearestOnGeodesicCurve3d(Eigen::Vector3d targetAxis, Eigen::Vector3d targetPosition, Eigen::Vector3d position);
	Eigen::VectorXd NearestOnGeodesicCurve(Eigen::VectorXd targetAxis, Eigen::VectorXd targetPosition, Eigen::VectorXd position);
	Eigen::VectorXd RotatePosition(Eigen::VectorXd pos, Eigen::VectorXd rot);
	Eigen::Vector3d JointPositionDifferences(Eigen::Vector3d q2, Eigen::Vector3d q1);
	Eigen::Vector3d LinearPositionDifferences(Eigen::VectorXd v2, Eigen::Vector3d v1, Eigen::Vector3d q1);
	Eigen::Vector3d Rotate3dVector(Eigen::Vector3d v, Eigen::Vector3d r);
	void SetBodyNodeColors(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& color);
	void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector3d& color);
	void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector4d& color);
	std::vector<Eigen::VectorXd> Align(std::vector<Eigen::VectorXd> data, Eigen::VectorXd target);

	void EditBVH(std::string& path);
	Eigen::Quaterniond GetYRotation(Eigen::Quaterniond q);

	Eigen::Vector3d changeToRNNPos(Eigen::Vector3d pos);
	Eigen::Isometry3d getJointTransform(dart::dynamics::SkeletonPtr skel, std::string bodyname);
	Eigen::Vector4d rootDecomposition(dart::dynamics::SkeletonPtr skel, Eigen::VectorXd positions);
	Eigen::VectorXd solveIK(dart::dynamics::SkeletonPtr skel, const std::string& bodyname, const Eigen::Vector3d& delta,  const Eigen::Vector3d& offset);
	Eigen::VectorXd solveMCIK(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints);
	Eigen::VectorXd solveMCIKRoot(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints);
	Eigen::Matrix3d projectToXZ(Eigen::Matrix3d m);
	Eigen::Vector3d projectToXZ(Eigen::Vector3d v);

	std::vector<std::string> split(const std::string &s, char delim);
	Eigen::MatrixXd getPseudoInverse(Eigen::MatrixXd m);
}

#endif