#include "Functions.h"
#include "CharacterConfigurations.h"
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <Eigen/Eigenvalues>

namespace py = pybind11;
namespace DPhy
{
py::array_t<float> toNumPyArray(const std::vector<float>& val)
{
    size_t n = val.size();
    py::array_t<float> array(n);
    py::buffer_info buf = array.request(true);

    std::memcpy(buf.ptr, val.data(), n * sizeof(float));
	return array;
}
py::array_t<double> toNumPyArray(const std::vector<double>& val)
{
    size_t n = val.size();
    py::array_t<double> array(n);
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);

    for (int i = 0; i < n; i++) {
        ptr[i] = (float)val[i];
    }
	return array;
}
py::array_t<double> toNumPyArray(const std::vector<Eigen::VectorXd>& val)
{
	uint n = val.size();
	uint m = val[0].rows();
	py::array_t<double> array({n, m});
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
	    for (int j = 0; j < m; j++) {
	        ptr[i*m+j] = (float)val[i](j);
	    }
	}

	return array;	
}
py::array_t<double> toNumPyArray(const std::vector<Eigen::MatrixXd>& val)
{
	uint n = val.size();
	uint m = val[0].rows();
	uint l = val[0].cols();

    py::array_t<double> array({n, m, l});
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);

	int index = 0;
	for(int i=0;i<n;i++) {
        for (int j = 0; j < m; j++)
            for (int k = 0; k < l; k++)
                ptr[i*m*l + j*l + k] = (double)val[i](j,k);
    }

	return array;
}
py::array_t<double> toNumPyArray(const std::vector<std::vector<float>>& val)
{
	uint n = val.size();
	uint m = val[0].size();

    py::array_t<double> array({n, m});
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);

	int index = 0;
	for(int i=0;i<n;i++)
	{
        std::memcpy(ptr + i*m, val[i].data(), m*sizeof(double));
	}

	return array;
}
py::array_t<double> toNumPyArray(const std::vector<std::vector<double>>& val)
{
	uint n = val.size();
	uint m = val[0].size();

    py::array_t<double> array({n, m});
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
		    ptr[i*m+j] = (double)val[i][j];
		}
	}

	return array;
}
//always return 1-dim array
py::array_t<double> toNumPyArray(const std::vector<bool>& val)
{
	uint n = val.size();

    py::array_t<double> array(n);
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
        ptr[i] = (double)val[i];
	}

	return array;
}

//always return 1-dim array
py::array_t<double> toNumPyArray(const Eigen::VectorXd& vec)
{
	uint n = vec.rows();

    py::array_t<double> array(n);
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);
	for(int i =0;i<n;i++)
	{
        ptr[i] = (double)vec(i);
	}
	return array;
}
//always return 2-dim array
py::array_t<double> toNumPyArray(const Eigen::MatrixXd& matrix)
{
	uint n = matrix.rows();
	uint m = matrix.cols();

    py::array_t<double> array({n, m});
    py::buffer_info buf = array.request(true);
    double* ptr = reinterpret_cast<double*>(buf.ptr);

	int index = 0;
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
            ptr[i*m+j] = (double)matrix(i,j);
		}
	}

	return array;
}
//always return 2-dim array
py::array_t<double> toNumPyArray(const Eigen::Isometry3d& T)
{
	uint n = 4;
	uint m = 4;

    py::array_t<float> array({n, m});
    py::buffer_info buf = array.request(true);
	double* dest = reinterpret_cast<double*>(buf.ptr);

	Eigen::Matrix3d R = T.linear();
	Eigen::Vector3d p = T.translation();
	dest[0] = T(0,0),dest[1] = T(0,1),dest[2] = T(0,2),dest[3] = p[0];
	dest[4] = T(1,0),dest[5] = T(1,1),dest[6] = T(1,2),dest[7] = p[1];
	dest[8] = T(2,0),dest[9] = T(2,1),dest[10] = T(2,2),dest[11] = p[2];
	dest[12] = 0.0,dest[13] = 0.0,dest[14] = 0.0,dest[15] = 1.0;

	return array;
}
Eigen::VectorXd toEigenVector(py::array_t<double>& array)
{
	Eigen::VectorXd vec(array.shape(0));

    py::buffer_info buf = array.request();
	double* srcs = reinterpret_cast<double*>(buf.ptr);

	for(int i=0;i<array.shape(0);i++)
	{
		vec(i) = (double)srcs[i];
	}
	return vec;
}
Eigen::VectorXd toEigenVector(py::array_t<double>& array, int n)
{
	Eigen::VectorXd vec(n);

    py::buffer_info buf = array.request();
	double* srcs = reinterpret_cast<double*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
		vec(i) = (double)srcs[i];
	}
	return vec;
}
std::vector<Eigen::VectorXd> toEigenVectorVector(py::array_t<double>& array)
{
	std::vector<Eigen::VectorXd> mat;
	mat.resize(array.shape(0));

    py::buffer_info buf = array.request();
	double* srcs = reinterpret_cast<double*>(buf.ptr);
	int n = array.shape(0);
	int m = array.shape(1);

	for(int i=0;i<n;i++){
		mat[i].resize(m);
		for(int j=0;j<m;j++)
			mat[i](j) = (double)srcs[i*m+j];
	}

	return mat;	
}
Eigen::MatrixXd toEigenMatrix(py::array_t<double>& array)
{
	Eigen::MatrixXd mat(array.shape(0),array.shape(1));

    py::buffer_info buf = array.request();
	double* srcs = reinterpret_cast<double*>(buf.ptr);
    int n = array.shape(0);
    int m = array.shape(1);

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			mat(i,j) = srcs[i*m+j];
		}
	}
	return mat;
}
Eigen::MatrixXd toEigenMatrix(py::array_t<double>& array,int n,int m)
{
	Eigen::MatrixXd mat(n,m);

    py::buffer_info buf = array.request();
	double* srcs = reinterpret_cast<double*>(buf.ptr);

	for(int i=0;i<n;i++)
	{
		for(int j=0;j<m;j++)
		{
			mat(i,j) = srcs[i*m+j];
		}
	}
	return mat;
}
std::vector<bool> toStdVector(const py::list& list)
{
	std::vector<bool> vec(py::len(list));
	for(int i =0;i<vec.size();i++)
	    vec[i] = list[i].cast<bool>();
	return vec;
}

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
       	*(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim=' ') {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

std::string join(const std::vector<std::string> &v, char delim=' '){
	std::stringstream ss;
	for(size_t i = 0; i < v.size(); ++i)
	{
		if(i != 0)
			ss << delim;
		ss << v[i];
	}

	return ss.str();
}

std::vector<double> split_to_double(const std::string& input, int num)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    for(int i = 0; i < num; i++){
        result.push_back(std::stold(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

std::vector<double> split_to_double(const std::string& input)
{
    std::vector<double> result;
    std::string::size_type sz = 0, nsz = 0;
    while(sz< input.length()){
        result.push_back(std::stold(input.substr(sz), &nsz));
        sz += nsz;
    }
    return result;
}

Eigen::Vector3d string_to_vector3d(const std::string& input){
	std::vector<double> v = split_to_double(input, 3);
	Eigen::Vector3d res;
	res << v[0], v[1], v[2];

	return res;
}

Eigen::VectorXd string_to_vectorXd(const std::string& input, int n){
	std::vector<double> v = split_to_double(input, n);
	Eigen::VectorXd res(n);
	for(int i = 0; i < n; i++){
		res[i] = v[i];
	}
	return res;
}

Eigen::VectorXd string_to_vectorXd(const std::string& input){
    std::vector<double> v = split_to_double(input);
    Eigen::VectorXd res(v.size());
    for(int i = 0; i < v.size(); i++){
        res[i] = v[i];
    }
    return res;
}

    Eigen::Matrix3d string_to_matrix3d(const std::string& input){
	std::vector<double> v = split_to_double(input, 9);
	Eigen::Matrix3d res;
	res << v[0], v[1], v[2],
			v[3], v[4], v[5],
			v[6], v[7], v[8];

	return res;
}
std::string vectorXd_to_string(const Eigen::VectorXd& vec) {
	std::string str= "";
	for(int i = 0; i < vec.rows(); i++) {
		str += std::to_string(vec(i));
		if(i != vec.rows() - 1)
			str += " ";
	}
	return str;
}

double RadianClamp(double input){
	return std::fmod(input+M_PI, 2*M_PI)-M_PI;
}

double exp_of_squared(const Eigen::VectorXd& vec,double sigma)
{
	return exp(-1.0*vec.dot(vec)/(sigma*sigma)/vec.rows());
}
double exp_of_squared(const Eigen::Vector3d& vec,double sigma)
{
	return exp(-1.0*vec.dot(vec)/(sigma*sigma)/vec.rows());
}
double exp_of_squared(const Eigen::MatrixXd& mat,double sigma)
{
	return exp(-1.0*mat.squaredNorm()/(sigma*sigma)/mat.size());
}


std::pair<int, double> maxCoeff(const Eigen::VectorXd& in){
	double m = 0;
	int idx = 0;
	for(int i = 0; i < in.rows(); i++){
		if( m < in[i]){
			m = in[i];
			idx = i;
		}
	}
	return std::make_pair(idx, m);
}

void SetBodyNodeColors(dart::dynamics::BodyNode* bn, const Eigen::Vector3d& color)
{
	auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
	for(auto visualShapeNode : visualShapeNodes)
		visualShapeNode->getVisualAspect()->setColor(color);
}

void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector3d& color)
{
	// Set the color of all the shapes in the object
	for(std::size_t i=0; i < object->getNumBodyNodes(); ++i)
	{
		Eigen::Vector3d c = color;
		dart::dynamics::BodyNode* bn = object->getBodyNode(i);
		if(bn->getName() == "Neck")
			c.head<3>() *= 0.5;
		auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
		for(auto visualShapeNode : visualShapeNodes)
			visualShapeNode->getVisualAspect()->setColor(c);
	}
}

void SetSkeletonColor(const dart::dynamics::SkeletonPtr& object, const Eigen::Vector4d& color)
{
	// Set the color of all the shapes in the object
	for(std::size_t i=0; i < object->getNumBodyNodes(); ++i)
	{
		Eigen::Vector4d c = color;
		dart::dynamics::BodyNode* bn = object->getBodyNode(i);
		if(bn->getName() == "Neck")
			c.head<3>() *= 0.5;
		auto visualShapeNodes = bn->getShapeNodesWith<dart::dynamics::VisualAspect>();
		for(auto visualShapeNode : visualShapeNodes)
			visualShapeNode->getVisualAspect()->setRGBA(c);
	}
}

std::vector<dart::dynamics::BodyNode*> GetChildren(const dart::dynamics::SkeletonPtr& skel, 
												   const dart::dynamics::BodyNode* parent){
	std::vector<dart::dynamics::BodyNode*> childs;
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{
		auto bn = skel->getBodyNode(i);
		auto pn = bn->getParentBodyNode();
		if(pn && !pn->getName().compare(parent->getName()))
			childs.push_back(bn);
	}
	return childs;
}
Eigen::Quaterniond DARTPositionToQuaternion(Eigen::Vector3d in){
	if( in.norm() < 1e-8 ){
		return Eigen::Quaterniond::Identity();
	}
	Eigen::AngleAxisd aa(in.norm(), in.normalized());
	Eigen::Quaterniond q(aa);
	QuaternionNormalize(q);
	return q;
}

Eigen::Vector3d QuaternionToDARTPosition(const Eigen::Quaterniond& in){
	Eigen::AngleAxisd aa(in);
	double angle = aa.angle();
	angle = std::fmod(angle+M_PI, 2*M_PI)-M_PI;
	return angle*aa.axis();
}

Eigen::VectorXd BlendPosition(Eigen::VectorXd target_a, Eigen::VectorXd target_b, double weight, bool blend_rootpos) {

	Eigen::VectorXd result(target_a.rows());
	result = target_a;

	for(int i = 0; i < result.size(); i += 3) {
		if (i == 3) {
			if(blend_rootpos)	result.segment<3>(i) = (1 - weight) * target_a.segment<3>(i) + weight * target_b.segment<3>(i); 
			else result[4] = (1-weight) * target_a[4] + weight * target_b[4]; 
		} else if (i == 0 && !blend_rootpos) {
			Eigen::Vector3d v_a = target_a.segment<3>(i);
			Eigen::Vector3d v_b = target_b.segment<3>(i);
	
			result(i) = v_a(0) * (1-weight) + v_b(0) * weight;
			result(i+1) = target_a(i+1);
			result(i+2) = v_a(2) * (1-weight) + v_b(2) * weight;
		} else {
			Eigen::AngleAxisd v1_aa(target_a.segment<3>(i).norm(), target_a.segment<3>(i).normalized());
			Eigen::AngleAxisd v2_aa(target_b.segment<3>(i).norm(), target_b.segment<3>(i).normalized());
					
			Eigen::Quaterniond v1_q(v1_aa);
			Eigen::Quaterniond v2_q(v2_aa);

			result.segment<3>(i) = QuaternionToDARTPosition(v1_q.slerp(weight, v2_q)); 
		}
	}

	return result;
}
Eigen::VectorXd BlendVelocity(Eigen::VectorXd target_a, Eigen::VectorXd target_b, double weight) {

	Eigen::VectorXd result(target_a.rows());
	result= (1 - weight) * target_a + weight * target_b; 
		
	return result;
}
Eigen::VectorXd RotatePosition(Eigen::VectorXd pos, Eigen::VectorXd rot)
{
	Eigen::VectorXd vec(pos.rows());
	for(int i = 0; i < pos.rows(); i += 3) {
		if(i != 3) {
			Eigen::AngleAxisd aa1 = Eigen::AngleAxisd(pos.segment<3>(i).norm(), pos.segment<3>(i).normalized());
			Eigen::AngleAxisd aa2 = Eigen::AngleAxisd(rot.segment<3>(i).norm(), rot.segment<3>(i).normalized());
			Eigen::Matrix3d m;
			m = aa1 * aa2;
			Eigen::AngleAxisd vec_seg(m);
			vec.segment<3>(i) = vec_seg.axis() * vec_seg.angle();
		} else {
			vec.segment<3>(i) = pos.segment<3>(i);
		}
	}
	return vec;
}
Eigen::Vector3d Rotate3dVector(Eigen::Vector3d v, Eigen::Vector3d r) {
	Eigen::AngleAxisd aa1 = Eigen::AngleAxisd(v.norm(), v.normalized());
	Eigen::AngleAxisd aa2 = Eigen::AngleAxisd(r.norm(), r.normalized());
  	Eigen::AngleAxisd aa;
  	aa = aa1 * aa2;
  	return aa.axis() * aa.angle();
}

Eigen::Vector3d JointPositionDifferences(Eigen::Vector3d q2, Eigen::Vector3d q1)
{
	Eigen::AngleAxisd aa1 = Eigen::AngleAxisd(q1.norm(), q1.normalized());
	Eigen::AngleAxisd aa2 = Eigen::AngleAxisd(q2.norm(), q2.normalized());
  	Eigen::AngleAxisd aa;
  	aa = aa1.inverse() * aa2;
  	return aa.axis() * aa.angle();
}
Eigen::Vector3d LinearPositionDifferences(Eigen::VectorXd v2, Eigen::Vector3d v1, Eigen::Vector3d q1)
{
	Eigen::AngleAxisd aa1 = Eigen::AngleAxisd(q1.norm(), q1.normalized());
	Eigen::Vector3d diff = v2 - v1;

	return aa1.inverse() * diff;
}
Eigen::Vector3d NearestOnGeodesicCurve3d(Eigen::Vector3d targetAxis, Eigen::Vector3d targetPosition, Eigen::Vector3d position) {
	Eigen::Quaterniond v1_q = DARTPositionToQuaternion(position);
	Eigen::Quaterniond q = DARTPositionToQuaternion(targetPosition);
	Eigen::Vector3d axis = targetAxis.normalized();
	double ws = v1_q.w();
	Eigen::Vector3d vs = v1_q.vec();
	double w0 = q.w();
	Eigen::Vector3d v0 = q.vec();

	double a = ws*w0 + vs.dot(v0);
	double b = w0*(axis.dot(vs)) - ws*(axis.dot(v0)) + vs.dot(axis.cross(v0));

	double alpha = atan2( a,b );

	double t1 = -2*alpha + M_PI;
	Eigen::Quaterniond t1_q(Eigen::AngleAxisd(t1, axis));
	double t2 = -2*alpha - M_PI;
	Eigen::Quaterniond t2_q(Eigen::AngleAxisd(t2, axis));

	if (v1_q.dot(t1_q) > v1_q.dot(t2_q))
	{	
		return QuaternionToDARTPosition(t1_q);
	} else {
		return QuaternionToDARTPosition(t2_q);
	}
}
Eigen::VectorXd NearestOnGeodesicCurve(Eigen::VectorXd targetAxis, Eigen::VectorXd targetPosition, Eigen::VectorXd position){
	Eigen::VectorXd result(targetAxis.rows());
	result.setZero();
	for(int i = 0; i < targetAxis.size(); i += 3) {
		if (i!= 3) {
			result.segment<3>(i) = NearestOnGeodesicCurve3d(targetAxis.segment<3>(i), targetPosition.segment<3>(i), position.segment<3>(i));
		}
	}
	return result;
}
void QuaternionNormalize(Eigen::Quaterniond& in){
	if(in.w() < 0){
		in.coeffs() *= -1;
	}
}

// void EditBVH(std::string& path){
// 	double scale = 100;
// 	std::ifstream ifs(path);
// 	std::vector<std::string> out;
// 	std::string line;

// 	while(true){
// 		if(!std::getline(ifs, line))
// 			break;

// 		if(line == "MOTION"){
// 			out.push_back(line);
// 			break;
// 		}

// 		int space_count = line.length();
// 		for(int i = 0; i < line.length(); i++){
// 			if(line[i] != ' '){
// 				space_count = i;
// 				break;
// 			}
// 		}
// 		if(space_count == line.length()){
// 			out.push_back(line);
// 			continue;
// 		}

// 		std::vector<std::string> sp = split(line, ' ');
// 		if(sp[space_count] == "OFFSET"){
// 			sp[space_count+1] = std::to_string(std::stold(sp[space_count+1])*scale);
// 			sp[space_count+2] = std::to_string(std::stold(sp[space_count+2])*scale);
// 			sp[space_count+3] = std::to_string(std::stold(sp[space_count+3])*scale);

// 			std::string new_line = join(sp);
// 			out.push_back(new_line);
// 		}
// 		else{
// 			out.push_back(line);
// 			continue;			
// 		}
// 	}
// 	std::getline(ifs, line);
// 	out.push_back(line);
// 	std::getline(ifs, line);
// 	out.push_back(line);

// 	while(std::getline(ifs, line)){
// 		std::vector<std::string> sp = split(line, ' ');
// 		Eigen::Vector3d pos, rot;
// 		pos << std::stold(sp[0]), std::stold(sp[1]), std::stold(sp[2]);
// 		rot << std::stold(sp[3]), std::stold(sp[4]), std::stold(sp[5]);
// 		rot = rot*M_PI/180.;

// 		pos = pos * scale;
// 		double tmp = pos[0];
// 		pos[0] = pos[2];
// 		pos[2] = -tmp;

// 		Eigen::AngleAxisd rotaa;
// 		rotaa = Eigen::AngleAxisd(rot[0], Eigen::Vector3d::UnitZ())
// 			* Eigen::AngleAxisd(rot[1], Eigen::Vector3d::UnitX())
// 			* Eigen::AngleAxisd(rot[2], Eigen::Vector3d::UnitY());

// 		rotaa = Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitY())*rotaa;
// 		Eigen::Matrix3d m(rotaa);
// 		Eigen::Vector3d fixed_rot = m.eulerAngles(2,0,1);
// 		fixed_rot = fixed_rot * 180./M_PI;

// 		sp[0] = std::to_string(pos[0]);
// 		sp[1] = std::to_string(pos[1]);
// 		sp[2] = std::to_string(pos[2]);
// 		sp[3] = std::to_string(fixed_rot[0]);
// 		sp[4] = std::to_string(fixed_rot[1]);
// 		sp[5] = std::to_string(fixed_rot[2]);

// 		std::string new_line = join(sp);
// 		out.push_back(new_line);
// 	}	
// 	ifs.close();

// 	std::ofstream outputfile(path.substr(0,path.length()-4) + std::string("_fixed_c.bvh"));
// 	for(auto& s : out){
// 		outputfile << s << std::endl;
// 	}
// 	outputfile.close();
// }

Eigen::Quaterniond GetYRotation(Eigen::Quaterniond q){
	// from body joint vector
	Eigen::Vector3d rotated = q._transformVector(Eigen::Vector3d::UnitZ());
	double angle = atan2(rotated[0], rotated[2]);
	Eigen::Quaterniond ret(Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()));

	return ret;
}


// Eigen::Vector3d changeToRNNPos(Eigen::Vector3d pos){
// 	Eigen::Vector3d ret;
// 	ret[0] = pos[2]*100;
// 	ret[1] = (pos[1]-ROOT_HEIGHT_OFFSET)*100;
// 	ret[2] = -pos[0]*100;
// 	return ret;
// }

Eigen::Isometry3d getJointTransform(dart::dynamics::SkeletonPtr skel, std::string bodyname){
	return skel->getBodyNode(bodyname)->getParentBodyNode()->getWorldTransform()
		*skel->getBodyNode(bodyname)->getParentJoint()->getTransformFromParentBodyNode();
}

Eigen::Vector4d rootDecomposition(dart::dynamics::SkeletonPtr skel, Eigen::VectorXd positions){
	// DEBUG : decomposition
	Eigen::VectorXd p_save = skel->getPositions();
	skel->setPositions(positions);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);

	Eigen::Isometry3d femur_l_transform = getJointTransform(skel, "FemurL");
	Eigen::Isometry3d femur_r_transform = getJointTransform(skel, "FemurR");

	Eigen::Vector3d up_vec = Eigen::Vector3d::UnitY();
	Eigen::Vector3d x_vec = femur_l_transform.translation() - femur_r_transform.translation();
	x_vec.normalize();
	Eigen::Vector3d z_vec = x_vec.cross(up_vec);
	z_vec[1] = 0;
	z_vec.normalize();
	double angle = std::atan2(z_vec[0], z_vec[2]);

	skel->setPositions(p_save);

	Eigen::AngleAxisd aa_root(angle, Eigen::Vector3d::UnitY());
	Eigen::AngleAxisd aa_hip(positions.segment<3>(0).norm(), positions.segment<3>(0).normalized());

	Eigen::Vector3d hip_dart = QuaternionToDARTPosition(Eigen::Quaterniond(aa_root).inverse()*Eigen::Quaterniond(aa_hip));
	
	Eigen::Vector4d ret;
	ret << angle, hip_dart;

	return ret;
}

Eigen::VectorXd solveIK(dart::dynamics::SkeletonPtr skel, const std::string& bodyname, const Eigen::Vector3d& delta, const Eigen::Vector3d& offset)
{
	auto bn = skel->getBodyNode(bodyname);
	int foot_l_idx = skel->getBodyNode("FootL")->getParentJoint()->getIndexInSkeleton(0);
	int foot_r_idx = skel->getBodyNode("FootR")->getParentJoint()->getIndexInSkeleton(0);
	int footend_l_idx = skel->getBodyNode("FootEndL")->getParentJoint()->getIndexInSkeleton(0);
	int footend_r_idx = skel->getBodyNode("FootEndR")->getParentJoint()->getIndexInSkeleton(0);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);
	Eigen::VectorXd newPose = skel->getPositions();
	Eigen::Vector3d tp = delta;
	for(std::size_t i = 0; i < 1000; ++i)
	{
		Eigen::Vector3d deviation = tp - bn->getTransform()*offset;
		if(deviation.norm() < 0.001)
			break;
		// Eigen::Vector3d localCOM = bn->getCOM(bn);
		dart::math::LinearJacobian jacobian = skel->getLinearJacobian(bn, offset);
		jacobian.block<3,6>(0,0).setZero();
		// jacobian.block<3,3>(0,foot_l_idx).setZero();
		// jacobian.block<3,3>(0,foot_r_idx).setZero();
		jacobian.block<3,3>(0,footend_l_idx).setZero();
		jacobian.block<3,3>(0,footend_r_idx).setZero();
		// jacobian.block<3,2>(0,femur_l_idx+1).setZero();
		// jacobian.block<3,2>(0,femur_r_idx+1).setZero();
		// jacobian.block<3,2>(0,tibia_l_idx+1).setZero();
		// jacobian.block<3,2>(0,tibia_r_idx+1).setZero();

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3d inv_singular_value;
		
		inv_singular_value.setZero();
		for(int k=0;k<3;k++)
		{
			if(svd.singularValues()[k]==0)
				inv_singular_value(k,k) = 0.0;
			else
				inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
		}


		Eigen::MatrixXd jacobian_inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();

		// Eigen::VectorXd gradient = jacobian.colPivHouseholderQr().solve(deviation);
		Eigen::VectorXd gradient = jacobian_inv * deviation;
		double prev_norm = deviation.norm();
		double gamma = 0.5;
		for(int j = 0; j < 24; j++){
			Eigen::VectorXd newDirection = gamma * gradient;
			Eigen::VectorXd np = newPose + newDirection;
			skel->setPositions(np);
			skel->computeForwardKinematics(true, false, false);
			double new_norm = (tp - bn->getTransform()*offset).norm();
			if(new_norm < prev_norm){
				newPose = np;
				break;
			}
			gamma *= 0.5;
		}
	}
	return newPose;
}
Eigen::VectorXd solveMCIKRoot(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints)
{
	Eigen::VectorXd newPose = skel->getPositions();
	int num_constraints = constraints.size();

	std::vector<dart::dynamics::BodyNode*> bodynodes(num_constraints);
	std::vector<Eigen::Vector3d> targetposes(num_constraints);
	std::vector<Eigen::Vector3d> offsets(num_constraints);

	for(int i = 0; i < num_constraints; i++){
		bodynodes[i] = skel->getBodyNode(std::get<0>(constraints[i]));
		targetposes[i] = std::get<1>(constraints[i]);
		offsets[i] = std::get<2>(constraints[i]);
	}

	int not_improved = 0;
	for(std::size_t i = 0; i < 100; i++)
	{

		// make deviation vector and jacobian matrix
		Eigen::VectorXd deviation(num_constraints*3);
		for(int j = 0; j < num_constraints; j++){
			deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
		}
		if(deviation.norm() < 0.001)
			break;

		int nDofs = skel->getNumDofs();
		Eigen::MatrixXd jacobian_concatenated(3*num_constraints, nDofs);
		for(int j = 0; j < num_constraints; j++){
			dart::math::LinearJacobian jacobian = skel->getLinearJacobian(bodynodes[j], offsets[j]);
			jacobian.block(0, 0, 3, 1).setZero();
			jacobian.block(0, 2, 3, 1).setZero();
			jacobian.block(0, 6, 3, nDofs - 6).setZero();
			jacobian_concatenated.block(3*j, 0, 3, nDofs) = jacobian;
		}
		// std::cout << jacobian_concatenated << std::endl;

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian_concatenated, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd inv_singular_value(3*num_constraints, 3*num_constraints);
		
		inv_singular_value.setZero();
		for(int k=0;k<3*num_constraints;k++)
		{
			if(svd.singularValues()[k]<1e-8)
				inv_singular_value(k,k) = 0.0;
			else
				inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
		}


		Eigen::MatrixXd jacobian_inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();
		// std::cout << svd.singularValues().transpose() << std::endl;
		// std::cout << svd.matrixV().size() << std::endl;

		// std::cout << jacobian_inv << std::endl;
		// exit(0);
		// Eigen::VectorXd gradient = jacobian.colPivHouseholderQr().solve(deviation);
		Eigen::VectorXd gradient = jacobian_inv * deviation;
		double prev_norm = deviation.norm();
		double gamma = 0.5;
		not_improved++;
		for(int j = 0; j < 24; j++){
			Eigen::VectorXd newDirection = gamma * gradient;
			Eigen::VectorXd np = newPose + newDirection;
			skel->setPositions(np);
			skel->computeForwardKinematics(true, false, false);

			Eigen::VectorXd new_deviation(num_constraints*3);
			for(int j = 0; j < num_constraints; j++){
				new_deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
			}
			double new_norm = new_deviation.norm();
			if(new_norm < prev_norm){
				newPose = np;
				not_improved = 0;
				break;
			}
			gamma *= 0.5;
		}
		if(not_improved > 1){
			break;
		}
	}
	return newPose;
}
Eigen::VectorXd solveMCIK(dart::dynamics::SkeletonPtr skel, const std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>>& constraints)
{
	int foot_l_idx = skel->getBodyNode("FootL")->getParentJoint()->getIndexInSkeleton(0);
	int foot_r_idx = skel->getBodyNode("FootR")->getParentJoint()->getIndexInSkeleton(0);
	int footend_l_idx = skel->getBodyNode("FootEndL")->getParentJoint()->getIndexInSkeleton(0);
	int footend_r_idx = skel->getBodyNode("FootEndR")->getParentJoint()->getIndexInSkeleton(0);
	int femur_l_idx = skel->getBodyNode("FemurL")->getParentJoint()->getIndexInSkeleton(0);
	int femur_r_idx = skel->getBodyNode("FemurR")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_l_idx = skel->getBodyNode("TibiaL")->getParentJoint()->getIndexInSkeleton(0);
	int tibia_r_idx = skel->getBodyNode("TibiaR")->getParentJoint()->getIndexInSkeleton(0);

	Eigen::VectorXd newPose = skel->getPositions();
	int num_constraints = constraints.size();

	std::vector<dart::dynamics::BodyNode*> bodynodes(num_constraints);
	std::vector<Eigen::Vector3d> targetposes(num_constraints);
	std::vector<Eigen::Vector3d> offsets(num_constraints);

	for(int i = 0; i < num_constraints; i++){
		bodynodes[i] = skel->getBodyNode(std::get<0>(constraints[i]));
		targetposes[i] = std::get<1>(constraints[i]);
		offsets[i] = std::get<2>(constraints[i]);
	}

	int not_improved = 0;
	for(std::size_t i = 0; i < 100; i++)
	{

		// make deviation vector and jacobian matrix
		Eigen::VectorXd deviation(num_constraints*3);
		for(int j = 0; j < num_constraints; j++){
			deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
		}
		if(deviation.norm() < 0.001)
			break;

		int nDofs = skel->getNumDofs();
		Eigen::MatrixXd jacobian_concatenated(3*num_constraints, nDofs);
		for(int j = 0; j < num_constraints; j++){
			dart::math::LinearJacobian jacobian = skel->getLinearJacobian(bodynodes[j], offsets[j]);
			// jacobian.block<3,3>(0,0).setZero();
			// jacobian.block<3,3>(0,foot_l_idx).setZero();
			// jacobian.block<3,3>(0,foot_r_idx).setZero();
			jacobian.block<3,3>(0,footend_l_idx).setZero();
			jacobian.block<3,3>(0,footend_r_idx).setZero();
			// jacobian.block<3,2>(0,femur_l_idx+1).setZero();
			// jacobian.block<3,2>(0,femur_r_idx+1).setZero();
			jacobian.block<3,2>(0,tibia_l_idx+1).setZero();
			jacobian.block<3,2>(0,tibia_r_idx+1).setZero();

			jacobian_concatenated.block(3*j, 0, 3, nDofs) = jacobian;
		}
		// std::cout << jacobian_concatenated << std::endl;

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian_concatenated, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::MatrixXd inv_singular_value(3*num_constraints, 3*num_constraints);
		
		inv_singular_value.setZero();
		for(int k=0;k<3*num_constraints;k++)
		{
			if(svd.singularValues()[k]<1e-8)
				inv_singular_value(k,k) = 0.0;
			else
				inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
		}


		Eigen::MatrixXd jacobian_inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();
		// std::cout << svd.singularValues().transpose() << std::endl;
		// std::cout << svd.matrixV().size() << std::endl;

		// std::cout << jacobian_inv << std::endl;
		// exit(0);
		// Eigen::VectorXd gradient = jacobian.colPivHouseholderQr().solve(deviation);
		Eigen::VectorXd gradient = jacobian_inv * deviation;
		double prev_norm = deviation.norm();
		double gamma = 0.5;
		not_improved++;
		for(int j = 0; j < 24; j++){
			Eigen::VectorXd newDirection = gamma * gradient;
			Eigen::VectorXd np = newPose + newDirection;
			skel->setPositions(np);
			skel->computeForwardKinematics(true, false, false);

			Eigen::VectorXd new_deviation(num_constraints*3);
			for(int j = 0; j < num_constraints; j++){
				new_deviation.segment<3>(j*3) = targetposes[j] - bodynodes[j]->getTransform()*offsets[j];
			}
			double new_norm = new_deviation.norm();
			if(new_norm < prev_norm){
				newPose = np;
				not_improved = 0;
				break;
			}
			gamma *= 0.5;
		}
		if(not_improved > 1){
			break;
		}
	}
	return newPose;
}
std::vector<Eigen::VectorXd> Align(const std::vector<Eigen::VectorXd>& data,const Eigen::VectorXd& target) {
	std::vector<Eigen::VectorXd> result = data;

	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform(data[0].head<6>());
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform(target);

	
	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();

	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());			
	T01.linear() =  dart::math::expMapRot(DPhy::projectToXZ(p01));
	T01.translation()[1] = 0;
	Eigen::Isometry3d T0_gen = T01*T0_phase;

	for(int i = 0; i < data.size(); i++) {
		Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(data[i].head<6>());
		T_current = T0_phase.inverse()*T_current;
		T_current = T0_gen*T_current;

		result[i].head<6>() = dart::dynamics::FreeJoint::convertToPositions(T_current);
	}

	return result;
}
Eigen::Matrix3d projectToXZ(Eigen::Matrix3d m) {
	Eigen::AngleAxisd m_v;
	m_v = m;
	Eigen::Vector3d nearest = DPhy::NearestOnGeodesicCurve3d(Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 0), m_v.axis() * m_v.angle());
	Eigen::AngleAxisd nearest_aa(nearest.norm(), nearest.normalized());
	Eigen::Matrix3d result;
	result = nearest_aa;
	return result;

}
Eigen::Vector3d projectToXZ(Eigen::Vector3d v) {

	Eigen::Vector3d nearest = DPhy::NearestOnGeodesicCurve3d(Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 0, 0), v);
	return nearest;

}
Eigen::MatrixXd getPseudoInverse(Eigen::MatrixXd m){
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
	int min = std::min(m.cols(), m.rows());
	Eigen::MatrixXd inv_singular_value(min, min);
		
	inv_singular_value.setZero();
	for(int k=0;k<min;k++)
	{
		if(svd.singularValues()[k]<1e-8)
			inv_singular_value(k,k) = 0.0;
		else
			inv_singular_value(k,k) = 1.0/svd.singularValues()[k];
	}

	Eigen::MatrixXd inv = svd.matrixV()*inv_singular_value*svd.matrixU().transpose();
	return inv;
}
double doubleRand(double min, double max){
	double w = double(std::rand()/RAND_MAX);

	return (1-w) * min + w * max;
}
}