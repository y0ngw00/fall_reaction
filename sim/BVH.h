/* CS3242 3D Modeling and Animation
 * Programming Assignment II
 * School of Computing
 * National University of Singapore
 */

#ifndef  _BVH_H_
#define  _BVH_H_
#include "GL/glut.h"
#include "dart/dart.hpp"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <string>
#include <map>

// precision used
#define _BVH_PRECISION_ 6

 
namespace DPhy
{
enum  ChannelEnum
{
	X_ROTATION, Y_ROTATION, Z_ROTATION,
	X_POSITION, Y_POSITION, Z_POSITION
};

//typedef unsigned int uint;

typedef struct {
	ChannelEnum type;
	uint index;
} CHANNEL;
typedef struct JOINT JOINT;

struct JOINT
{

	std::string name;                     // joint name
	JOINT* parent;                        // point to joint parent	
	Eigen::Vector3d offset;               // joint offset
	std::vector<CHANNEL*> channels;
	std::vector<JOINT*> children;         // joint's children	

	Eigen::Matrix3d mRotation;            // transformation stored by a translation and a quaternion (for animation)

	//Eigen::Matrix4d matrix;                     // transformation stored by 4x4 matrix (for reference only)


};

class BVH
{
private:
	JOINT* rootJoint;
    std::vector<float> initialoffset;
	bool load_success;
	std::map<std::string, JOINT*> JointList;
	std::map<std::string, JOINT*> mMapping;
	std::vector<CHANNEL*> bvh_channels;

    std::vector<Eigen::VectorXd> mMotions;
    int num_frames;
    float mTimestep;
    int num_motion_channels;
    Eigen::Vector3d mRootCOM;

public:
	BVH();
	BVH(std::string filename);
	~BVH();

private:
	// write a joint info to bvh hierarchy part
	void writeJoint(JOINT* joint, std::ostream& stream, int level);

	// clean up stuff, used in destructor
	void clear();

public:

	 // load a bvh file
	void load( std::string filename);

	// is the bvh file successfully loaded? 
	bool IsLoadSuccess() const { return load_success; }

public:
	const Eigen::Vector3d GetRootCOM(){return mRootCOM;}

	// get the JOINT pointer from joint name 
	JOINT* getJoint(std::string joint_name);
	Eigen::Matrix3d GetTransform(std::string joint_name);
	void SetMotion(double t);
	// get the pointer to mation data at frame_no
	// NOTE: frame_no is treated as frame_no % total_frames
	std::vector<Eigen::VectorXd> getMotionData() { return mMotions; }
	// get a pointer to the root joint 
	const JOINT* getRootJoint() const { return rootJoint; }
	// get the list of the joint
	const std::map<std::string, JOINT*> getJointList() const { return JointList; }
	// get the number of frames 
	unsigned GetNumFrames() const { return num_frames; }
	unsigned GetNumChannels() const { return num_motion_channels; }
	float GetTimestep() const { return mTimestep; }
    void AddMapping(const std::string& body_node,const std::string& bvh_node);
    std::vector<std::vector<BVH*>> getBVHContainer();


    void SetRotation(JOINT* joint, const Eigen::VectorXd& m_t);
	void SetRotation(JOINT* joint,const Eigen::Matrix3d& R_t);
	Eigen::Matrix3d GetRotation(JOINT* joint);
	Eigen::Matrix3d GetRotation(std::string joint_name);



};

};
#endif
