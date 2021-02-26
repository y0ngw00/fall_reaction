/* CS3242 3D Modeling and Animation
 * Programming Assignment II
 * School of Computing
 * National University of Singapore
 */

#include "BVH.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
using namespace std;


// JOINT *Head, *Neck, *Spine, *Spine1, *Spine2, *Hips;
// JOINT *LeftShoulder, *LeftArm, *LeftForeArm, *LeftHand;
// JOINT *RightShoulder, *RightArm, *RightForeArm, *RightHand;
// JOINT *LeftUpLeg, *LeftLeg, *LeftFoot, *LeftToe;
// JOINT *RightUpLeg, *RightLeg, *RightFoot, *RightToe;

static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

Eigen::Matrix3d
R_x(double x)
{
	double cosa = cos(x*3.141592/180.0);
	double sina = sin(x*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	1,0		,0	  ,
		0,cosa	,-sina,
		0,sina	,cosa ;
	return R;
}
Eigen::Matrix3d R_y(double y)
{
	double cosa = cos(y*3.141592/180.0);
	double sina = sin(y*3.141592/180.0);
	Eigen::Matrix3d R;
	R <<cosa ,0,sina,
		0    ,1,   0,
		-sina,0,cosa;
	return R;	
}
Eigen::Matrix3d R_z(double z)
{
	double cosa = cos(z*3.141592/180.0);
	double sina = sin(z*3.141592/180.0);
	Eigen::Matrix3d R;
	R<<	cosa,-sina,0,
		sina,cosa ,0,
		0   ,0    ,1;
	return R;		
}

 
namespace DPhy
{

BVH::BVH()
{
	load_success = false;
}

BVH::BVH(std::string filename)
{
	load_success = false;
	load(filename);
    //setPointers();
}

BVH::~BVH()
{
	clear();
}

void BVH::clear()
{
	JointList.clear();
	bvh_channels.clear();
	load_success = false;
	rootJoint = NULL;
}


void
BVH::
SetRotation(JOINT* joint, const Eigen::VectorXd& m_t)
{
	joint->mRotation.setIdentity();
	for(int i=0;i<joint->channels.size();i++)
	{
		CHANNEL *channel = joint->channels[i];
		//std::cout<<joint->name<<"'s location : "<<channel->index<<std::endl;
		if (channel->type == X_ROTATION)
			joint->mRotation = joint->mRotation*R_x(m_t[channel->index]);
		else if (channel->type == Y_ROTATION)
			joint->mRotation = joint->mRotation*R_y(m_t[channel->index]);
		else if (channel->type == Z_ROTATION)
			joint->mRotation = joint->mRotation*R_z(m_t[channel->index]);
	}
}
void
BVH::
SetRotation(JOINT* joint, const Eigen::Matrix3d& R_t)
{
	joint->mRotation = R_t;
}
Eigen::Matrix3d
BVH::
GetRotation(JOINT* joint)
{
	return joint->mRotation;
}

Eigen::Matrix3d
BVH::
GetRotation(std::string joint_name)
{
	JOINT* b = getJoint(joint_name);
	return b->mRotation;
}


void BVH::load(std::string bvh_file_name)
{
	#define BUFFER_LENGTH 1024*4
	std::ifstream file;
	char buffer[BUFFER_LENGTH];

	std::vector<JOINT*> joint_stack;
	JOINT* joint = NULL;
	JOINT* new_joint = NULL;
	double x, y, z;
	clear();

	file.open(bvh_file_name, std::ios::in);
	if (!file.is_open()) return;

	// Read Hierarchy of Skeleton
	while (file>>buffer)
	{
        if ( !strcmp( buffer, "{" ) )
        {
            joint_stack.push_back( joint );
            joint = new_joint;
        }
        else if ( !strcmp( buffer, "}" ) )
        {
            joint = joint_stack.back();
            joint_stack.pop_back();
        }
		else if (!strcmp(buffer, "ROOT") || !strcmp(buffer, "JOINT") )
		{
			new_joint = new JOINT();
			new_joint->parent = joint;

			if (joint)
				joint->children.push_back(new_joint);
			else
				rootJoint = new_joint; //the root
			
			// Read joint name
			file>>buffer;
            string joint_name =  std::string(buffer);
            
            //rtrim(joint_name);
            new_joint->name = joint_name;
			JointList[new_joint->name] = new_joint;
		}

		else if ( !strcmp(buffer, "End"))
		{
			new_joint = new JOINT();
			new_joint->parent = joint;

			new_joint->channels.clear();
			//add children 
			if (joint)
				joint->children.push_back(new_joint);
			else
				rootJoint = new_joint; //can an endsite be root? -cuizh
			//add to joint collection
			JointList[new_joint->name] = new_joint;
		}

		else if ( !strcmp(buffer, "OFFSET") )
		{
			file>>x;
			file>>y;
			file>>z;
			joint->offset[0] = x;
			joint->offset[1] = y;
			joint->offset[2] = z;
			continue;
		}

		else if ( !strcmp(buffer, "CHANNELS" ) )
		{
			//The number of channels(DOF of joint)
			file>>buffer; 
			
			joint->channels.resize(buffer ? atoi(buffer) : 0);

			for (uint i = 0; i < joint->channels.size(); i++)
			{
				CHANNEL* channel = new CHANNEL();
				
				channel->index = bvh_channels.size();
				bvh_channels.push_back(channel);
				
				joint->channels[i] = channel;

				file>>buffer;
                string param =  std::string(buffer);
                //rtrim(param);
                const char* param_ch = param.c_str();
				if (strcmp(param_ch, "Xrotation") == 0)
					channel->type = X_ROTATION;
				else if (strcmp(param_ch, "Yrotation") == 0)
					channel->type = Y_ROTATION;
				else if (strcmp(param_ch, "Zrotation") == 0)
					channel->type = Z_ROTATION;
				else if (strcmp(param_ch, "Xposition") == 0)
					channel->type = X_POSITION;
				else if (strcmp(param_ch, "Yposition") == 0)
					channel->type = Y_POSITION;
				else if (strcmp(param_ch, "Zposition") == 0)
					channel->type = Z_POSITION;


				joint->channels[i] = channel;
			}
		}

		else if ( !strcmp(buffer, "MOTION") )
			break;
	}

    num_motion_channels = bvh_channels.size();

	// Read Motion data
    file>>buffer;
    if ( strcmp( buffer, "Frames:" ))  goto bvh_error;
    
    // The number of frame
    file>>buffer;
    num_frames = atoi(buffer);
    mMotions.resize(num_frames);

    file>>buffer;
    file>>buffer;
    if (strcmp( buffer, "Time:" ))  goto bvh_error;    
    file>>buffer;
    mTimestep = atof( buffer );

    for(auto& m_t : mMotions)
		m_t = Eigen::VectorXd::Zero(num_motion_channels);

    for (uint i=0; i<num_frames; i++)
    {
    	for ( uint j=0; j<num_motion_channels; j++ )
    	{
    		file>>buffer;
    		mMotions[i][j] = atof(buffer);
    	}
    }

    file.close();
	load_success = true;
	std::cout<<"Bvh File is loaded"<<std::endl;
	std::cout<<"The number of frames : "<<num_frames<<std::endl;
	std::cout<<"The time step : "<<mTimestep<<std::endl;
	std::cout<<std::endl;
	return;

bvh_error:
	std::cout<<"This bvh file is corrupt"<<std::endl;
	file.close();
}
void
BVH::
SetMotion(double t)
{
	// Define the motion between frame t and t+1 with linear interpolation.
	int k = ((int)std::floor(t/mTimestep))%num_frames;
	int k1 = std::min(k+1, num_frames - 1);
	double dt = (t/mTimestep - std::floor(t/mTimestep));
	std::vector<Eigen::Matrix3d> R_0,R_1;

	//R0
	for(auto& bn: mMapping)
	{
		SetRotation(bn.second,mMotions[k]);
	}
	for(auto& bn:mMapping)
		R_0.push_back(GetRotation(bn.second));
	//R1
	for(auto& bn: mMapping)
	{
		SetRotation(bn.second,mMotions[k1]);
	}

	for(auto& bn:mMapping)
		R_1.push_back(GetRotation(bn.second));

	//slerp
	int count = 0;
	for(auto& bn:mMapping)
	{
		Eigen::Matrix3d exp_w = (R_0[count].transpose())*R_1[count];
		Eigen::AngleAxisd w(exp_w);
		Eigen::Matrix3d R_t = R_0[count]*dart::math::expMapRot(dt*w.angle()*w.axis());
		SetRotation(bn.second,R_t);
		count++;
	}

	Eigen::Vector3d root_k = mMotions[k].segment<3>(0);
	Eigen::Vector3d root_k1 = mMotions[k1].segment<3>(0);

	mRootCOM.setZero();
	mRootCOM = (root_k*(1-dt) + root_k1*dt - mRootCOM)*0.01;
	
}

std::vector<std::vector<BVH*>>
BVH::
getBVHContainer() {

	std::vector<std::vector<BVH*>> bvh_container;
	BVH* bvh_start;
	vector<BVH*> bvh_normal, bvh_fast;


    bvh_normal.clear();
    bvh_start = new BVH("../Motionbvh/mrl/tpose2.bvh");
    bvh_container.push_back({bvh_start});

    BVH* bvh_walk = new BVH("../Motionbvh/cmu/16_21_walk.bvh");
    bvh_normal.push_back(bvh_walk);
    BVH* bvh_rturn= new BVH("../Motionbvh/cmu/16_13_walk, veer right.bvh");
    bvh_normal.push_back(bvh_rturn);
    BVH* bvh_lturn = new BVH("../Motionbvh/cmu/16_11_walk, veer left.bvh");
    bvh_normal.push_back(bvh_lturn);

    bvh_container.push_back(bvh_normal);

    BVH* bvh_fwalk = new BVH("../Motionbvh/cmu/16_35_run&jog.bvh");
    bvh_fast.push_back(bvh_fwalk);
    BVH* bvh_frturn = new BVH("../Motionbvh/cmu/16_39_run&jog, veer right.bvh");
    bvh_fast.push_back(bvh_frturn);
    BVH* bvh_flturn = new BVH("../Motionbvh/cmu/16_37_run&jog, veer left.bvh");
    bvh_fast.push_back(bvh_flturn);

    bvh_container.push_back(bvh_fast);
//    {
//        std::vector<std::unique_ptr<BVH>> bvhs = std::make_unique<BVH>();
//    }
    return bvh_container; 
}

JOINT* BVH::getJoint(std::string joint_name)
{
	std::map<std::string, JOINT*>::const_iterator  i = JointList.find(joint_name);
	JOINT* j = (i != JointList.end()) ? (*i).second : NULL;
	if (j == NULL) {
		std::map<std::string, JOINT*>::const_iterator  u = mMapping.find(joint_name);
		JOINT* v = (u != mMapping.end()) ? (*u).second : NULL;
		if (v==NULL)
			std::cout << "JOINT <" << joint_name << "> is not loaded!\n";
		j = v;
	}
	return j;
}


void
BVH::
AddMapping(const std::string& body_node,const std::string& bvh_node)
{	
	JOINT* b = getJoint(bvh_node);
	mMapping.insert(std::make_pair(body_node,b));
}

 
};