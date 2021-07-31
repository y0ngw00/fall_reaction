#include "ReferenceManager.h"

using namespace dart::dynamics;
namespace DPhy
{

ReferenceManager::
ReferenceManager(Character* character)
{
	this-> mCharacter = character;
	this-> skel = mCharacter->GetSkeleton();
	this-> mDOF = skel->getPositions().rows();
	this-> mBlendingInterval = 10;

	this-> smooth_time = 10;
	this->mFramePerMotion = 400;

	
	mMotions_raw.clear();
	mMotions_gen.clear();
	mMotions_phase.clear();
	mMotions_container.clear();

	motion_list.clear();
	motion_type.clear();

	contact.clear();
	contact.push_back("RightToe");
	contact.push_back("RightFoot");
	contact.push_back("LeftToe");
	contact.push_back("LeftFoot");

	mEndEffectors.clear();
	mEndEffectors.push_back("RightFoot");
	mEndEffectors.push_back("LeftFoot");
	mEndEffectors.push_back("LeftHand");
	mEndEffectors.push_back("RightHand");

}

int
ReferenceManager::
GetMotionIndex(std::string motion_name){
	int it = std::find(this->motion_list.begin(), this->motion_list.end(), motion_name) - this->motion_list.begin();
	return it;
}

void
ReferenceManager::
GetMotionInfo(std::string file_path, std::vector<std::string>& motiontype,std::vector<std::string>& motionlist){
	char buffer[100];

	std::ifstream txtread;
	txtread.open(file_path);

	std::vector<Eigen::VectorXd> param_list;	

	while(txtread>>buffer){
		if ( !strcmp( buffer, "MOTION_TYPE" ) )
        {
            txtread>>buffer;
            this->mNumMotionType = atoi(buffer);
        }
        else{
        	motionlist.push_back(std::string(buffer));

			txtread>>buffer;
			motiontype.push_back(std::string(buffer));
        }	
	}
	txtread.close();
	return;
}

void
ReferenceManager::
LoadMotionFromBVH(std::string filename)
{
	
	mMotions_gen.clear();
	mMotions_container.clear();
	std::string txt_path = std::string(PROJECT_DIR) + filename;

	std::ifstream txtread;
	txtread.open(txt_path);
	if(!txtread.is_open()){
		std::cout<<"Text file does not exist from : "<< txt_path << std::endl;
		return;
	}
	txtread.close();
	
	GetMotionInfo(txt_path, this->motion_type,this->motion_list);
	std::cout<<"The Number of Motion Files : "<<motion_list.size()<<std::endl;
	this->mNumMotions=motion_list.size();
	this->mMotionPhases.resize(mNumMotions);
	this->mMotionPhases.setZero();

	int it=0;
	int num_pose =0;

	std::map<std::string,std::string> bvhMap = mCharacter->GetSkelMap(); 

	for(auto p :motion_list){
		
		mMotions_raw.clear();
		mMotions_phase.clear();

		std::string path = std::string(PROJECT_DIR) + "/motion/"+p;

		BVH* bvh = new DPhy::BVH(path);
		if(!bvh->IsLoadSuccess()){
			std::cout<<"Loading bvh is failed from : "<< path << std::endl;
			return;
		}

		std::cout << "load trained data from: " << path << std::endl;

		int dof = this->mDOF;
		
		for(auto jnt :bvhMap){
			bvh->AddMapping(jnt.first,jnt.second);  //first = xml body node, second = bvh node
		}


		double t = 0;
		num_pose += bvh->GetNumFrames();
		for(int i = 0; i < bvh->GetNumFrames(); i++)
		{
			Eigen::VectorXd pos = Eigen::VectorXd::Zero(dof);
			//Set p
			bvh->SetMotion(t);

			for(auto jnt :bvhMap)
			{
				//get Bodynode and Transform

				dart::dynamics::BodyNode* bn = this->skel->getBodyNode(jnt.first);
				Eigen::Matrix3d R = bvh->GetRotation(jnt.second);

				//get Joint 
				dart::dynamics::Joint* jn = bn->getParentJoint();
				Eigen::Vector3d a = dart::dynamics::BallJoint::convertToPositions(R);

				//?
				a = QuaternionToDARTPosition(DARTPositionToQuaternion(a));

				//The joint is a ball joint or free joint,
				if(dynamic_cast<dart::dynamics::BallJoint*>(jn)!=nullptr
					|| dynamic_cast<dart::dynamics::FreeJoint*>(jn)!=nullptr){
					pos.segment<3>(jn->getIndexInSkeleton(0)) = a;  // insert euler angles of the joint
				}


				//The joint is a revolute joint,
				else if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){ // 
					if(jnt.first.find("Arm") != std::string::npos)
						pos[jn->getIndexInSkeleton(0)] = a[1];  // In the case of Arm, insert angle at Y-axis
					else	
						pos[jn->getIndexInSkeleton(0)] = a[0];	// Else, insert angle at X-axis

					if(pos[jn->getIndexInSkeleton(0)]>M_PI)      // Set angle between 0 and 360
						pos[jn->getIndexInSkeleton(0)] -= 2*M_PI;
					else if(pos[jn->getIndexInSkeleton(0)]<-M_PI)
						pos[jn->getIndexInSkeleton(0)] += 2*M_PI;
				}

			}


			pos.segment<3>(3) = bvh->GetRootCOM();  		// Insert COM position 
			Eigen::VectorXd v;

			if(t != 0)
			{
				//
				v = skel->getPositionDifferences(pos, mMotions_raw.back().GetPosition()) / 0.033;
				for(auto& jn : skel->getJoints()){
					if(dynamic_cast<dart::dynamics::RevoluteJoint*>(jn)!=nullptr){
						double v_ = v[jn->getIndexInSkeleton(0)];
						if(v_ > M_PI){
							v_ -= 2*M_PI;
						}
						else if(v_ < -M_PI){
							v_ += 2*M_PI;
						}
						v[jn->getIndexInSkeleton(0)] = v_;
					}
				}
				mMotions_raw.back().SetVelocity(v);
			}
			mMotions_raw.push_back(Motion(pos, Eigen::VectorXd(pos.rows())));
			
			skel->setPositions(pos);
			//skel->computeForwardKinematics(true,false,false);

			std::vector<bool> c;
			for(int j = 0; j < contact.size(); j++) {

				Eigen::Vector3d p = skel->getBodyNode(contact[j])->getWorldTransform().translation();

				c.push_back(p[1] < 0.04);
			}


			t += bvh->GetTimestep();

		}

		mMotions_raw.back().SetVelocity(mMotions_raw.front().GetVelocity());

		mPhaseLength = mMotions_raw.size();
		mTimeStep = bvh->GetTimestep();
		this->mMotionPhases[it]=mPhaseLength;

		for(int i = 0; i < mPhaseLength; i++) {
			mMotions_phase.push_back(Motion(mMotions_raw[i]));
		 }
		 
		delete bvh;
		this->GenerateMotionsFromSinglePhase(this->mFramePerMotion, true, mMotions_phase, this->mMotions_container);
		it++;
	}
	SelectMotion(0);

	this->mNumPose = num_pose;


}

void 
ReferenceManager::
GenerateMotionsFromSinglePhase(int frames, bool blend, std::vector<Motion>& p_phase, std::vector<std::vector<Motion>>& p_container)
{
	mLock.lock();
	std::vector<Motion> p_gen;

	// p_gen clear?
	// while(!p_gen.empty()){
	// 	Motion m = p_gen.back();
	// 	p_gen.pop_back();
	// 	delete m;
	// }		

	dart::dynamics::SkeletonPtr skel = mCharacter->GetSkeleton();

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	
	skel->setPositions(p_phase[0].GetPosition());
	skel->computeForwardKinematics(true,false,false);

	Eigen::Vector3d p0_footl = skel->getBodyNode("LeftFoot")->getWorldTransform().translation();
	Eigen::Vector3d p0_footr = skel->getBodyNode("RightFoot")->getWorldTransform().translation();

	//Start pose
	Eigen::Isometry3d T0_phase = dart::dynamics::FreeJoint::convertToTransform(p_phase[0].GetPosition().head<6>());
	//End pose
	Eigen::Isometry3d T1_phase = dart::dynamics::FreeJoint::convertToTransform(p_phase.back().GetPosition().head<6>());

	Eigen::Isometry3d T0_gen = T0_phase;
	
	//Displacement
	Eigen::Isometry3d T01 = T1_phase*T0_phase.inverse();


	Eigen::Vector3d p01 = dart::math::logMap(T01.linear());			
	T01.linear() = dart::math::expMapRot(DPhy::projectToXZ(p01));
	T01.translation()[1] = 0;

	
	for(int i = 0; i < frames; i++) {
		
	 	int phase = i % mPhaseLength;
		
	 	if(i < mPhaseLength) {
	 		p_gen.push_back(Motion(p_phase[i]));
	 	} 
	 	else {
	 		Eigen::VectorXd pos;
			if(phase == 0) { //The End of unit motion
				std::vector<std::tuple<std::string, Eigen::Vector3d, Eigen::Vector3d>> constraints;
				
				skel->setPositions(p_gen.back().GetPosition());
				//skel->computeForwardKinematics(true,false,false);

				// Align the orientation of next motion
				Eigen::VectorXd p = p_phase[phase].GetPosition();
				p.segment<3>(3) = p_gen.back().GetPosition().segment<3>(3);
				skel->setPositions(p);
				//skel->computeForwardKinematics(true,false,false);

				pos = p;
				T0_gen = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());

			} 
			else { 
				pos = p_phase[phase].GetPosition();
				Eigen::Isometry3d T_current = dart::dynamics::FreeJoint::convertToTransform(pos.head<6>());
				// The orientation displacement to blend
				Eigen::Isometry3d T0_phase_gen = T0_gen* T0_phase.inverse();

				// Blending motion
				if(phase < smooth_time){
					Eigen::Quaterniond Q0_phase_gen(T0_phase_gen.linear());
					double slerp_t = (double)phase/smooth_time; 
					slerp_t = 0.5*(1-cos(M_PI*slerp_t)); //smooth slerp t [0,1]
					
					Eigen::Quaterniond Q_blend = Q0_phase_gen.slerp(slerp_t, Eigen::Quaterniond::Identity());
					T0_phase_gen.linear() = Eigen::Matrix3d(Q_blend);
					T_current = T0_phase_gen* T_current;
				}
				else{
					// Not blend motion. Actually this is not neccessary
					T0_phase_gen.linear() = Eigen::Matrix3d::Identity(); 
					T_current = T0_phase_gen* T_current;
				}

				pos.head<6>() = dart::dynamics::FreeJoint::convertToPositions(T_current);
			}

			Eigen::VectorXd vel = skel->getPositionDifferences(pos, p_gen.back().GetPosition()) / 0.033;
			p_gen.back().SetVelocity(vel);
			p_gen.push_back(Motion(pos, vel));

			// No blending 
			if(blend && phase == 0) {
				for(int j = mBlendingInterval; j > 0; j--) {
					double weight = 1.0 - j / (double)(mBlendingInterval+1);
					Eigen::VectorXd oldPos = p_gen[i - j].GetPosition();
					p_gen[i - j].SetPosition(DPhy::BlendPosition(oldPos, pos, weight));
					vel = skel->getPositionDifferences(p_gen[i - j].GetPosition(), p_gen[i - j - 1].GetPosition()) / 0.033;
			 		p_gen[i - j - 1].SetVelocity(vel);
				}
			}
		}
	}
	p_container.push_back(p_gen);
	mLock.unlock();



}

void 
ReferenceManager::
SelectMotion(int i){
	mMotions_gen = this->mMotions_container[i];
}


Motion
ReferenceManager::
GetMotion(double t)
{

	dart::dynamics::SkeletonPtr skel = mCharacter->GetSkeleton();

	if(mMotions_gen.size()-1 < t) {
	 	return mMotions_gen.back();
	}
	
	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	

	if (k0 == k1)
		return mMotions_gen[k0];
	else {
		return Motion(DPhy::BlendPosition((mMotions_gen)[k1].GetPosition(), (mMotions_gen)[k0].GetPosition(), 1 - (t-k0)), 
				DPhy::BlendVelocity((mMotions_gen)[k1].GetVelocity(), (mMotions_gen)[k0].GetVelocity(), 1 - (t-k0)));		
	}
}
Eigen::VectorXd 
ReferenceManager::
GetPosition(double t) 
{
	auto& skel = mCharacter->GetSkeleton();

	if((mMotions_gen).size()-1 < t) {
	 	return (mMotions_gen).back().GetPosition();
	}
	
	int k0 = (int) std::floor(t);
	int k1 = (int) std::ceil(t);	
	if (k0 == k1)
		return (mMotions_gen)[k0].GetPosition();
	else
		return DPhy::BlendPosition((mMotions_gen)[k1].GetPosition(), (mMotions_gen)[k0].GetPosition(), 1 - (t-k0));	
}

double 
ReferenceManager::
GetTimeStep(double t) {
	// if(adaptive) {
	// 	t = std::fmod(t, mPhaseLength);
	// 	int k0 = (int) std::floor(t);
	// 	int k1 = (int) std::ceil(t);	
	// 	if (k0 == k1) {
	// 		return mTimeStep_adaptive[k0];
	// 	}
	// 	else if(k1 >= mTimeStep_adaptive.size())
	// 		return (1 - (t - k0)) * mTimeStep_adaptive[k0] + (t-k0) * mTimeStep_adaptive[0];
	// 	else
	// 		return (1 - (t - k0)) * mTimeStep_adaptive[k0] + (t-k0) * mTimeStep_adaptive[k1];
	// } else 
		return 1.0;
}

}
