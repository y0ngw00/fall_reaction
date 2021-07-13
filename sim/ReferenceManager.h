#ifndef __REFERENCE_MANAGER_H__
#define __REFERENCE_MANAGER_H__
#include "Functions.h"
#include "Character.h"
#include "CharacterConfigurations.h"
#include "BVH.h"
//#include "MultilevelSpline.h"
//#include "RegressionMemory.h"
#include <tuple>
#include <mutex>
#include <tinyxml.h>
#include <fstream>
#include <stdlib.h>
#include <cmath>
namespace DPhy
{

class Motion
{
public:
	Motion(Motion* m) {
		position = m->position;
		velocity = m->velocity;
	}
	Motion(Eigen::VectorXd pos) {
		position = pos;
	}
	Motion(Eigen::VectorXd pos, Eigen::VectorXd vel) {
		position = pos;
		velocity = vel;
	}
	void SetPosition(Eigen::VectorXd pos) { position = pos; }
	void SetVelocity(Eigen::VectorXd vel) { velocity = vel; }

	Eigen::VectorXd GetPosition() { return position; }
	Eigen::VectorXd GetVelocity() { return velocity; }

protected:
	Eigen::VectorXd position;
	Eigen::VectorXd velocity;

};

class ReferenceManager
{
public:
	ReferenceManager(Character* character=nullptr);
	
	void LoadMotionFromBVH(std::string filename);
	void LoadMotionPairFromBVH(std::string filename);
	void GenerateMotionsFromSinglePhase(int frames, bool blend, std::vector<Motion*>& p_phase, std::vector<std::vector<Motion*>>& p_container);
	void SelectMotion(int i);
	int GetNumMotions() {return this->mNumMotions;}
	Motion* GetMotion(double t);
	std::vector<Motion*> GetMotions(int id){return this->mMotions_container[id]; }

	Eigen::VectorXd GetPosition(double t);
	double GetTimeStep(double t);
	int GetPhaseLength() {return mPhaseLength; }
	int GetPhaseLength(int id) {return this->mMotionPhases[id]; }
	int GetDOF() {return mDOF;}
	std::vector<double> GetContacts(double t);

	void ResetOptimizationParameters(bool reset_displacement=true);
	void setRecord(){mRecord = true;}

	int GetNumPose() { return this->mNumPose;}
	void SetRandomTarget(const Eigen::Vector3d& root_pos);

	Eigen::Vector3d GetTargetPosition(){return this->target_pos;}
	double GetTargetPositionLimit(){return this->mMaxTargetDist;}
	double GetAccessThreshold(){return this->dist_threshold;}
	double GetTargetSpeed(){return this->mTargetSpeed;}

		



protected:
	Character* mCharacter;
	dart::dynamics::SkeletonPtr skel;

	int mDOF;
	int mPhaseLength;
	int smooth_time;
	int mNumMotions;
	int mFramePerMotion;
	double mTimeStep;

	double mSlaves;
	std::mutex mLock;

	int mBlendingInterval;
	bool mRecord;

	double mMeanTrackingReward;
	double mMeanParamReward;

	Eigen::VectorXd mMotionPhases;


	std::vector<std::string> contact;
	std::vector<std::vector<bool>> mContacts;
	
	std::vector<Motion*> mMotions_raw;
	std::vector<Motion*> mMotions_phase;
	std::vector<Motion*> mMotions_gen;
	std::vector<std::vector<Motion*>> mMotions_container;
	std::vector<Motion*> mMotions_gen_adaptive;


	int mNumPose;
	int motion_it;
	std::vector<std::string> mEndEffectors;

	Eigen::Vector3d target_pos;
	double max_dist;
	double mMaxTargetDist;
	double dist_threshold;
	double mTargetSpeed;



};

};

#endif