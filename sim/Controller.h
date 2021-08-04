#ifndef __CONTROLLER_H__
#define __CONTROLLER_H__
#include "dart/dart.hpp"
#include "BVH.h"
#include "CharacterConfigurations.h"
#include "Character.h"
#include "SkeletonBuilder.h"
#include "Functions.h"
#include "ReferenceManager.h"
#include <tuple>
#include <queue>
#include <Eigen/QR>
#include <fstream>
#include <numeric>
#include <algorithm>
namespace DPhy
{

class Controller
{
public:
	Controller(ReferenceManager* ref, std::string character_path, bool record=false, int id=0);

	
	void initPhysicsEnv();

	void Step();
	void Reset(bool RSI);

	Eigen::Isometry3d getReferenceTransform();
	Eigen::Vector3d projectOnVector(const Eigen::Vector3d& u, const Eigen::Vector3d& v);
	Eigen::VectorXd actuate(const Eigen::VectorXd& target_pos);



	std::vector<double> GetTrackingReward(Eigen::VectorXd& position, Eigen::VectorXd& position2, 
						Eigen::VectorXd& velocity, Eigen::VectorXd& velocity2, bool useVelocity);
	std::vector<double> GetHeadingReward();
	double GetParamReward();
	void UpdateReward();
	
	Eigen::VectorXd GetState();
	Eigen::VectorXd RecordPose();
	Eigen::VectorXd RecordVel();
	Eigen::VectorXd GetAgentFeature(){return this->mAgentFeatureSet;}
	Eigen::VectorXd GetExpertFeature(){return this->mExpertFeatureSet;}
	
	void UpdateTerminalInfo();
	bool IsTerminalState() {return this->mIsTerminal; }
	bool IsNanAtTerminal() {return this->mIsNanAtTerminal;}

	Eigen::VectorXd GetAgentParam(const Eigen::Matrix3d R_ref_inv);
	Eigen::VectorXd GetExpertParam();
	
	void SaveDisplayedData(std::string directory, bool bvh);
	void SetParam(const float new_theta, const float new_height, const float new_speed);
	void SetTarget(const Eigen::Vector3d& root_pos, bool random);


	const dart::simulation::WorldPtr& GetWorld() {return mWorld;}
	const dart::dynamics::SkeletonPtr& GetSkeleton() {return this->mCharacter->GetSkeleton();}

	int GetNumState() { return this->mNumState;}
	int GetNumAction() { return this->mNumAction;}
	int GetNumFeature() { return this->mNumFeature;}
	int GetNumPose() { return this->mNumPose;}
	double GetReward() {return mRewardParts[0]; }
	std::vector<std::string> GetRewardLabels() {return mRewardLabels; }

	void SetAction(const Eigen::VectorXd& action){ this->mActions = action; }

	double GetTimeElapsed(){return this->mTimeElapsed;}
	double GetCurrentFrame(){return this->mCurrentFrame;}
	double GetCurrentLength() {return this->mCurrentFrame - this->mStartFrame; }
	double GetStartFrame(){ return this->mStartFrame; }
	int GetTerminationReason() {return terminationReason; }
	std::vector<double> GetRewardByParts() {return mRewardParts; }



	Eigen::Vector3d GetTargetPosition(){return this->target_dir;}
	double GetTargetPositionLimit(){return this->mMaxTargetDist;}
	double GetAccessThreshold(){return this->dist_threshold;}
	double GetTargetSpeed(){return this->mTargetSpeed;}



protected:
	dart::dynamics::SkeletonPtr mGround;
	dart::simulation::WorldPtr mWorld;
	Character* mCharacter;
	ReferenceManager* mReferenceManager;
	int id;
	std::string mPath;

	double w_p,w_v,w_com,w_ee;


	int mSimPerCon;
	int mControlHz;
	int mSimulationHz;

	int mNumState, mNumAction,mNumFeature,mNumPose;
	int mNumMotions;
	Eigen::VectorXd mActions;
	std::vector<double> mRewardParts;

	double mCurrentFrameOnPhase;
	double mTrackingRewardTrajectory;


	
	double mTimeElapsed;
	int mStartFrame;
	int mCurrentFrame;
	int nTotalSteps;
	int mInterestedDof;


	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;

	Eigen::Vector3d mPPrevCOMvel;
	Eigen::Vector3d mPrevCOMvel;
	Eigen::Vector3d COM_prev;

	double mMass;
	bool mIsTerminal;
	bool mIsNanAtTerminal;
	int terminationReason;

	std::vector<std::string> mContacts;
	std::vector<std::string> mEndEffectors;
	std::vector<std::string> mMotionType;
	std::vector<std::string> mRewardLabels;


	double mAdaptiveStep;
	int mRewardDof;
	bool mRecord;

	Eigen::Vector3d mStartRoot; //middle of two feet at 0th frame

	int motion_it;

	Eigen::VectorXd mPPrevAgentPose;
	Eigen::VectorXd mPPrevAgentVel;
	Eigen::VectorXd mPPrevExpertPose;
	Eigen::VectorXd mPPrevExpertVel;
	
	Eigen::VectorXd mPrevAgentPose;
	Eigen::VectorXd mPrevAgentVel;
	Eigen::VectorXd mPrevExpertPose;
	Eigen::VectorXd mPrevExpertVel;
	Eigen::VectorXd mAgentFeatureSet;
	Eigen::VectorXd mExpertFeatureSet;

	int mNumMotionType;
	int mNumMotionParam;




	double target_speed;
	Eigen::Vector3d target_dir;

	float theta;
	float theta_prev;

	Eigen::Matrix3d target_rot;
	float height;
	float speed;

	double max_dist;
	double mMinTargetDist;
	double mMaxTargetDist;
	double dist_threshold;
	double mTargetSpeed;

};
}



#endif