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
	Controller(ReferenceManager* ref, std::string character_path, bool record=false, int id=0,bool test=false);

	
	void initPhysicsEnv();

	void Step();



	const dart::simulation::WorldPtr& GetWorld() {return mWorld;}
	const dart::dynamics::SkeletonPtr& GetSkeleton() {return this->mCharacter->GetSkeleton();}

	std::vector<double> GetTrackingReward(Eigen::VectorXd position, Eigen::VectorXd position2, 
						Eigen::VectorXd velocity, Eigen::VectorXd velocity2, bool useVelocity);
	std::vector<double> GetRecoveryReward(Eigen::VectorXd position,Eigen::VectorXd position2);
	std::vector<double> GetBreakfallReward(Eigen::VectorXd position);
	double GetParamReward();
	void UpdateReward();
	double GetReward() {return mRewardParts[0]; }
	std::vector<std::string> GetRewardLabels() {return mRewardLabels; }

	Eigen::VectorXd GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel);
	Eigen::VectorXd GetState();

	bool CheckCollisionWithGround(std::string bodyName);

	void UpdateTerminalInfo();
	void SaveStepInfo();
	void ClearRecord();
	bool IsTerminalState() {return this->mIsTerminal; }
	bool IsNanAtTerminal() {return this->mIsNanAtTerminal;}

	bool FollowBvh();
	void SetSkeletonWeight(double mass);
	void Reset(bool RSI);


	int GetNumState() { return this->mNumState;}
	int GetNumAction() { return this->mNumAction;}
	void SetAction(const Eigen::VectorXd& action){ this->mActions = action; }

	double GetTimeElapsed(){return this->mTimeElapsed;}
	double GetCurrentFrame(){return this->mCurrentFrame;}
	double GetCurrentLength() {return this->mCurrentFrame - this->mStartFrame; }
	double GetStartFrame(){ return this->mStartFrame; }
	int GetTerminationReason() {return terminationReason; }

	Eigen::VectorXd GetPositions(int idx) { return this->mRecordPosition[idx]; }
	Eigen::Vector3d GetCOM(int idx) { return this->mRecordCOM[idx]; }
	Eigen::VectorXd GetVelocities(int idx) { return this->mRecordVelocity[idx]; }
	double GetPhase(int idx) { return this->mRecordPhase[idx]; }
	Eigen::VectorXd GetTargetPositions(int idx) { return this->mRecordTargetPosition[idx]; }
	Eigen::VectorXd GetBVHPositions(int idx) { return this->mRecordBVHPosition[idx]; }
	int GetRecordSize() { return this->mRecordPosition.size(); }
	std::pair<bool, bool> GetFootContact(int idx) { return this->mRecordFootContact[idx]; }
	std::vector<double> GetRewardByParts() {return mRewardParts; } 

	void SetRandomForce();
	void SaveDisplayedData(std::string directory, bool bvh);

	Eigen::Vector3d GetForceDir(){return this->ext_dir;}
	double GetForceSize() {return this->ext_force;}
	int GetForceFrame() {return this->mHitFrame;}



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

	int mNumState, mNumAction;
	Eigen::VectorXd mActions;
	std::vector<double> mRewardParts;

	double mCurrentFrameOnPhase;
	double mPrevFrameOnPhase;
	double mTrackingRewardTrajectory;


	
	double mTimeElapsed;
	int mStartFrame;
	int mCurrentFrame;
	int nTotalSteps;
	int mInterestedDof;

	double mPrevFrame;
	Eigen::VectorXd mTlPrev;

	Eigen::VectorXd mPrevTargetPositions;

	Eigen::VectorXd mTargetPositions;
	Eigen::VectorXd mTargetVelocities;

	Eigen::VectorXd mPDTargetPositions;
	Eigen::VectorXd mPDTargetVelocities;

	double mMass;
	bool mIsTerminal;
	bool mIsNanAtTerminal;
	int terminationReason;

	std::tuple<double, double, double> mRescaleParameter;
	std::vector<std::string> mContacts;
	std::vector<std::string> mEndEffectors;
	std::vector<std::string> mRewardLabels;

	std::unique_ptr<dart::collision::CollisionGroup> mCGEL, mCGER, mCGL, mCGR, mCGG, mCGHR, mCGHL, mCGOBJ; 

	std::unique_ptr<dart::collision::CollisionGroup> mCGAL, mCGAR, mCGH, mCL,mCR; 


	double mAdaptiveStep;
	int mRewardDof;
	bool mRecord;

	std::vector<Eigen::VectorXd> mRecordPosition;
	std::vector<Eigen::VectorXd> mRecordVelocity;
	std::vector<Eigen::Vector3d> mRecordCOM;
	std::vector<Eigen::VectorXd> mRecordTargetPosition;
	std::vector<Eigen::VectorXd> mRecordBVHPosition;
	std::vector<double> mRecordPhase;
	std::vector<std::pair<bool, bool>> mRecordFootContact;

	Eigen::Vector6d mRootZero;
	Eigen::Vector6d mDefaultRootZero;
	Eigen::Vector3d mStartRoot; //root 0th frame
	Eigen::Vector3d mRootZeroDiff; //root 0th frame
	Eigen::Vector3d mStartFoot; //middle of two feet at 0th frame

	bool mtest;
	bool isHit=false;
	int mHitFrame;
	int ext_force;
	Eigen::Vector3d ext_dir;
	std::vector<std::string> mTargetBody;

	std::vector<std::string> mFallGuard;
	std::vector<std::string> mFallProtect;


};
}



#endif