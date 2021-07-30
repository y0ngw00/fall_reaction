
#include "Controller.h"
using namespace dart::dynamics;
namespace DPhy
{

Controller::Controller(ReferenceManager* ref, std::string character_path, bool record, int id)
	:mControlHz(30),mSimulationHz(300),mCurrentFrame(0),
	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{
	initPhysicsEnv();

	this->mRescaleParameter = std::make_tuple(1.0, 1.0, 1.0);
	this->mRecord = record;
	this->mReferenceManager = ref;
	this->mNumMotions = mReferenceManager->GetNumMotions();
	this->id = id;

	this->mCharacter = new DPhy::Character(character_path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());
	this->mPath = character_path;

	this->mMass = mCharacter->GetSkeleton()->getMass();

	this->mSimPerCon = mSimulationHz / mControlHz;
	
	this->mCurrentFrame = 0;
	this->mCurrentFrameOnPhase = 0;
	this->motion_it=0;
 	this->target_pos = {0,0,1};


	Eigen::VectorXd kp(this->mCharacter->GetSkeleton()->getNumDofs());
	Eigen::VectorXd kv(this->mCharacter->GetSkeleton()->getNumDofs());
	kp.setZero();
	kv.setZero();
	this->mCharacter->SetPDParameters(kp,kv);

	mContacts.clear();
	mContacts.push_back("RightToe");
	mContacts.push_back("RightFoot");
	mContacts.push_back("LeftToe");
	mContacts.push_back("LeftFoot");

	mInterestedDof = mCharacter->GetSkeleton()->getNumDofs()-6;
	mRewardDof = mCharacter->GetSkeleton()->getNumDofs();

	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	this->mCGL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("LeftFoot"));
	this->mCGR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("RightFoot"));
	this->mCGEL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("LeftToe"));
	this->mCGER = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("RightToe"));
	this->mCGHL = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("LeftHand"));
	this->mCGHR = collisionEngine->createCollisionGroup(this->mCharacter->GetSkeleton()->getBodyNode("RightHand"));
	this->mCGG = collisionEngine->createCollisionGroup(this->mGround.get());
	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	// mActions = Eigen::VectorXd::Zero(mInterestedDof + 1);
	mActions = Eigen::VectorXd::Zero(dof-6);
	mActions.setZero();

	mMotionType.clear();
	mMotionType.push_back("idle");
	mMotionType.push_back("walk");
	mMotionType.push_back("jump");
	mControlObjective.resize(3);
	

	mEndEffectors.clear();
	mEndEffectors.push_back("RightFoot");
	mEndEffectors.push_back("LeftFoot");
	mEndEffectors.push_back("LeftHand");
	mEndEffectors.push_back("RightHand");
	//mEndEffectors.push_back("Head");

	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	this->mPDTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mPDTargetVelocities = Eigen::VectorXd::Zero(dof);

	//temp
	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();

	this->mNumMotionType = this->mReferenceManager->GetNumMotionType();
	this->mNumMotionParam = 4;

	this->mNumFeature = 2 * (RecordPose().rows() + RecordVel().rows())+ mNumMotionParam;
	this->mNumPose = this->mReferenceManager->GetNumPose();



	ClearRecord();
	
	mRewardLabels.clear();
	
	mRewardLabels.push_back("total");
	// mRewardLabels.push_back("p");
	// mRewardLabels.push_back("v");
	this->mRewardParts.resize(mRewardLabels.size(), 0.0);

	if(mRecord) mReferenceManager->setRecord();

}

void 
Controller::
initPhysicsEnv()
{
	this->mWorld = std::make_shared<dart::simulation::World>();
	this->mWorld->setTimeStep(1.0/(double)mSimulationHz);
	this->mWorld->setGravity(Eigen::Vector3d(0,-9.81,0));	
	this->mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::DARTCollisionDetector::create());
	dynamic_cast<dart::constraint::BoxedLcpConstraintSolver*>(mWorld->getConstraintSolver())->setBoxedLcpSolver(std::make_shared<dart::constraint::PgsBoxedLcpSolver>());
	this->mGround = DPhy::SkeletonBuilder::BuildFromFile(std::string(PROJECT_DIR)+std::string("/character/ground.xml")).first;
	this->mGround->getBodyNode(0)->setFrictionCoeff(1.0);
	this->mWorld->addSkeleton(this->mGround);

}
void 
Controller::
Step()
{	
	if(IsTerminalState()) {
		std::cout<<mCurrentFrame<<" , Terminal state"<<std::endl;
		return;
	}

	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	for(int i = 0; i < num_body_nodes; i++){
		Eigen::Vector3d joint_angle = mActions.segment(3*i,3);
		double len = joint_angle.norm();
		if(len>2*M_PI){
			joint_angle *= 2*M_PI / len;
		}
		mActions.segment(3*i,3) = joint_angle;
	}

	// for(int i = 0; i < mActions.size(); i++){
	// 	mActions[i] = dart::math::clip(mActions[i], -M_PI, M_PI);
	// }

	

	this->mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame+=1;
	this->mCurrentFrameOnPhase+=1;
	nTotalSteps += 1;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	//Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	//Eigen::VectorXd p_now = p_v_target->GetPosition();
	//this->mTargetPositions = p_now ; //p_v_target->GetPosition();
	//this->mTargetVelocities = mCharacter->GetSkeleton()->getPositionDifferences(mTargetPositions, mPrevTargetPositions) / 0.033;
	//mPrevTargetPositions = mTargetPositions;
	//delete p_v_target;

	Eigen::VectorXd torque;

	Eigen::VectorXd PDTargetPosition = Eigen::VectorXd::Zero(dof);
	//PDTargetPosition.head(6) = p_now.head(6);
	PDTargetPosition.tail(dof-6) = mActions;
	
	for(int i = 0; i < this->mSimPerCon; i++){

		// mCharacter->GetSkeleton()->setSPDTarget(mPDTargetPositions, 600, 49);
		// mWorld->step(false);
		
		// Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
		// Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(PDTargetPosition, 600, 49, mWorld->getConstraintSolver());
		Eigen::VectorXd torque = this->actuate(PDTargetPosition);

		mCharacter->GetSkeleton()->setForces(torque);
		mWorld->step(false);	
		mTimeElapsed += 1;
	}


	Eigen::VectorXd mCurrAgentPose = RecordPose();
	Eigen::VectorXd mCurrAgentVel = RecordVel();
	Eigen::VectorXd mAgentParam = GetAgentParam();

	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();


	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	Eigen::VectorXd p_ref = p_v_target->GetPosition();
	Eigen::VectorXd v_ref = p_v_target->GetVelocity();
	delete p_v_target;

	skel->setPositions(p_ref);
	skel->setVelocities(v_ref);


	Eigen::VectorXd mCurrExpertPose = RecordPose();
	Eigen::VectorXd mCurrExpertVel = RecordVel();
	Eigen::VectorXd mExpertParam = GetExpertParam();

	Eigen::VectorXd mMotionClass(this->mNumMotionType);
	mMotionClass.setZero();
	std::string type = this->mReferenceManager->GetMotionType(this->motion_it);
	for(int i=0; i<mNumMotionType;i++){
		if(!type.compare( mMotionType[i])){
			mMotionClass[i]=1.0;
		}
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);

	this->mAgentFeatureSet<<mCurrAgentPose,mPrevAgentPose,mCurrAgentVel,mPrevAgentVel,mAgentParam;
	this->mExpertFeatureSet<<mCurrExpertPose,mPrevExpertPose,mCurrExpertVel,mPrevExpertVel,mExpertParam;

	if(dart::math::isNan(mExpertFeatureSet)){
		std::cout<<"NAN occurs in EXPERT feature";
		std::cout<<mExpertFeatureSet.transpose()<<std::endl;
	}
	if(dart::math::isNan(mAgentFeatureSet)){
		std::cout<<"NAN occurs in AGENT feature";
		std::cout<<mAgentFeatureSet.transpose()<<std::endl;
	}

	this->mPPrevPosition = this->mPrevPosition;
	this->mPrevPosition = p_save;


	this->mPPrevAgentPose = mPrevAgentPose;
	this->mPPrevAgentVel = mPrevAgentVel;
	this->mPPrevExpertPose = mPrevExpertPose;
	this->mPPrevExpertVel = mPrevExpertVel;

	this->mPrevAgentPose = mCurrAgentPose;
	this->mPrevAgentVel = mCurrAgentVel;
	this->mPrevExpertPose = mCurrExpertPose;
	this->mPrevExpertVel = mCurrExpertVel;

	if(this->mCurrentFrameOnPhase > mReferenceManager->GetPhaseLength()){
		this->mCurrentFrameOnPhase -= mReferenceManager->GetPhaseLength();
		mRootZero = mCharacter->GetSkeleton()->getPositions().segment<6>(0);		
		mDefaultRootZero = mReferenceManager->GetMotion(mCurrentFrame)->GetPosition().segment<6>(0);
		mRootZeroDiff = mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase)->GetPosition().segment<3>(3);
		this->mStartRoot = this->mCharacter->GetSkeleton()->getPositions().segment<3>(3);
	}	

	this->UpdateReward();

	this->UpdateTerminalInfo();

	if(mRecord) {
		SaveStepInfo();
	}



	if((int)this->mCurrentFrame % 120 ==0){
		Eigen::Vector3d com = mCharacter->GetSkeleton()->getCOM();
		SetRandomTarget(com);
	}



}

Eigen::VectorXd
Controller::
actuate(const Eigen::VectorXd& target_pose)
{
	auto skel = mCharacter->GetSkeleton();
	Eigen::VectorXd kp = Eigen::VectorXd::Constant(skel->getNumDofs(),300.0);
	Eigen::VectorXd kv = Eigen::VectorXd::Constant(skel->getNumDofs(),30.0);
	kp.head<6>().setZero();
	kv.head<6>().setZero();

	Eigen::VectorXd q = skel->getPositions();
	Eigen::VectorXd dq = skel->getVelocities();
	double dt = skel->getTimeStep();
	// Eigen::MatrixXd M_inv = skel->getInvMassMatrix();
	Eigen::MatrixXd M_inv = (skel->getMassMatrix() + Eigen::MatrixXd(dt*kv.asDiagonal())).inverse();

	Eigen::VectorXd qdqdt = q + dq*dt;

	Eigen::VectorXd p_diff = -kp.cwiseProduct(skel->getPositionDifferences(qdqdt,target_pose));
	Eigen::VectorXd v_diff = -kv.cwiseProduct(dq);
	Eigen::VectorXd ddq = M_inv*(-skel->getCoriolisAndGravityForces()+p_diff+v_diff+skel->getConstraintForces());

	Eigen::VectorXd tau = p_diff + v_diff - dt*kv.cwiseProduct(ddq);

	tau.head<6>().setZero();

	return tau;
}
void
Controller::
SaveStepInfo() 
{
	mRecordBVHPosition.push_back(mReferenceManager->GetPosition(mCurrentFrame));
	mRecordTargetPosition.push_back(this->target_pos);
	mRecordPosition.push_back(mCharacter->GetSkeleton()->getPositions());
	mRecordVelocity.push_back(mCharacter->GetSkeleton()->getVelocities());
	mRecordCOM.push_back(mCharacter->GetSkeleton()->getCOM());
	mRecordPhase.push_back(mCurrentFrame);

	bool rightContact = CheckCollisionWithGround("RightFoot") || CheckCollisionWithGround("RightToe");
	bool leftContact = CheckCollisionWithGround("LeftFoot") || CheckCollisionWithGround("LeftToe");

	mRecordFootContact.push_back(std::make_pair(rightContact, leftContact));

}
void
Controller::
UpdateTerminalInfo()
{	
	Eigen::VectorXd p_ideal = mTargetPositions;
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];

	Eigen::Vector3d lf = mCharacter->GetSkeleton()->getBodyNode("LeftUpLeg")->getWorldTransform().translation();
	Eigen::Vector3d rf = mCharacter->GetSkeleton()->getBodyNode("RightUpLeg")->getWorldTransform().translation();
	Eigen::Vector3d ls = mCharacter->GetSkeleton()->getBodyNode("LeftShoulder")->getWorldTransform().translation();
	Eigen::Vector3d rs = mCharacter->GetSkeleton()->getBodyNode("RightShoulder")->getWorldTransform().translation();
	Eigen::Vector3d right_vector = ((rf-lf)+(rs-ls))/2.;
	right_vector[1]= 0;
	Eigen::Vector3d forward_vector=  Eigen::Vector3d::UnitY().cross(right_vector);
	double forward_angle= std::atan2(forward_vector[0], forward_vector[2]);

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	//skel->setPositions(mTargetPositions);
	//skel->computeForwardKinematics(true,false,false);

	Eigen::Isometry3d root_diff = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	
	Eigen::AngleAxisd root_diff_aa(root_diff.linear());
	double angle = RadianClamp(root_diff_aa.angle());
	Eigen::Vector3d root_pos_diff = root_diff.translation();


	// check nan
	if(dart::math::isNan(p)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 3;
	}
	if(dart::math::isNan(v)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 4;
	}
	if(dart::math::isNan(this->mAgentFeatureSet)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 5;
	}
	if(dart::math::isNan(this->mExpertFeatureSet)){
		mIsNanAtTerminal = true;
		mIsTerminal = true;
		terminationReason = 6;
	}

	// if(root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
	// 	mIsTerminal = true;
	// 	terminationReason = 2;
	// }

	double cur_height_limit = TERMINAL_ROOT_HEIGHT_UPPER_LIMIT;
	if(root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > TERMINAL_ROOT_HEIGHT_UPPER_LIMIT){
		mIsTerminal = true;
		terminationReason = 1;
	}
	// else if(!mRecord && std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
	// 	mIsTerminal = true;
	// 	terminationReason = 5;
	// }
	else if(!mRecord && mCurrentFrame > mReferenceManager->GetPhaseLength()*10) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}

	if(mRecord) {
		if(mIsTerminal) std::cout << "terminate Reason : "<<terminationReason << std::endl;
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);

}
void 
Controller::
ClearRecord() 
{
	this->mRecordVelocity.clear();
	this->mRecordPosition.clear();
	this->mRecordCOM.clear();
	this->mRecordTargetPosition.clear();
	this->mRecordBVHPosition.clear();
	this->mRecordPhase.clear();
	this->mRecordFootContact.clear();
}

std::vector<double>
Controller::
GetHeadingReward()
{

	dart::dynamics::SkeletonPtr skel = this->mCharacter->GetSkeleton();
	int dof = skel->getNumDofs();
	int num_body_nodes = skel->getNumBodyNodes();


	// double v_target = GetTargetSpeed();
	// Eigen::Vector3d p_target = GetTargetPosition();

	// // double sig_p = 1.0;
	// // double sig_v = v_target * v_target / 4;
	double sig_p = 1.5;
	double sig_v = 1.0;


	// Eigen::Isometry3d T_ref = this->getReferenceTransform();
	// Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	// Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	// Eigen::Vector3d p_COM = T_ref_inv*skel->getCOM();
	// Eigen::Vector3d v_COM = R_ref_inv*skel->getCOMLinearVelocity();

	// Eigen::Vector3d p_diff = p_target - p_COM;
	// p_diff[1] = 0.0;

	// double v_diff_reward = 0;
	
	// if (p_diff.norm() < this->dist_threshold)
	// {
	// 	v_diff_reward = 1.0;
	// }
	// else{

	// 	double v = v_COM.dot(p_diff.normalized());
	// 	double v_diff = v_target - v;

	// 	if(v<0) v_diff_reward=0;
	// 	else{
	// 		v_diff_reward = std::exp(-pow(v_diff,2)/(sig_v*sig_v));
	// 	}
	// }


	double tar_speed = this->mTargetSpeed;
	Eigen::Vector3d tar_dir = (this->target_pos - skel->getCOM()).normalized();

	Eigen::Vector3d com = skel->getCOM();
	double step_dur = skel->getTimeStep();
	Eigen::Vector3d avg_vel = (com - this->COM_prev) / step_dur;

	double avg_speed = tar_dir.dot(avg_vel);

	double vel_reward = 0;
	if (avg_speed > 0.0)
	{
		double vel_err = tar_speed - avg_speed;
		vel_reward = std::exp(-sig_v * vel_err * vel_err);
	}


	this->COM_prev = com;
	// double rew_p = exp_of_squared(p_diff,sig_p);
	double rew_v = vel_reward;


	// double rew_p_w = 0.6;
	// double rew_v_w = 0.4;

	// double reward = rew_p_w * rew_p + rew_v_w * rew_v;
	double reward = rew_v;

	std::vector<double> rewards;
	rewards.push_back(reward);
	// rewards.push_back(rew_p);
	// rewards.push_back(rew_v);
		//skel->computeForwardKinematics(true,true,false);
	return rewards;
}

std::vector<double> 
Controller::
GetTrackingReward(Eigen::VectorXd& position, Eigen::VectorXd& position2, 
	Eigen::VectorXd& velocity, Eigen::VectorXd& velocity2, bool useVelocity)
{

	dart::dynamics::SkeletonPtr skel = this->mCharacter->GetSkeleton();
	int dof = skel->getNumDofs();
	int num_body_nodes = skel->getNumBodyNodes();

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	std::vector<double> rewards;
	rewards.clear();


	// (q_t - q_0)
	Eigen::VectorXd p_diff_reward = skel->getPositionDifferences(position, position2);
	
	// (dq_t - dq_0)
	Eigen::VectorXd v_diff_reward;
	if(useVelocity) {
		v_diff_reward = skel->getVelocityDifferences(velocity, velocity2);
	}
	skel->setPositions(position);
	Eigen::Vector3d COM_1 = skel->getCOM();


	Eigen::VectorXd ee_diff_reward(mEndEffectors.size()*3);
	ee_diff_reward.setZero();
	std::vector<Eigen::Isometry3d> ee_transforms;	
	for(int i=0;i<mEndEffectors.size(); i++){
		ee_transforms.push_back(skel->getBodyNode(mEndEffectors[i])->getWorldTransform());
	}
	
	skel->setPositions(position2);
	Eigen::Vector3d COM_2 = skel->getCOM();

	for(int i=0;i<mEndEffectors.size();i++){
		Eigen::Isometry3d diff = ee_transforms[i].inverse() * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee_diff_reward.segment<3>(3*i) = diff.translation();
	}

	Eigen::Vector3d com_diff_reward = COM_1 - COM_2;


	double sig_p = 0.3; 
	double sig_v = 3.0;	
	double sig_com = 0.2;		
	double sig_ee = 0.2;		

	double r_p = exp_of_squared(p_diff_reward,sig_p);
	double r_v;
	if(useVelocity)
		r_v = exp_of_squared(v_diff_reward,sig_v);
	double r_ee = exp_of_squared(ee_diff_reward,sig_ee);
	double r_com = exp_of_squared(com_diff_reward,sig_com);
	double r_time = exp(-pow((mActions[mInterestedDof] - 1),2)*40);


	rewards.push_back(r_p);
	rewards.push_back(r_com);
	rewards.push_back(r_ee);
	if(useVelocity)
		rewards.push_back(r_v);
	rewards.push_back(r_time);
	
	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	//skel->computeForwardKinematics(true,true,false);
	return rewards;

}

void
Controller::
UpdateReward()
{
	dart::dynamics::SkeletonPtr skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	// std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(p, mTargetPositions,
	// 							 v, mTargetVelocities,  true);
	std::vector<double> task_rewards = this->GetHeadingReward();
	// double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();


	mRewardParts.clear();
	// double r_tot = 0.95 * (tracking_rewards_bvh[0] * tracking_rewards_bvh[1] *tracking_rewards_bvh[2] * tracking_rewards_bvh[3])  + 0.05 * tracking_rewards_bvh[4];
	if(dart::math::isNan(task_rewards[0])){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(task_rewards[0]);
		// mRewardParts.push_back(task_rewards[1]);
		// mRewardParts.push_back(task_rewards[2]);
	}
}

Eigen::Vector3d
Controller::
projectOnVector(const Eigen::Vector3d& u, const Eigen::Vector3d& v)
{
	Eigen::Vector3d projection;
	projection = v.dot(u)/v.dot(v)*v;
	return projection;
}

Eigen::Isometry3d
Controller::
getReferenceTransform()
{
	auto skel = mCharacter->GetSkeleton();
	Eigen::Isometry3d T = skel->getBodyNode(0)->getTransform();
	Eigen::Matrix3d R = T.linear();
	Eigen::Vector3d p = T.translation();
	Eigen::Vector3d z = R.col(2);
	Eigen::Vector3d y = Eigen::Vector3d::UnitY();
	z -= this->projectOnVector(z, y);
	p -= this->projectOnVector(p, y);

	z.normalize();
	Eigen::Vector3d x = y.cross(z);

	R.col(0) = x;
	R.col(1) = y;
	R.col(2) = z;

	T.linear() = R;
	T.translation() = p;

	return T;
}

Eigen::VectorXd
Controller::
GetState()
{
	if(mIsTerminal && terminationReason != 8){
		return Eigen::VectorXd::Zero(mNumState);
	}

	if(1)
	{
		dart::dynamics::SkeletonPtr skel = mCharacter->GetSkeleton();
		Eigen::Isometry3d T_ref = this->getReferenceTransform();
		Eigen::Isometry3d T_ref_inv = T_ref.inverse();
		Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

		int n = skel->getNumBodyNodes();
		std::vector<Eigen::Vector3d> ps(n),vs(n),ws(n);
		std::vector<Eigen::MatrixXd> Rs(n);

		for(int i=0;i<n;i++)
		{
			Eigen::Isometry3d Ti = T_ref_inv*(skel->getBodyNode(i)->getTransform());

			ps[i] = Ti.translation();
			Rs[i] = Ti.linear();

			vs[i] = R_ref_inv*skel->getBodyNode(i)->getLinearVelocity();
			ws[i] = R_ref_inv*skel->getBodyNode(i)->getAngularVelocity();
		}
		Eigen::Vector3d p_com = T_ref_inv*skel->getCOM();
		Eigen::Vector3d v_com = R_ref_inv*skel->getCOMLinearVelocity();

		std::vector<Eigen::Vector3d> states(5*n+2);

		int o = 0;
		for(int i=0;i<n;i++) states[o+i] = ps[i]; o += n;
		for(int i=0;i<n;i++) states[o+i] = Rs[i].col(0); o += n;
		for(int i=0;i<n;i++) states[o+i] = Rs[i].col(1); o += n;
		for(int i=0;i<n;i++) states[o+i] = vs[i]; o += n;
		for(int i=0;i<n;i++) states[o+i] = ws[i]; o += n;

		states[o+0] = p_com;
		states[o+1] = v_com;


		Eigen::Vector4d out_goal;
		Eigen::Vector3d pos_diff = this->target_pos - skel->getCOM();
		//pos_diff[1] = 0;

		double tar_dist = pos_diff.norm();
		if (tar_dist > 0.0001)
		{
			pos_diff = T_ref * pos_diff;
			pos_diff /= tar_dist;
		}
		else
		{
			pos_diff = {0,0,0};
		}



		if(pos_diff.norm()<0.00001){
			pos_diff= {0,0,1.0};
		}
		else{
			pos_diff = pos_diff.normalized();	
		}

		out_goal[0] = pos_diff[0];
		out_goal[1] = pos_diff[1];
		out_goal[2] = pos_diff[2];
		out_goal[3] = this->mTargetSpeed;


		Eigen::VectorXd s(states.size()*3 + 4);
		for(int i=0;i<states.size();i++)
		{
			s.segment<3>(i*3) = states[i];
		}


		s.tail(4) = out_goal;

		return s;
	}
	

	// if(0)
	// {
	// 	dart::dynamics::SkeletonPtr skel = mCharacter->GetSkeleton();
	
	// 	double root_height = skel->getRootBodyNode()->getCOM()[1];

	// 	Eigen::VectorXd p_save = skel->getPositions();
	// 	Eigen::VectorXd v_save = skel->getVelocities();
	// 	Eigen::VectorXd p,v;
	// 	// p.resize(p_save.rows()-6);
	// 	// p = p_save.tail(p_save.rows()-6);


	// 	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	// 	int num_p = (n_bnodes - 1) * 6;
	// 	p.resize(num_p);

	// 	for(int i = 1; i < n_bnodes; i++){
	// 		Eigen::Isometry3d transform = skel->getBodyNode(i)->getRelativeTransform();
	// 		// Eigen::Quaterniond q(transform.linear());
	// 		//	ret.segment<6>(6*i) << rot, transform.translation();
	// 		p.segment<6>(6*(i-1)) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
	// 								 transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2);
	// 	}

	// 	v = v_save;

	// 	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	// 	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	// 	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	// 	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);

	// 	// The angles and velocity of end effector & root info
	// 	Eigen::VectorXd ee;
	// 	ee.resize(mEndEffectors.size()*3);
	// 	for(int i=0;i<mEndEffectors.size();i++)
	// 	{
	// 		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
	// 		ee.segment<3>(3*i) << transform.translation();
	// 	}
	// 	double t = mReferenceManager->GetTimeStep(mCurrentFrameOnPhase);

	// 	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame+t);
	// 	Eigen::VectorXd p_now = p_v_target->GetPosition();
	// 	// The rotation and translation of end effector in the future(future 1 frame)
		
	// 	delete p_v_target;

	// 	double phase = ((int) mCurrentFrame % mReferenceManager->GetPhaseLength()) / (double) mReferenceManager->GetPhaseLength();
	// 	Eigen::VectorXd state;


	// 	double com_diff = 0;

	// 	//Eigen::Vector3d target_pos = Eigen::Vector3d::Zero();
	// 	//mReferenceManager->GetTargetPosition();
		
	// 	state.resize(p.rows()+v.rows()+ee.rows()+1+1);
	// 	state<< p, v, ee,up_vec_angle, root_height;
		

	// 	return state;
	// }
}

Eigen::VectorXd
Controller::
RecordPose(){
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();

	int n = skel->getNumBodyNodes();
	std::vector<Eigen::Vector3d> ps(n);
	std::vector<Eigen::MatrixXd> Rs(n);

	for(int i=0;i<n;i++)
	{
		Eigen::Isometry3d Ti = T_ref_inv*(skel->getBodyNode(i)->getTransform());
		ps[i] = Ti.translation();
		Rs[i] = Ti.linear();
	}

	Eigen::VectorXd ee;
	ee.resize(mEndEffectors.size()*3);
	for(int i=0;i<mEndEffectors.size();i++)
	{
		Eigen::Isometry3d transform = T_ref_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee.segment<3>(3*i) << transform.translation();
	}

	Eigen::VectorXd mPose(3*n *3 + ee.rows());

	int o = 0;
	for(int i=0;i<n;i++) mPose.segment<3>(3*(o+i)) = ps[i]; o += n;
	for(int i=0;i<n;i++) mPose.segment<3>(3*(o+i)) = Rs[i].col(0); o += n;
	for(int i=0;i<n;i++) mPose.segment<3>(3*(o+i)) = Rs[i].col(1); o += n;
	for(int i=0;i<mEndEffectors.size();i++) mPose.segment<3>(3*(o+i)) = ee.segment<3>(3*i);
	
	return mPose;
}

Eigen::VectorXd
Controller::
RecordVel(){
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	int n = skel->getNumBodyNodes();
	std::vector<Eigen::Vector3d> vs(n),ws(n);

	for(int i=0;i<n;i++)
	{
		vs[i] = R_ref_inv*skel->getBodyNode(i)->getLinearVelocity();
		ws[i] = R_ref_inv*skel->getBodyNode(i)->getAngularVelocity();
	}

	Eigen::VectorXd mVel(2*n *3);

	int o = 0;
	for(int i=0;i<n;i++) mVel.segment<3>(3*(o+i)) = vs[i]; o += n;
	for(int i=0;i<n;i++) mVel.segment<3>(3*(o+i)) = ws[i]; o += n;
	
	return mVel;
}

Eigen::VectorXd
Controller::
GetAgentParam(){
	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();

	skel->setPositions(this->mPrevPosition);

	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	Eigen::Vector3d v_com1 = R_ref_inv*skel->getCOMLinearVelocity();

	
	skel->setPositions(this->mPPrevPosition);
	Eigen::Vector3d v_com0 = R_ref_inv*skel->getCOMLinearVelocity();

	skel->setPositions(p_save);
	Eigen::Vector3d v_com2 = R_ref_inv*skel->getCOMLinearVelocity();

	Eigen::Vector3d dir = (v_com0 + 2* v_com1 + v_com2)/4;
	if(dir.norm()<0.00001){
		dir ={0,1.0,0.0};
	}
	Eigen::Vector4d out;
	out.head(3) = dir.normalized();
	out[4] = dir.norm();

	
	
	return out;
}

Eigen::VectorXd
Controller::
GetExpertParam(){
	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::VectorXd p_save = skel->getPositions();

	int n1 = mCurrentFrame-1 > 0 ? mCurrentFrame-1 : mCurrentFrame;
	Motion* p_v_target = mReferenceManager->GetMotion(n1);
	Eigen::VectorXd p_ref = p_v_target->GetPosition();
	skel->setPositions(p_ref);

	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	Eigen::Vector3d v_com1 = R_ref_inv*skel->getCOMLinearVelocity();

	int n0 = n1-1 > 0 ? n1-1 : n1;
	p_v_target = mReferenceManager->GetMotion(n0);
	p_ref = p_v_target->GetPosition();
	skel->setPositions(p_ref);

	Eigen::Vector3d v_com0 = R_ref_inv*skel->getCOMLinearVelocity();
	
	delete p_v_target;

	skel->setPositions(p_save);
	Eigen::Vector3d v_com2 = R_ref_inv*skel->getCOMLinearVelocity();

	Eigen::Vector3d dir = (v_com0 + 2* v_com1 + v_com2)/4;
	if(dir.norm()<0.00001){
		dir ={0,1.0,0.0};
	}
	Eigen::Vector4d out;
	out.head(3) = dir.normalized();
	out[4] = dir.norm();
	
	return out;
}


bool
Controller::
FollowBvh()
{	
	if(this->mIsTerminal)
		return false;
	auto& skel = mCharacter->GetSkeleton();

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	mTargetPositions = p_v_target->GetPosition();
	mTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	for(int i=0;i<this->mSimPerCon;i++)
	{
		skel->setPositions(mTargetPositions);
		skel->setVelocities(mTargetVelocities);
		skel->computeForwardKinematics(true, true, false);
	}
	this->mCurrentFrame += 1;
	this->nTotalSteps += 1;
	return true;
}

void
Controller::
SetSkeletonWeight(double mass)
{

	double m_new = mass / mMass;

	std::vector<std::tuple<std::string, Eigen::Vector3d, double>> deform;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	for(int i = 0; i < n_bnodes; i++){
		std::string name = mCharacter->GetSkeleton()->getBodyNode(i)->getName();
		deform.push_back(std::make_tuple(name, Eigen::Vector3d(1, 1, 1), m_new));
	}
	DPhy::SkeletonBuilder::DeformSkeleton(mCharacter->GetSkeleton(), deform);
	mMass = mCharacter->GetSkeleton()->getMass();
}

void 
Controller::
Reset(bool RSI)
{
	this->mWorld->reset();

	this->mTargetPositions.setZero();
	this->mTargetVelocities.setZero();


	
	this-> motion_it = std::rand()%this->mNumMotions;
	mReferenceManager->SelectMotion(motion_it);
	auto& skel = this->mCharacter->GetSkeleton();

	
	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();
	//RSI
	if(RSI) {
		this->mCurrentFrame = (int) dart::math::Random::uniform(0.0, mReferenceManager->GetPhaseLength()-5.0);
	}

	Motion* p_v_target;
	p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	// Eigen::VectorXd Initialpose = p_v_target->GetPosition();
	// this->mTargetVelocities = p_v_target->GetVelocity();
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	// this->mTargetPositions[3]=0;
	// this->mTargetPositions[5]=0;
	
	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);

	this->mPrevPosition = mTargetPositions;
	this->mPPrevPosition = mTargetPositions;
	this->COM_prev = skel->getCOM();

	this->mCurrentFrameOnPhase = this->mCurrentFrame;
	this->mStartFrame = this->mCurrentFrame;
	this->nTotalSteps = 0;
	this->mTimeElapsed = 0;

	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	int n = skel->getNumBodyNodes();
	std::vector<Eigen::Vector3d> ps(n),vs(n),ws(n);
	std::vector<Eigen::MatrixXd> Rs(n);

	for(int i=0;i<n;i++)
	{
		Eigen::Isometry3d Ti = T_ref_inv*(skel->getBodyNode(i)->getTransform());

		ps[i] = Ti.translation();
		Rs[i] = Ti.linear();

		vs[i] = R_ref_inv*skel->getBodyNode(i)->getLinearVelocity();
		ws[i] = R_ref_inv*skel->getBodyNode(i)->getAngularVelocity();
	}
	Eigen::VectorXd ee;
	ee.resize(mEndEffectors.size()*3);
	for(int i=0;i<mEndEffectors.size();i++)
	{
		Eigen::Isometry3d transform = T_ref_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee.segment<3>(3*i) << transform.translation();
	}

	this->mPPrevAgentPose.resize(3*3*n + ee.rows());
	this->mPPrevAgentVel.resize(2*3*n);
	this->mPPrevAgentPose.setZero();
	this->mPPrevAgentVel.setZero();

	this->mPrevAgentPose.resize(3*3*n + ee.rows());
	this->mPrevAgentVel.resize(2*3*n);
	this->mPrevAgentPose.setZero();
	this->mPrevAgentVel.setZero();

	int o = 0;
	for(int i=0;i<n;i++) this->mPrevAgentPose.segment<3>(3*(o+i)) = ps[i]; o += n;
	for(int i=0;i<n;i++) this->mPrevAgentPose.segment<3>(3*(o+i)) = Rs[i].col(0); o += n;
	for(int i=0;i<n;i++) this->mPrevAgentPose.segment<3>(3*(o+i)) = Rs[i].col(1); o += n;
	for(int i=0;i<mEndEffectors.size();i++) this->mPrevAgentPose.segment<3>(3*(o+i)) = ee.segment<3>(3*i);

	o = 0;
	for(int i=0;i<n;i++) this->mPrevAgentVel.segment<3>(3*(o+i)) = vs[i]; o += n;
	for(int i=0;i<n;i++) this->mPrevAgentVel.segment<3>(3*(o+i)) = ws[i]; o += n;

	this->mPPrevAgentPose = this->mPrevAgentPose;
	this->mPPrevAgentVel = this->mPrevAgentVel;

	this->mPPrevExpertPose = this->mPPrevAgentPose;
	this->mPPrevExpertVel = this->mPPrevAgentVel;
	this->mPrevExpertPose = this->mPrevAgentPose;
	this->mPrevExpertVel = this->mPrevAgentVel;


	this-> mNumFeature = (mPrevAgentPose.rows()+mPrevAgentVel.rows())*2 + mNumMotionParam;
	this-> mAgentFeatureSet.resize(mNumFeature);
	this-> mExpertFeatureSet.resize(mNumFeature);
	this-> mAgentFeatureSet.setZero();
	this-> mExpertFeatureSet.setZero();

	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;
	ClearRecord();
	SaveStepInfo();

	Eigen::VectorXd tl_ee(3 + mEndEffectors.size() * 3);
	tl_ee.segment<3>(0) = skel->getRootBodyNode()->getWorldTransform().translation();
	for(int i = 0; i < mEndEffectors.size(); i++) {
		tl_ee.segment<3>(i*3 + 3) = skel->getBodyNode(mEndEffectors[i])->getWorldTransform().translation();
	}
	mRootZero = mTargetPositions.segment<6>(0);
	this->mRootZeroDiff= mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase)->GetPosition().segment<3>(3);

	mDefaultRootZero = mRootZero; 

	mTlPrev = tl_ee;	

	mPrevFrame = mCurrentFrame;

	mPrevTargetPositions = mTargetPositions;

	this->mStartRoot = this->mCharacter->GetSkeleton()->getPositions().segment<3>(3);
	this->mRootZeroDiff= mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase)->GetPosition().segment<3>(3);


	SetRandomTarget(this->mStartRoot);


}

bool
Controller::
CheckCollisionWithGround(std::string bodyName){
	auto collisionEngine = mWorld->getConstraintSolver()->getCollisionDetector();
	dart::collision::CollisionOption option;
	dart::collision::CollisionResult result;
	if(bodyName == "RightFoot"){
		bool isCollide = collisionEngine->collide(this->mCGR.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftFoot"){
		bool isCollide = collisionEngine->collide(this->mCGL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "RightToe"){
		bool isCollide = collisionEngine->collide(this->mCGER.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftToe"){
		bool isCollide = collisionEngine->collide(this->mCGEL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "RightHand"){
		bool isCollide = collisionEngine->collide(this->mCGHR.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else if(bodyName == "LeftHand"){
		bool isCollide = collisionEngine->collide(this->mCGHL.get(), this->mCGG.get(), option, &result);
		return isCollide;
	}
	else{ // error case
		std::cout << "check collision : bad body name" << std::endl;
		return false;
	}
}

void
Controller::
SetRandomTarget(const Eigen::Vector3d& root_pos){
	double time_min = 1;
	double time_max = 5;
	this->mTargetSpeed = 0.5 + (2.5-0.5) * double(std::rand())/RAND_MAX;

	this->mMinTargetDist = 1.0;
	this->mMaxTargetDist = 3.0;
	this->dist_threshold = 0.1;
 	
	std::srand(std::time(NULL));

	double dist = mMinTargetDist + (mMaxTargetDist-mMinTargetDist) * double(std::rand())/RAND_MAX;
	double theta = -M_PI + 2 * M_PI * double(std::rand())/RAND_MAX;

	bool sharp_turn = double(std::rand())/RAND_MAX < 0.025;
	if(!sharp_turn){
		theta = theta< M_PI/6 ? theta : M_PI/6;
		theta = theta> -M_PI/6 ? theta : -M_PI/6;
	}
	this->target_pos[0] = root_pos[0] + dist * std::sin(theta);
	this->target_pos[1] = 0.6 + (1.2-0.6) * double(std::rand())/RAND_MAX;
	this->target_pos[2] = root_pos[2] + dist * std::cos(theta);



}


void
Controller::SaveDisplayedData(std::string directory, bool bvh) {

}

}