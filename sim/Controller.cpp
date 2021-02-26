
#include "Controller.h"
using namespace dart::dynamics;
namespace DPhy
{

Controller::Controller(ReferenceManager* ref, std::string character_path, bool record, int id)
	:mControlHz(30),mSimulationHz(150),mCurrentFrame(0),
	w_p(0.35),w_v(0.1),w_ee(0.3),w_com(0.25),
	terminationReason(-1),mIsNanAtTerminal(false), mIsTerminal(false)
{

	initPhysicsEnv();
	this->mRescaleParameter = std::make_tuple(1.0, 1.0, 1.0);
	this->mRecord = record;
	this->mReferenceManager = ref;
	this->id = id;

	this->mCharacter = new DPhy::Character(character_path);
	this->mWorld->addSkeleton(this->mCharacter->GetSkeleton());
	this->mPath = character_path;

	this->mMass = mCharacter->GetSkeleton()->getMass();

	this->mSimPerCon = mSimulationHz / mControlHz;
	
	this->mCurrentFrameOnPhase = 0;


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

	mInterestedDof = mCharacter->GetSkeleton()->getNumDofs() - 6;
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
	
	mActions = Eigen::VectorXd::Zero(mInterestedDof + 1);
	mActions.setZero();

	mEndEffectors.clear();
	mEndEffectors.push_back("RightFoot");
	mEndEffectors.push_back("LeftFoot");
	mEndEffectors.push_back("LeftHand");
	mEndEffectors.push_back("RightHand");
	mEndEffectors.push_back("Head");

	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	this->mPDTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mPDTargetVelocities = Eigen::VectorXd::Zero(dof);

	//temp
	this->mRewardParts.resize(7, 0.0);
	this->mNumState = this->GetState().rows();

	this->mNumAction = mActions.size();
	ClearRecord();
	
	mRewardLabels.clear();
	
	mRewardLabels.push_back("total");
	mRewardLabels.push_back("p");
	mRewardLabels.push_back("com");
	mRewardLabels.push_back("ee");
	mRewardLabels.push_back("v");
	mRewardLabels.push_back("time");

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

	Eigen::VectorXd s = this->GetState();

	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	for(int i = 0; i < mInterestedDof; i++){
		mActions[i] = dart::math::clip(mActions[i]*0.2, -0.7*M_PI, 0.7*M_PI);
	}
	int sign = 1;
	if(mActions[mInterestedDof] < 0)
		sign = -1;
	
	mActions[mInterestedDof] = dart::math::clip(mActions[mInterestedDof]*1.2, -2.0, 1.0);
	mActions[mInterestedDof] = exp(mActions[mInterestedDof]);

	this->mPrevFrameOnPhase = this->mCurrentFrameOnPhase;
	this->mCurrentFrame+=1;
	this->mCurrentFrameOnPhase+=1;
	nTotalSteps += 1;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	Eigen::VectorXd p_now = p_v_target->GetPosition();
	this->mTargetPositions = p_now ; //p_v_target->GetPosition();
	this->mTargetVelocities = mCharacter->GetSkeleton()->getPositionDifferences(mTargetPositions, mPrevTargetPositions) / 0.033;
	delete p_v_target;

	p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	this->mPDTargetPositions = p_v_target->GetPosition();
	this->mPDTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;


	int count_dof = 0;

	for(int i = 1; i <= num_body_nodes; i++){
		int idx = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getIndexInSkeleton(0);
		int dof = mCharacter->GetSkeleton()->getBodyNode(i)->getParentJoint()->getNumDofs();
		mPDTargetPositions.block(idx, 0, dof, 1) += mActions.block(count_dof, 0, dof, 1);
		count_dof += dof;
	}

	Eigen::VectorXd torque;
	
	for(int i = 0; i < this->mSimPerCon; i++){

		// mCharacter->GetSkeleton()->setSPDTarget(mPDTargetPositions, 600, 49);
		// mWorld->step(false);
		
		// torque limit
		Eigen::VectorXd torque = mCharacter->GetSkeleton()->getSPDForces(mPDTargetPositions, 600, 49, mWorld->getConstraintSolver());
		for(int j = 0; j < num_body_nodes; j++) {
			int idx = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getIndexInSkeleton(0);
			int dof = mCharacter->GetSkeleton()->getBodyNode(j)->getParentJoint()->getNumDofs();
			std::string name = mCharacter->GetSkeleton()->getBodyNode(j)->getName();
			double torquelim = mCharacter->GetTorqueLimit(name) * 1.5;
			double torque_norm = torque.block(idx, 0, dof, 1).norm();
			
			torque.block(idx, 0, dof, 1) = std::max(-torquelim, std::min(torquelim, torque_norm)) * torque.block(idx, 0, dof, 1).normalized();
		}

		mCharacter->GetSkeleton()->setForces(torque);
		mWorld->step(false);	
		mTimeElapsed += 1;
	}

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

	mPrevTargetPositions = mTargetPositions;

}

void
Controller::
SaveStepInfo() 
{
	mRecordBVHPosition.push_back(mReferenceManager->GetPosition(mCurrentFrame));
	mRecordTargetPosition.push_back(mTargetPositions);
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

	skel->setPositions(mTargetPositions);
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

	if(!mRecord && root_pos_diff.norm() > TERMINAL_ROOT_DIFF_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 2;
	}

	double cur_height_limit = TERMINAL_ROOT_HEIGHT_UPPER_LIMIT;
	if(!mRecord && root_y<TERMINAL_ROOT_HEIGHT_LOWER_LIMIT || root_y > cur_height_limit){
		mIsTerminal = true;
		terminationReason = 1;
	}
	else if(!mRecord && std::abs(angle) > TERMINAL_ROOT_DIFF_ANGLE_THRESHOLD){
		mIsTerminal = true;
		terminationReason = 5;
	}
	else if(mCurrentFrame > mReferenceManager->GetPhaseLength() * 10) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}

	if(mRecord) {
		if(mIsTerminal) std::cout << "terminate Reason : "<<terminationReason << std::endl;
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	//skel->computeForwardKinematics(true,true,false);

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
GetTrackingReward(Eigen::VectorXd position, Eigen::VectorXd position2, 
	Eigen::VectorXd velocity, Eigen::VectorXd velocity2, bool useVelocity)
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

	
	double scale = 1.0;

	double sig_p = 0.4 * scale; 
	double sig_v = 3 * scale;	
	double sig_com = 0.2 * scale;		
	double sig_ee = 0.5 * scale;		

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
	std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(skel->getPositions(), mTargetPositions,
								 skel->getVelocities(), mTargetVelocities,  true);
	double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();


	mRewardParts.clear();
	double r_tot = 0.95 * (tracking_rewards_bvh[0] * tracking_rewards_bvh[2] * tracking_rewards_bvh[3])  + 0.05 * tracking_rewards_bvh[4];
	if(dart::math::isNan(r_tot)){
		mRewardParts.resize(mRewardLabels.size(), 0.0);
	}
	else {
		mRewardParts.push_back(r_tot);
		mRewardParts.push_back(tracking_rewards_bvh[0]);
		mRewardParts.push_back(tracking_rewards_bvh[1]);
		mRewardParts.push_back(tracking_rewards_bvh[2]);
		mRewardParts.push_back(tracking_rewards_bvh[3]);
		mRewardParts.push_back(tracking_rewards_bvh[4]);
	}
}
Eigen::VectorXd 
Controller::
GetEndEffectorStatePosAndVel(const Eigen::VectorXd pos, const Eigen::VectorXd vel) {
	Eigen::VectorXd ret;
	auto& skel = mCharacter->GetSkeleton();
	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	int num_ee = mEndEffectors.size();
	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();

	skel->setPositions(pos);
	skel->setVelocities(vel);
	skel->computeForwardKinematics(true, true, false);

	ret.resize((num_ee)*12+15);
//	ret.resize((num_ee)*9+12);

	for(int i=0;i<num_ee;i++)
	{		
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		//Eigen::Quaterniond q(transform.linear());
		// Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
		ret.segment<9>(9*i) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
							   transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2), 
							   transform.translation();
//		ret.segment<6>(6*i) << rot, transform.translation();
	}


	for(int i=0;i<num_ee;i++)
	{
	    int idx = skel->getBodyNode(mEndEffectors[i])->getParentJoint()->getIndexInSkeleton(0);
		ret.segment<3>(9*num_ee + 3*i) << vel.segment<3>(idx);
//	    ret.segment<3>(6*num_ee + 3*i) << vel.segment<3>(idx);

	}

	// root diff with target com
	Eigen::Isometry3d transform = cur_root_inv * skel->getRootBodyNode()->getWorldTransform();
	//Eigen::Quaterniond q(transform.linear());

	Eigen::Vector3d rot = QuaternionToDARTPosition(Eigen::Quaterniond(transform.linear()));
	Eigen::Vector3d root_angular_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getAngularVelocity();
	Eigen::Vector3d root_linear_vel_relative = cur_root_inv.linear() * skel->getRootBodyNode()->getCOMLinearVelocity();

	ret.tail<15>() << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
					  transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2),
					  transform.translation(), root_angular_vel_relative, root_linear_vel_relative;
//	ret.tail<12>() << rot, transform.translation(), root_angular_vel_relative, root_linear_vel_relative;

	// restore
	skel->setPositions(p_save);
	skel->setVelocities(v_save);
	//skel->computeForwardKinematics(true, true, false);

	return ret;
}

Eigen::VectorXd
Controller::
GetState()
{
	// State Component : Joint_angle, Joint_velocity, up_vector_angle, p_next,
	if(mIsTerminal && terminationReason != 8){
		return Eigen::VectorXd::Zero(mNumState);
	}
	dart::dynamics::SkeletonPtr skel = mCharacter->GetSkeleton();
	
	double root_height = skel->getRootBodyNode()->getCOM()[1];

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();
	Eigen::VectorXd p,v;
	// p.resize(p_save.rows()-6);
	// p = p_save.tail(p_save.rows()-6);

	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();
	int num_p = (n_bnodes - 1) * 6;
	p.resize(num_p);

	for(int i = 1; i < n_bnodes; i++){
		Eigen::Isometry3d transform = skel->getBodyNode(i)->getRelativeTransform();
		// Eigen::Quaterniond q(transform.linear());
		//	ret.segment<6>(6*i) << rot, transform.translation();
		p.segment<6>(6*(i-1)) << transform.linear()(0,0), transform.linear()(0,1), transform.linear()(0,2),
								 transform.linear()(1,0), transform.linear()(1,1), transform.linear()(1,2);
	}

	v = v_save;

	dart::dynamics::BodyNode* root = skel->getRootBodyNode();
	Eigen::Isometry3d cur_root_inv = root->getWorldTransform().inverse();

	Eigen::Vector3d up_vec = root->getTransform().linear()*Eigen::Vector3d::UnitY();
	double up_vec_angle = atan2(std::sqrt(up_vec[0]*up_vec[0]+up_vec[2]*up_vec[2]),up_vec[1]);

	// The angles and velocity of end effector & root info
	Eigen::VectorXd ee;
	ee.resize(mEndEffectors.size()*3);
	for(int i=0;i<mEndEffectors.size();i++)
	{
		Eigen::Isometry3d transform = cur_root_inv * skel->getBodyNode(mEndEffectors[i])->getWorldTransform();
		ee.segment<3>(3*i) << transform.translation();
	}
	double t = mReferenceManager->GetTimeStep(mCurrentFrameOnPhase);

	Motion* p_v_target = mReferenceManager->GetMotion(mCurrentFrame+t);
	Eigen::VectorXd p_now = p_v_target->GetPosition();
	// The rotation and translation of end effector in the future(future 1 frame)
	Eigen::VectorXd p_next = GetEndEffectorStatePosAndVel(p_now, p_v_target->GetVelocity()*t);
	
	delete p_v_target;

	double phase = ((int) mCurrentFrame % mReferenceManager->GetPhaseLength()) / (double) mReferenceManager->GetPhaseLength();
	Eigen::VectorXd state;

	double com_diff = 0;
	
	state.resize(p.rows()+v.rows()+1+1+p_next.rows()+ee.rows()+2);
	state<< p, v, up_vec_angle, root_height, p_next, mAdaptiveStep, ee, mCurrentFrameOnPhase;

	return state;
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
	dart::dynamics::SkeletonPtr skel = mCharacter->GetSkeleton();
	// skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();
	//RSI
	if(RSI) {
		this->mCurrentFrame = (int) dart::math::Random::uniform(0.0, mReferenceManager->GetPhaseLength()-5.0);
	}
	// else {
	// 	this->mCurrentFrame = 0; // 0;
	// 	this->mParamRewardTrajectory = 0;
	// 	this->mTrackingRewardTrajectory = 0;
	// 	mFitness.sum_contact = 0;
	// 	mFitness.sum_slide = 0;
	// 	mFitness.sum_hand_ct = 0;
	// 	mFitness.hand_ct_cnt = 0;
	// 	mFitness.sum_pos.resize(skel->getNumDofs());
	// 	mFitness.sum_vel.resize(skel->getNumDofs());
	// 	mFitness.sum_pos.setZero();
	// 	mFitness.sum_vel.setZero();
	// }

	this->mCurrentFrameOnPhase = this->mCurrentFrame;
	this->mStartFrame = this->mCurrentFrame;
	this->nTotalSteps = 0;
	this->mTimeElapsed = 0;

	Motion* p_v_target;
	p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	this->mTargetPositions = p_v_target->GetPosition();
	this->mTargetVelocities = p_v_target->GetVelocity();
	delete p_v_target;

	// // Eigen::VectorXd nextTargetPositions = mReferenceManager->GetPosition(mCurrentFrame+1, isAdaptive);
	// // this->mTargetVelocities = mCharacter->GetSkeleton()->getPositionDifferences(nextTargetPositions, mTargetPositions) / 0.033;
	// // std::cout <<  mCharacter->GetSkeleton()->getPositionDifferences(nextTargetPositions, mTargetPositions).segment<3>(3).transpose() << std::endl;
	this->mPDTargetPositions = mTargetPositions;
	this->mPDTargetVelocities = mTargetVelocities;

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);
	// skel->computeForwardKinematics(true,true,false);

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
	//mPrevFrame2 = mPrevFrame;

	mPrevTargetPositions = mTargetPositions;

	this->mStartRoot = this->mCharacter->GetSkeleton()->getPositions().segment<3>(3);
	this->mRootZeroDiff= mRootZero.segment<3>(3) - mReferenceManager->GetMotion(mCurrentFrameOnPhase)->GetPosition().segment<3>(3);
	
	// dbg_LeftPoints= std::vector<Eigen::Vector3d>();
	// dbg_RightPoints= std::vector<Eigen::Vector3d>();
	// dbg_LeftConstraintPoint= Eigen::Vector3d::Zero();
	// dbg_RightConstraintPoint= Eigen::Vector3d::Zero();
	
	// // std::cout<<"RSI : "<<mCurrentFrame<<std::endl;
	// if(leftHandConstraint && mCurrentFrame <30) removeHandFromBar(true);
	// if(rightHandConstraint && mCurrentFrame <30) removeHandFromBar(false);

	// //45, 59
	// left_detached= (mCurrentFrame >=37) ? true: false; 
	// right_detached= (mCurrentFrame >=51) ? true: false;

	// min_hand = 10000;


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
Controller::SaveDisplayedData(std::string directory, bool bvh) {

}

}