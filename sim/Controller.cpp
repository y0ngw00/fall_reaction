
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
	this->motion_it=0;
 	this->target_dir = {0,0,1};


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

	int num_body_nodes = mInterestedDof / 3;
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	// mActions = Eigen::VectorXd::Zero(mInterestedDof + 1);
	mActions = Eigen::VectorXd::Zero(dof-6);
	mActions.setZero();

	mMotionType.clear();
	mMotionType.push_back("idle");
	mMotionType.push_back("walk");
	mMotionType.push_back("jump");


	mEndEffectors.clear();
	mEndEffectors.push_back("RightFoot");
	mEndEffectors.push_back("LeftFoot");
	mEndEffectors.push_back("LeftHand");
	mEndEffectors.push_back("RightHand");
	//mEndEffectors.push_back("Head");

	this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	this->mTargetVelocities = Eigen::VectorXd::Zero(dof);

	//temp
	this->mNumState = this->GetState().rows();
	this->mNumAction = mActions.size();

	this->mNumMotionType = this->mReferenceManager->GetNumMotionType();
	this->mNumMotionParam = 4;

	this->mNumFeature = 2 * (RecordPose().rows() + RecordVel().rows())+ mNumMotionParam;
	this->mNumPose = this->mReferenceManager->GetNumPose();

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

	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

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

	this->mCurrentFrame+=1;
	nTotalSteps += 1;
	int n_bnodes = mCharacter->GetSkeleton()->getNumBodyNodes();


	Eigen::VectorXd torque;
	Eigen::VectorXd PDTargetPosition = Eigen::VectorXd::Zero(dof);
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

	if(!mRecord){



		Eigen::VectorXd mCurrAgentPose = RecordPose();
		Eigen::VectorXd mCurrAgentVel = RecordVel();
		Eigen::VectorXd mAgentParam = GetAgentParam(R_ref_inv);

		auto& skel = this->mCharacter->GetSkeleton();
		Eigen::VectorXd p_save = skel->getPositions();
		Eigen::VectorXd v_save = skel->getVelocities();


		Motion p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
		Eigen::VectorXd p_ref = p_v_target.GetPosition();
		Eigen::VectorXd v_ref = p_v_target.GetVelocity();


		skel->setPositions(p_ref);
		skel->setVelocities(v_ref);


		Eigen::VectorXd mCurrExpertPose = RecordPose();
		Eigen::VectorXd mCurrExpertVel = RecordVel();
		Eigen::VectorXd mExpertParam = GetExpertParam();



		// Eigen::VectorXd mMotionClass(this->mNumMotionType);
		// mMotionClass.setZero();
		// std::string type = this->mReferenceManager->GetMotionType(this->motion_it);
		// for(int i=0; i<mNumMotionType;i++){
		// 	if(!type.compare( mMotionType[i])){
		// 		mMotionClass[i]=1.0;
		// 	}
		// }

		skel->setPositions(p_save);
		skel->setVelocities(v_save);

		// this->mPPrevCOMvel = this->mPrevCOMvel;



		this->mAgentFeatureSet.setZero();
		this->mExpertFeatureSet.setZero();

		int o=0;
		for(int i=0;i<mCurrAgentPose.rows();i++) mAgentFeatureSet[o+i] = mCurrAgentPose[i]; o += mCurrAgentPose.rows();
		for(int i=0;i<mPrevAgentPose.rows();i++) mAgentFeatureSet[o+i] = mPrevAgentPose[i]; o += mPrevAgentPose.rows();
		for(int i=0;i<mCurrAgentVel.rows();i++) mAgentFeatureSet[o+i] = mCurrAgentVel[i]; o += mCurrAgentVel.rows();
		for(int i=0;i<mPrevAgentVel.rows();i++) mAgentFeatureSet[o+i] = mPrevAgentVel[i]; o += mPrevAgentVel.rows();
		for(int i=0;i<mAgentParam.rows();i++) mAgentFeatureSet[o+i] = mAgentParam[i]; o += mAgentParam.rows();
		assert (n == mAgentFeatureSet.rows());
		
		o=0;
		for(int i=0;i<mCurrExpertPose.rows();i++) mExpertFeatureSet[o+i] = mCurrExpertPose[i]; o += mCurrExpertPose.rows();
		for(int i=0;i<mPrevExpertPose.rows();i++) mExpertFeatureSet[o+i] = mPrevExpertPose[i]; o += mPrevExpertPose.rows();
		for(int i=0;i<mCurrExpertVel.rows();i++) mExpertFeatureSet[o+i] = mCurrExpertVel[i]; o += mCurrExpertVel.rows();
		for(int i=0;i<mPrevExpertVel.rows();i++) mExpertFeatureSet[o+i] = mPrevExpertVel[i]; o += mPrevExpertVel.rows();
		for(int i=0;i<mExpertParam.rows();i++) mExpertFeatureSet[o+i] = mExpertParam[i]; o += mExpertParam.rows();
		assert (n == mExpertFeatureSet.rows());

		if(dart::math::isNan(mExpertFeatureSet)){
			std::cout<<"NAN occurs in EXPERT feature";
			std::cout<<mExpertFeatureSet.transpose()<<std::endl;
		}
		if(dart::math::isNan(mAgentFeatureSet)){
			std::cout<<"NAN occurs in AGENT feature";
			std::cout<<mAgentFeatureSet.transpose()<<std::endl;
		}


		this->mPrevCOMvel = skel->getCOMLinearVelocity();

		this->mPrevAgentPose = mCurrAgentPose;
		this->mPrevAgentVel = mCurrAgentVel;
		this->mPrevExpertPose = mCurrExpertPose;
		this->mPrevExpertVel = mCurrExpertVel;
	}
	

	if(this->mCurrentFrame > mReferenceManager->GetPhaseLength()){
		this->mCurrentFrame -= mReferenceManager->GetPhaseLength();
		this->mStartRoot = this->mCharacter->GetSkeleton()->getPositions().segment<3>(3);
	}	

	this->UpdateReward();
	this->UpdateTerminalInfo();
	
	this->target_dir = this->mCharacter->GetSkeleton()->getCOMLinearVelocity();
	this->target_dir[1]=0;
	this->target_dir = this->target_rot*target_dir;

	if(mRecord){
		Eigen::Vector3d com = mCharacter->GetSkeleton()->getCOM();
		if(this->theta!= this->theta_prev)
			SetTarget(com, !mRecord);
		theta_prev = theta;
	}

	else{
		if((int)this->nTotalSteps % 120 ==0){
			Eigen::Vector3d com = mCharacter->GetSkeleton()->getCOM();
			SetTarget(com, !mRecord);
		}
		
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
UpdateTerminalInfo()
{	
	auto& skel = this->mCharacter->GetSkeleton();

	Eigen::VectorXd p = skel->getPositions();
	Eigen::VectorXd v = skel->getVelocities();
	Eigen::Vector3d root_pos = skel->getPositions().segment<3>(3);
	Eigen::Isometry3d cur_root_inv = skel->getRootBodyNode()->getWorldTransform().inverse();
	double root_y = skel->getBodyNode(0)->getTransform().translation()[1];

	Eigen::VectorXd p_save = skel->getPositions();
	Eigen::VectorXd v_save = skel->getVelocities();


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
	else if(!mRecord && this->nTotalSteps > 600 ) { // this->mBVH->GetMaxFrame() - 1.0){
		mIsTerminal = true;
		terminationReason =  8;
	}

	if(mRecord) {
		if(mIsTerminal) std::cout << "terminate Reason : "<<terminationReason << std::endl;
	}

	skel->setPositions(p_save);
	skel->setVelocities(v_save);

}

std::vector<double>
Controller::
GetHeadingReward()
{

	dart::dynamics::SkeletonPtr skel = this->mCharacter->GetSkeleton();
	int dof = skel->getNumDofs();
	int num_body_nodes = skel->getNumBodyNodes();


	double sig_v = 0.5;
	double sig_dir = 0.3;


	// double target_speed = this->mTargetSpeed;

	// Eigen::Vector3d com = skel->getCOM();
	// double step_dur = skel->getTimeStep();
	// Eigen::Vector3d avg_vel = (com - this->COM_prev) / step_dur;

	// double avg_speed = this->target_dir.dot(avg_vel);

	// double vel_reward = 0;
	// double dir_reward = 0;
	// if (avg_speed > 0.0)
	// {
	// 	double vel_err = target_speed - avg_speed;
	// 	vel_reward = std::exp(-sig_v * vel_err * vel_err);
	// }


	// this->COM_prev = com;
	// double rew_v = vel_reward;


	double target_speed = this->mTargetSpeed;

	Eigen::Vector3d com = skel->getCOM();
	double step_dur = skel->getTimeStep();
	Eigen::Vector3d vel = skel->getCOMLinearVelocity();
	vel[1]=0;
	Eigen::Vector3d vel_dir = vel.normalized();
	double avg_speed = vel.norm();
	

	double dir_err = this->target_dir.dot(vel_dir) -1;

	double dir_reward = std::exp(-dir_err * dir_err/(sig_dir * sig_dir));

	double vel_reward = 0;
	if (avg_speed > 0.0)
	{
		double vel_err = target_speed - avg_speed;
		vel_reward = std::exp(- vel_err * vel_err / (sig_v * sig_v));
	}

	double rew_v = vel_reward * dir_reward;


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
	// Eigen::VectorXd p = skel->getPositions();
	// Eigen::VectorXd v = skel->getVelocities();
	// std::vector<double> tracking_rewards_bvh = this->GetTrackingReward(p, mTargetPositions,
	// 							 v, mTargetVelocities,  true);
	std::vector<double> task_rewards = this->GetHeadingReward();
	// double accum_bvh = std::accumulate(tracking_rewards_bvh.begin(), tracking_rewards_bvh.end(), 0.0) / tracking_rewards_bvh.size();

	this->mRewardParts.resize(mRewardLabels.size(), 0.0);
	this->mRewardParts.clear();
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
		Eigen::Vector3d pos_diff = this->target_dir;
		//pos_diff[1] = 0;

		// double tar_dist = pos_diff.norm();
		// if (tar_dist > 0.0001)
		// {
		// 	pos_diff = T_ref * pos_diff;
		// 	pos_diff /= tar_dist;
		// }
		// else
		// {
		// 	pos_diff = {0,0,0};
		// }



		// if(pos_diff.norm()<0.00001){
		// 	pos_diff= {0,0,1.0};
		// }
		// else{
		// 	pos_diff = pos_diff.normalized();	
		// }

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
	// 	double t = mReferenceManager->GetTimeStep(mCurrentFrame);

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
	auto &skel = this->mCharacter->GetSkeleton();

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
GetAgentParam(const Eigen::Matrix3d R_ref_inv){
	auto& skel = this->mCharacter->GetSkeleton();
	Eigen::Vector3d v_curr = skel->getCOMLinearVelocity();

	Eigen::Vector3d dir = R_ref_inv * (this->mPrevCOMvel + v_curr)/2;
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
	Eigen::Vector3d v_curr = skel->getCOMLinearVelocity();

	int n1 = mCurrentFrame-1 > 0 ? mCurrentFrame-1 : mCurrentFrame;
	Motion p_v_target = mReferenceManager->GetMotion(n1);
	Eigen::VectorXd p_ref = p_v_target.GetPosition();
	skel->setPositions(p_ref);

	Eigen::Isometry3d T_ref = this->getReferenceTransform();
	Eigen::Isometry3d T_ref_inv = T_ref.inverse();
	Eigen::Matrix3d R_ref_inv = T_ref_inv.linear();

	Eigen::Vector3d v_prev = skel->getCOMLinearVelocity();

	// int n0 = n1-1 > 0 ? n1-1 : n1;
	// p_v_target = mReferenceManager->GetMotion(n0);
	// p_ref = p_v_target.GetPosition();
	// skel->setPositions(p_ref);

	// Eigen::Vector3d v_com0 = R_ref_inv*skel->getCOMLinearVelocity();
	
	skel->setPositions(p_save);
	Eigen::Vector3d dir = R_ref_inv* (v_prev + v_curr)/2;
	if(dir.norm()<0.00001){
		dir ={0,1.0,0.0};
	}
	Eigen::Vector4d out;
	out.head(3) = dir.normalized();
	out[4] = dir.norm();
	
	return out;
}

void 
Controller::
Reset(bool RSI)
{
	this->mWorld->reset();
	
	this-> motion_it = std::rand()%this->mNumMotions;
	mReferenceManager->SelectMotion(motion_it);
	auto& skel = this->mCharacter->GetSkeleton();
	int n = skel->getNumBodyNodes();
	int dof = this->mCharacter->GetSkeleton()->getNumDofs(); 

	skel->clearConstraintImpulses();
	skel->clearInternalForces();
	skel->clearExternalForces();

	//RSI
	if(RSI) {
		this->mCurrentFrame = (int) dart::math::Random::uniform(0.0, mReferenceManager->GetPhaseLength()-5.0);
	}
	this->mStartFrame = this->mCurrentFrame;
	this->nTotalSteps = 0;
	this->mTimeElapsed = 0;

	Motion p_v_target = mReferenceManager->GetMotion(mCurrentFrame);
	this->mTargetPositions = p_v_target.GetPosition();
	this->mTargetVelocities = p_v_target.GetVelocity();

	skel->setPositions(mTargetPositions);
	skel->setVelocities(mTargetVelocities);


	this->mPrevAgentPose.setZero();
	this->mPrevAgentVel.setZero();
	this->mPrevAgentPose.resize(3*3*n + mEndEffectors.size()*3);
	this->mPrevAgentVel.resize(2*3*n);

	this->mPrevAgentPose = RecordPose();
	this->mPrevAgentVel = RecordVel();


	// this->mTargetPositions = Eigen::VectorXd::Zero(dof);
	// this->mTargetVelocities = Eigen::VectorXd::Zero(dof);
	// this->mTargetPositions[4] = mReferenceManager->GetInitialHeight()+0.03;

	// skel->setPositions(mTargetPositions);
	// skel->setVelocities(mTargetVelocities);

	this->mPrevCOMvel = skel->getCOMLinearVelocity();
	// this->mPPrevCOMvel = mPrevCOMvel;
	this->COM_prev = skel->getCOM();

	this->mPrevExpertPose = this->mPrevAgentPose;
	this->mPrevExpertVel = this->mPrevAgentVel;

	this-> mNumFeature = (mPrevAgentPose.rows()+mPrevAgentVel.rows())*2 + mNumMotionParam;
	this-> mAgentFeatureSet.setZero();
	this-> mExpertFeatureSet.setZero();
	this-> mAgentFeatureSet.resize(mNumFeature);
	this-> mExpertFeatureSet.resize(mNumFeature);


	this->mIsNanAtTerminal = false;
	this->mIsTerminal = false;

	this->mStartRoot = this->mCharacter->GetSkeleton()->getPositions().segment<3>(3);
	this->target_dir = skel->getCOMLinearVelocity();
	this->target_dir[1]=0;
	this->theta= 0;
	this->theta_prev=0;
	this->speed =1.0;

	SetTarget(this->mStartRoot, !mRecord);

}

void
Controller::
SetTarget(const Eigen::Vector3d& root_pos, bool random){
	double time_min = 1;
	double time_max = 5;
	

	this->mMinTargetDist = 1.0;
	this->mMaxTargetDist = 3.0;
	this->dist_threshold = 0.1;
 	
	std::srand(std::time(NULL));

	if(random){
		this->mTargetSpeed = 0.5 + (1.5-0.5) * double(std::rand())/RAND_MAX;
	}
	else{
		this->mTargetSpeed = this->speed;
	}

	double dist = mMinTargetDist + (mMaxTargetDist-mMinTargetDist) * double(std::rand())/RAND_MAX;
	if(random){
		float delta_th= -M_PI + 2 * M_PI * double(std::rand())/RAND_MAX;
	

		bool sharp_turn = double(std::rand())/RAND_MAX < 0.02;
		if(!sharp_turn){
			delta_th = -M_PI/6 + M_PI/3 * double(std::rand())/RAND_MAX;
		}
		this->theta += delta_th;
	}

	this->target_rot = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
	  * Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY())
	  * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitZ());
	



	// this->target_pos[0] = std::sin(this->theta);
	// if(random)
	// 	this->target_pos[1] = -0.2+0.4 * double(std::rand())/RAND_MAX;
	// else
	// 	this->target_pos[1] = this->height;
	// this->target_pos[2] = std::cos(this->theta);

}

void
Controller::
SetParam(const float new_theta, const float new_height, const float new_speed){
	this->theta = new_theta;
	this->height = new_height;
	this->speed = new_speed;
}


void
Controller::SaveDisplayedData(std::string directory, bool bvh) {

}

}