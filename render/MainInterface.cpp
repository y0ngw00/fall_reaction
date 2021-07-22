
#include "MainInterface.h"
#include <iostream>


MainInterface::
MainInterface(std::string bvh, std::string ppo, std::string amp):GLUTWindow()
{
	std::srand(std::time(NULL));
	this->character_path = std::string(PROJECT_DIR)+std::string("/character/") + std::string(REF_CHARACTER_TYPE) + std::string(".xml");

	this->frame_no = 0;
	mCamera = new Camera();

	this->drag_mouse_r=0;
	this->drag_mouse_l=0;
	this->last_mouse_x=0;
	this->last_mouse_y=0;

	this->mDisplayTimeout=33;
	this->begin = std::chrono::steady_clock::now();
	this->on_animation= false;

	this->speed_type=0;


	DPhy::Character* ref = new DPhy::Character(character_path);
    mReferenceManager = new DPhy::ReferenceManager(ref);


	this->mCurFrame = 0;
	//this->mTotalFrame = mReferenceManager->GetPhaseLength();
	this->mTotalFrame = 0;

	if(bvh!=""){
		int motion_it=0;
	    mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + bvh);
	    int mNumMotions = mReferenceManager->GetNumMotions();
    	motion_it = std::rand()%mNumMotions;
	    std::cout<<"Total number of motions : "<<mNumMotions<<std::endl;
	    std::cout<<"The motion ID is : "<<motion_it<<std::endl;
		mReferenceManager->SelectMotion(motion_it);

	    std::vector<Eigen::VectorXd> pos;
		
		phase = 0;
		int totalFrame_bvh = mReferenceManager->GetTotalFrameperMotion();

	    for(int i = 0; i < totalFrame_bvh; i++) {
	        Eigen::VectorXd p = mReferenceManager->GetPosition(phase);
	        p[3]-=1.5;
	        pos.push_back(p);
	        phase += mReferenceManager->GetTimeStep(phase);
	    }
		this->mSkel = DPhy::SkeletonBuilder::BuildFromFile(character_path).first;
		DPhy::SetSkeletonColor(mSkel, Eigen::Vector4d(164./255., 235./255.,	243./255., 1.0));

		this->render_bvh = true;
		this->play_bvh = true;

	    UpdateMotion(pos, "bvh");
	}
	std::cout<<1<<std::endl;
	if(ppo!=""){
		
		this->mSkel_sim = DPhy::SkeletonBuilder::BuildFromFile(character_path).first;

		DPhy::SetSkeletonColor(mSkel_sim, Eigen::Vector4d(164./255., 235./255.,	13./255., 1.0));
		initNetworkSetting("PPO",ppo);
		this->render_sim=true;
		this->play_sim = true;
	}
	else if(amp!=""){
		
		this->mSkel_amp = DPhy::SkeletonBuilder::BuildFromFile(character_path).first;

		DPhy::SetSkeletonColor(mSkel_amp, Eigen::Vector4d(38./255., 189./255.,	169./255., 1.0));

		initNetworkSetting("AMP",amp);
		this->render_amp=true;
		this->play_amp = true;

		this->mTargetPos.setZero();
	}
	

}

void
MainInterface::
display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	mCamera->viewupdate();

	DrawGround();
	DrawStrings();
	DrawSkeletons();
	glutSwapBuffers();

	
	if(this->isRecord)
		ogrCapture();

	
}
void
MainInterface::
SetFrame(int n)
{
	if(render_bvh && (n < mMotion_bvh.size()-1))
		mSkel->setPositions(mMotion_bvh[n]);
	else
		this->play_bvh = false;

	if(render_sim && (n < mMotion_sim.size()-1)) 
		mSkel_sim->setPositions(mMotion_sim[n]);
	else
		this->play_sim = false;

	if(render_amp && (n < mMotion_amp.size()-1)){
		mSkel_amp->setPositions(mMotion_amp[n]);
		this->mTargetPos = mMotion_tar[n];
	} 
		
	else
		this->play_amp = false;
}
void
MainInterface::
DrawStrings()
{
	std::vector<std::string> str_list;
	std::string frame = "Frame : " + std::to_string(this->mCurFrame);
	std::string reset = "Reset : R" ;
	std::string stop = "Stop : space" ;
	std::string prev = "Prev : Q" ;
	std::string next = "Next : P" ;
	std::string record = "Record : V" ;

	str_list.push_back(frame);
	str_list.push_back(reset);
	str_list.push_back(stop);
	str_list.push_back(prev);
	str_list.push_back(next);
	str_list.push_back(record);
	for(int i=0;i<str_list.size(); i++){
		GUI::DrawStringOnScreen(0.8, 0.9-0.025*i, str_list[i], true, Eigen::Vector3d::Zero());
	}
	
}

void
MainInterface::
DrawSkeletons()
{
	glPushMatrix();
	glTranslated(0.0, 0, 0);
	if(render_bvh)
		GUI::DrawSkeleton(this->mSkel, 0);
	if(render_sim)
		GUI::DrawSkeleton(this->mSkel_sim, 0);
	if(render_amp)
		GUI::DrawSkeleton(this->mSkel_amp, 0);

	glPushMatrix();
	glColor3f(0.2,0.2,0.2);
	glTranslated(this->mTargetPos[0],this->mTargetPos[1],this->mTargetPos[2]);
	GUI::DrawSphere(0.1);
	glPopMatrix();
	glPopMatrix();
}	

void
MainInterface::
DrawGround()
{
	float  size = 3.0f;
	int  num_x = 30, num_z = 30;
	double  ox, oz;

	// the tiled floor
	glBegin(GL_QUADS);
	glNormal3d(0.0, 1.0, 0.0);
	ox = -(num_x * size) / 2;
	for (int x = 0; x < num_x; x++, ox += size)
	{
		oz = -(num_z * size) / 2;
		for (int z = 0; z < num_z; z++, oz += size)
		{
			if (((x + z) % 2) == 0)
				glColor3f(1.0, 1.0, 1.0);
			else
				glColor3f(0.7, 0.7, 0.7);
			glVertex3d(ox, 0.0, oz);
			glVertex3d(ox, 0.0, oz + size);
			glVertex3d(ox + size, 0.0, oz + size);
			glVertex3d(ox + size, 0.0, oz);
		}
	}
	glEnd();
}

void
MainInterface::
initNetworkSetting(std::string type, std::string net) {

    Py_Initialize();
    try {
  //   	if(reg != "") {
		// 	p::object reg_main = p::import("regression");
	 //        this->mRegression = reg_main.attr("Regression")();
	 //        std::string path = std::string(PROJECT_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
	 //        this->mRegression.attr("initRun")(path, mReferenceManager->GetParamGoal().rows() + 1, mReferenceManager->GetDOF() + 1);
		// 	mRegressionMemory->LoadParamSpace(path + "param_space");
	 //        mParamRange = mReferenceManager->GetParamRange();
	       
	 //        path = std::string(PROJECT_DIR)+ std::string("/network/output/") + DPhy::split(reg, '/')[0] + std::string("/");
		// //	mRegressionMemory->SaveContinuousParamSpace(path + "param_cspace");
  //   	}
    	if(type.compare("PPO")==0) {
    		//if (reg!="") this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
    		this->mController = new DPhy::Controller(mReferenceManager, this->character_path,true); //adaptive=true, bool parametric=true, bool record=true
			//mController->SetGoalParameters(mReferenceManager->GetParamCur());
			py::object sys_module = py::module::import("sys");
			py::str module_dir = (std::string(PROJECT_DIR)+"/network").c_str();
			
			sys_module.attr("path").attr("insert")(1, module_dir);

    		py::object ppo_main = py::module::import("ppo");

			this->mPPO = ppo_main.attr("PPO")();

			std::string path = std::string(PROJECT_DIR)+ std::string("/network/output/") + net;
			this->mPPO.attr("initRun")(path,
									   this->mController->GetNumState(), 
									   this->mController->GetNumAction());

			RunPPO(type);
    	}
    	else if(type.compare("AMP")==0) {
    		//if (reg!="") this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
    		this->mController = new DPhy::Controller(mReferenceManager, this->character_path,true); //adaptive=true, bool parametric=true, bool record=true
			//mController->SetGoalParameters(mReferenceManager->GetParamCur());

			py::object sys_module = py::module::import("sys");
			py::str module_dir = (std::string(PROJECT_DIR)+"/network").c_str();
			sys_module.attr("path").attr("insert")(1, module_dir);

    		py::object amp_main = py::module::import("amp");

			this->mAMP = amp_main.attr("AMP")();

			std::string path = std::string(PROJECT_DIR)+ std::string("/network/output/") + net;

			this->mAMP.attr("initRun")(path,
									   this->mController->GetNumState(), 
									   this->mController->GetNumAction(), 
									   this->mController->GetNumFeature(),
									   this->mController->GetNumPose());

			RunPPO(type);
    	}

    
    } catch (const py::error_already_set&) {
        PyErr_Print();
    }    
}
void
MainInterface::
RunPPO(std::string type) {
	std::vector<Eigen::VectorXd> pos_sim;
	std::vector<Eigen::VectorXd> pos_amp;
	std::vector<Eigen::VectorXd> pos_tar;

	int count = 0;
	mController->Reset(false);
	this->mTiming= std::vector<double>();
	this->mTiming.push_back(this->mController->GetCurrentLength());

	if(type.compare("PPO")==0){
		while(!this->mController->IsTerminalState()) {
			Eigen::VectorXd state = this->mController->GetState();
			py::array_t<double> na = this->mPPO.attr("run")(DPhy::toNumPyArray(state));
			Eigen::VectorXd action = DPhy::toEigenVector(na, this->mController->GetNumAction());
			this->mController->SetAction(action);
			this->mController->Step();
			this->mTiming.push_back(this->mController->GetCurrentLength());
			
			count += 1;
		}

		for(int i = 0; i <= count; i++) {

			Eigen::VectorXd position = this->mController->GetPositions(i);
			//Eigen::VectorXd position_reg = this->mController->GetTargetPositions(i);
			// Eigen::VectorXd position_bvh = this->mController->GetBVHPositions(i);
			// position_bvh[3]-=1.5;
			// pos_bvh.push_back(position_bvh);

			//Eigen::VectorXd position_obj = this->mController->GetObjPositions(i);

			//pos_reg.push_back(position_reg);
			pos_sim.push_back(position);
			
			
		}
		UpdateMotion(pos_sim, "sim");
	}

	else if(type.compare("AMP")==0){
		while(!this->mController->IsTerminalState()) {
			Eigen::VectorXd state = this->mController->GetState();
			py::array_t<double> na = this->mAMP.attr("run")(DPhy::toNumPyArray(state));
			Eigen::VectorXd action = DPhy::toEigenVector(na, this->mController->GetNumAction());
			//for(int i=0;i<action.rows();i++)std::cout<<action[i]<<" ";
			//std::cout<<std::endl;

			this->mController->SetAction(action);
			this->mController->Step();
			this->mTiming.push_back(this->mController->GetCurrentLength());
			
			count += 1;
		}

		for(int i = 0; i <= count; i++) {

			Eigen::VectorXd position = this->mController->GetPositions(i);
			pos_amp.push_back(position);

			Eigen::VectorXd target_position = this->mController->GetTargetPositions(i);
			pos_tar.push_back(target_position);
			
			
		}
		UpdateMotion(pos_amp, "amp");
		UpdateMotion(pos_tar, "target");
	}



}
void 
MainInterface::
UpdateMotion(std::vector<Eigen::VectorXd> motion, const char* type)
{
	if(!strcmp(type,"bvh")) {
		mMotion_bvh = motion;	
	}
	else if(!strcmp(type,"sim")) {
		mMotion_sim = motion;		
	}
	else if(!strcmp(type,"amp")) {
		mMotion_amp = motion;	
	}
	else if(type == "target") {
		mMotion_tar = motion;	
	}
	 // else if(type == 4) {
	// 	mMotion_points_left = motion;
	// } else if(type == 5){
	// 	mMotion_obj = motion;
	// } else if(type == 6){
	// 	mMotion_points_right = motion;
	// }
	this->mCurFrame = 0;
	if(this->mTotalFrame == 0)
		this->mTotalFrame = motion.size();
	else if(mTotalFrame < motion.size())
		this->mTotalFrame = motion.size();
}


void
MainInterface::
motion(int mx, int my) 
{
	if ((drag_mouse_l==0) &&(drag_mouse_r==0))
		return;

	if (drag_mouse_r==1)
	{
		mCamera->Translate(mx,my,last_mouse_x,last_mouse_y);
	}
	else if (drag_mouse_l==1)
	{
		mCamera->Rotate(mx,my,last_mouse_x,last_mouse_y);
	}
	last_mouse_x = mx;
	last_mouse_y = my;
}





void 
MainInterface::
mouse(int button, int state, int mx, int my)
{
	if(button == 3 || button == 4){
		if (button == 3)
		{
			mCamera->Zoom(1);
		}
		else
		{
			mCamera->Zoom(-1);
		}
	}

	else if (state == GLUT_DOWN){
		if (button == GLUT_LEFT_BUTTON)
			drag_mouse_l = 1;
		else if(button == GLUT_RIGHT_BUTTON)
			drag_mouse_r = 1;
		last_mouse_x = mx;
		last_mouse_y = my;
	}

	else if(state == GLUT_UP){
		drag_mouse_l = 0;
		drag_mouse_r = 0;
	}

}
void
MainInterface::
Reset()
{
	this->mCurFrame = 0;
	//this->SetFrame(this->mCurFrame);
}




void 
MainInterface::
keyboard(unsigned char key, int mx, int my)
{
	if (key == 27) 
		exit(0);

	if (key == ' ') {
		if(!on_animation)
        	on_animation = true;
        else if(on_animation)
        	on_animation = false;
    }
    if (key == '1') 
		this->render_bvh = (this->render_bvh == false);
	if (key == '2')
		this->render_sim = (this->render_sim == false);
	if (key == 'r')
		Reset();
	// // change animation mode
	if (key == 'v') {
		if(!this->isRecord){
			std::cout<<"Recording Starts"<<std::endl;
        	this->isRecord = true;
        	glRecordInitialize();
        	ogrPrepareCapture();
        }
        else if(this->isRecord){
        	std::cout<<"Recording Ends"<<std::endl;
        	this->isRecord = false;
        	ogrStopCapture();
        }
    }

    if (key == 'q'){
    	if(mCurFrame>0) 
			this->mCurFrame--;
		else
			this->mCurFrame=0;
    }
	if (key == 'p')
    	if(mCurFrame<mTotalFrame-1) 
			this->mCurFrame++;
		else
			this->mCurFrame=mTotalFrame-1;
	if (key == 's') 
		glRecordInitialize();	
	// // change animation mode
	// if (key == 'a') {
	// 	animation_flag = (animation_flag + 1) % 5;
	// }
	// if (key == 'A') {
	// 	animation_flag = (animation_flag + 4) % 5;
	// }

	// // change animation scale
	// if (key == '_' || key == '-') {
	// 	animation_scale -= animation_scale_step;
	// 	if (animation_scale < animation_scale_min) animation_scale = animation_scale_min;
	// }
	// if (key == '+' || key == '=') {
	// 	animation_scale += animation_scale_step;
	// 	if (animation_scale > animation_scale_max) animation_scale = animation_scale_max;
	// }


}

void
MainInterface::
skeyboard(int key, int x, int y)
{

	// if(on_animation) {
 //        if (key == GLUT_KEY_LEFT) {
 //        	std::cout<<"LEFT"<<std::endl;
 //            if(speed_type ==0)return;
 //            BVH *temp_bvh = current_bvh;
 //            if (motion_type == 0)
 //                motion_type = mCharacter->getMotionrange() - 1;
 //            else
 //                motion_type--;
 //            current_bvh = mCharacter->getBVH(speed_type, motion_type);
 //            current_bvh = mCharacter->MotionBlend(frame_no, temp_bvh, current_bvh);
 //            begin = std::chrono::steady_clock::now();
 //        }
 //        if (key == GLUT_KEY_RIGHT) {
 //        	std::cout<<"RIGHT"<<std::endl;
 //            if(speed_type ==0)return;
 //            BVH *temp_bvh = current_bvh;
 //            if (motion_type == mCharacter->getMotionrange() - 1)
 //                motion_type = 0;
 //            else
 //                motion_type++;
 //            current_bvh = mCharacter->getBVH(speed_type, motion_type);
 //            current_bvh = mCharacter->MotionBlend(frame_no,temp_bvh, current_bvh);
 //            begin = std::chrono::steady_clock::now();
 //        }
 //        if (key == GLUT_KEY_UP) {
 //        	std::cout<<"UP"<<std::endl;
 //            BVH *temp_bvh = current_bvh;
 //            if (speed_type == mCharacter->getSpeedrange() - 1)
 //                speed_type = 0;
 //            else
 //                speed_type++;
 //            current_bvh = mCharacter->getBVH(speed_type, motion_type);
 //            current_bvh = mCharacter->MotionBlend(frame_no,temp_bvh, current_bvh);
 //            begin = std::chrono::steady_clock::now();
 //        }
 //        if (key == GLUT_KEY_DOWN) {
 //        	std::cout<<"DOWN"<<std::endl;
 //            BVH *temp_bvh = current_bvh;
 //            if (speed_type == 0)
 //                speed_type = mCharacter->getSpeedrange() - 1;
 //            else
 //                speed_type--;
 //            current_bvh = mCharacter->getBVH(speed_type, motion_type);
 //            current_bvh = mCharacter->MotionBlend(frame_no,temp_bvh, current_bvh);
 //            begin = std::chrono::steady_clock::now();
 //        }
 //    }

}


void
MainInterface::
Timer(int value) 
{
	std::chrono::steady_clock::time_point begin= std::chrono::steady_clock::now();
	//double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000000.;


	if(on_animation && (this->mCurFrame < this->mTotalFrame - 1) && (play_amp || play_sim || play_bvh)){
        this->mCurFrame++;
        SetFrame(this->mCurFrame);  	
    }
    SetFrame(this->mCurFrame);
	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.;
	

	glutTimerFunc(std::max(0.0,mDisplayTimeout-elapsed), TimerEvent,1);
	glutPostRedisplay();

}

void 
MainInterface::
reshape(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30, (double)w / h, 1, 300);
}