
#include "MainInterface.h"
#include <iostream>


MainInterface::
MainInterface(std::string bvh, std::string ppo):GLUTWindow()
{

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
	this->motion_type=0;

	
	DPhy::Character* ref = new DPhy::Character(character_path);
    mReferenceManager = new DPhy::ReferenceManager(ref);
    mReferenceManager->LoadMotionFromBVH(std::string("/motion/") + bvh);

    std::vector<Eigen::VectorXd> pos;
	
	phase = 0;

    for(int i = 0; i < 1000; i++) {
        Eigen::VectorXd p = mReferenceManager->GetPosition(phase);
        pos.push_back(p);
        phase += mReferenceManager->GetTimeStep(phase);
    }
    UpdateMotion(pos, "bvh");

	this->mCurFrame = 0;
	//this->mTotalFrame = mReferenceManager->GetPhaseLength();
	this->mTotalFrame = 1000;

	if(bvh!=""){
		this->mSkel = DPhy::SkeletonBuilder::BuildFromFile(character_path).first;
		DPhy::SetSkeletonColor(mSkel, Eigen::Vector4d(164./255., 235./255.,	243./255., 1.0));

		this->render_bvh = true;

	}
	

	if(ppo!=""){
		
		this->mSkel_sim = DPhy::SkeletonBuilder::BuildFromFile(character_path).first;

		DPhy::SetSkeletonColor(mSkel, Eigen::Vector4d(164./255., 235./255.,	13./255., 1.0));

		initNetworkSetting(ppo);
		this->render_sim=true;
		
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

	DrawSkeletons();
	glutSwapBuffers();

	
	if(this->isRecord)
		ogrCapture();

	//GUI::DrawStringOnScreen(0.8, 0.9, std::to_string(mCurFrame), true, Eigen::Vector3d::Zero());
}
void
MainInterface::
SetFrame(int n)
{
	if(render_bvh)
		mSkel->setPositions(mMotion_bvh[n]);
	if(render_sim) 
		mSkel_sim->setPositions(mMotion_sim[n]);
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
initNetworkSetting(std::string ppo) {

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
    	if(ppo != "") {
    		//if (reg!="") this->mController = new DPhy::Controller(mReferenceManager, true, true, true);
    		this->mController = new DPhy::Controller(mReferenceManager, this->character_path,true); //adaptive=true, bool parametric=true, bool record=true
			//mController->SetGoalParameters(mReferenceManager->GetParamCur());

			py::object sys_module = py::module::import("sys");
			py::str module_dir = (std::string(PROJECT_DIR)+"/network").c_str();
			sys_module.attr("path").attr("insert")(1, module_dir);

    		py::object ppo_main = py::module::import("ppo");
			this->mPPO = ppo_main.attr("PPO")();
			std::string path = std::string(PROJECT_DIR)+ std::string("/network/output/") + ppo;
			this->mPPO.attr("initRun")(path,
									   this->mController->GetNumState(), 
									   this->mController->GetNumAction());
			RunPPO();
    	}
    
    } catch (const py::error_already_set&) {
        PyErr_Print();
    }    
}
void
MainInterface::
RunPPO() {
	this->render_sim = true;
	std::vector<Eigen::VectorXd> pos_bvh;
	std::vector<Eigen::VectorXd> pos_reg;
	std::vector<Eigen::VectorXd> pos_sim;
	std::vector<Eigen::VectorXd> pos_obj;

	int count = 0;
	mController->Reset(false);
	this->mTiming= std::vector<double>();
	this->mTiming.push_back(this->mController->GetCurrentLength());

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
		Eigen::VectorXd position_bvh = this->mController->GetBVHPositions(i);
		position_bvh[3]-=1.5;

		//Eigen::VectorXd position_obj = this->mController->GetObjPositions(i);

		//pos_reg.push_back(position_reg);
		pos_sim.push_back(position);
		pos_bvh.push_back(position_bvh);
		
	}
	// Eigen::VectorXd root_bvh = mReferenceManager->GetPosition(0, false);
	// pos_sim =  DPhy::Align(pos_sim, root_bvh);
	// pos_reg =  DPhy::Align(pos_reg, root_bvh);
	UpdateMotion(pos_bvh, "bvh");
	UpdateMotion(pos_sim, "sim");
	//UpdateMotion(pos_reg, "reg");

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
	else if(!strcmp(type,"reg")) {
		mMotion_reg = motion;	
	}
	// else if(type == 3) {
	// 	mMotion_exp = motion;	
	// } else if(type == 4) {
	// 	mMotion_points_left = motion;
	// } else if(type == 5){
	// 	mMotion_obj = motion;
	// } else if(type == 6){
	// 	mMotion_points_right = motion;
	// }
	mCurFrame = 0;
	if(mTotalFrame == 0)
		mTotalFrame = motion.size();
	else if(mTotalFrame > motion.size()) {
		mTotalFrame = motion.size();
	}
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
        	ogrPrepareCapture();
        }
        else if(this->isRecord){
        	std::cout<<"Recording Ends"<<std::endl;
        	this->isRecord = false;
        	ogrStopCapture();
        }
    }
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


	if(on_animation && this->mCurFrame < this->mTotalFrame - 1){
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