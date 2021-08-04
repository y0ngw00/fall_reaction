
#include "MainInterface.h"
#include <iostream>


static bool show_demo_window = true;
static bool show_another_window = false;
static ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

MainInterface::
MainInterface(std::string bvh, std::string ppo, std::string amp):GLUTWindow()
{
	std::srand(std::time(0));
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
	this->guiDrawing = true;

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

		this->render_bvh = false;
		this->play_bvh = false;

	    mMotion_bvh = pos;
	}
	if(ppo!=""){
		
		this->mSkel_sim = DPhy::SkeletonBuilder::BuildFromFile(character_path).first;

		DPhy::SetSkeletonColor(mSkel_sim, Eigen::Vector4d(164./255., 235./255.,	13./255., 1.0));
		initNetworkSetting("PPO",ppo);
		this->render_sim=true;
		this->play_sim = true;
		Reset();
	}
	else if(amp!=""){
		
		this->mSkel_amp = DPhy::SkeletonBuilder::BuildFromFile(character_path).first;

		DPhy::SetSkeletonColor(mSkel_amp, Eigen::Vector4d(38./255., 189./255.,	169./255., 1.0));

		initNetworkSetting("AMP",amp);

		this->render_amp=true;
		this->play_amp = true;
		this->mTargetPos = {0,0,1};
		Reset();
	}


}



void
MainInterface::
display()
{

	ImGui_ImplOpenGL2_NewFrame();
    ImGui_ImplGLUT_NewFrame();

    ImGuiDisplay();
    // Rendering
    ImGui::Render();
    ImGuiIO& io = ImGui::GetIO();


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	mCamera->viewupdate();

	DrawGround();
	DrawStrings();
	DrawSkeletons();
	DrawArrows();

	 //glUseProgram(0); // You may want this if using this code in an OpenGL 3+ context where shaders may be bound, but prefer using the GL3+ code.
    ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

	glutSwapBuffers();

	
	if(this->isRecord)
		ogrCapture();

	
}

void
MainInterface::
ImGuiDisplay()
{
     // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
    if (show_demo_window)
        // ImGui::ShowDemoWindow(&show_demo_window);

    // // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
    {

        static int counter = 0;

        ImGui::Begin("Parameter status!");                          // Create a window called "Hello, world!" and append into it.

                    // Display some text (you can use a format strings too)
        // ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        // ImGui::Checkbox("Another Window", &show_another_window);
        ImGui::Text("Forwarding direction");
        // ImGui::Checkbox("Another Window", &show_another_window);
   
        ImGui::SliderFloat("Theta", &theta, -M_PI/6, M_PI/6);            // Edit 1 float using a slider from 0.0f to 1.0f
        
        ImGui::Text("Forwarding height");   
        ImGui::SliderFloat("Height", &height, -0.3f, 0.3f);

        ImGui::Text("Speed");   
        ImGui::SliderFloat("Speed", &speed, 0.5f, 1.5f);

        // ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        // if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
        //     counter++;
        // ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();
    }

    // // 3. Show another simple window.
    if (show_another_window)
    {
        ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
        if (ImGui::Button("Close Me"))
            show_another_window = false;
        ImGui::End();
    }
}



void
MainInterface::
SetFrame(int n)
{
	if(render_bvh)
		mSkel->setPositions(mMotion_bvh[n]);
	else
		this->play_bvh = false;

	if(render_sim && play_sim){
		Run("ppo");
		mSkel_sim->setPositions(this->mMotion_sim);
	}

	if(render_amp && play_amp){
		Run("amp");
		mSkel_amp->setPositions(this->mMotion_amp);
	} 

		
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
	// str_list.push_back(prev);
	// str_list.push_back(next);
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

	// glPushMatrix();
	// glColor3f(0.2,0.2,0.2);
	// glTranslated(this->mTargetPos[0],this->mTargetPos[1],this->mTargetPos[2]);
	// GUI::DrawSphere(0.1);
	// glPopMatrix();
	glPopMatrix();
}

void
MainInterface::
DrawArrows()
{
	int targetNode = 0;
	Eigen::Vector3d force_dir = this->mTargetPos;

	Eigen::Vector3d color = {255,0,0};

	Eigen::Vector3d origin = mSkel_amp->getBodyNode(targetNode)->getWorldTransform().translation();
	// GUI::DrawArrow3D(origin, force_dir,
 //            length, thickness,color);
	GUI::DrawArrow3D(origin, force_dir.normalized(),0.5, 0.03,color);
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
			this->mController->Reset(false);
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
			this->mController->Reset(false);
    	}

    
    } catch (const py::error_already_set&) {
        PyErr_Print();
    }    
}

void
MainInterface::
Run(std::string type){

	if(this->mController->IsTerminalState()) {
		play_sim = false;
		play_amp = false;
		
	}
	else{
		Eigen::VectorXd state = this->mController->GetState();
		py::array_t<double> na = this->mAMP.attr("run")(DPhy::toNumPyArray(state));

		Eigen::VectorXd action = DPhy::toEigenVector(na, this->mController->GetNumAction());
		//for(int i=0;i<action.rows();i++)std::cout<<action[i]<<" ";
		//std::cout<<std::endl;

		this->mController->SetAction(action);
		this->mController->Step();

		Eigen::VectorXd position = this->mController->GetSkeleton()->getPositions();
		if(type.compare("ppo")==0){
			this->mMotion_sim = position;


		}

		else if(type.compare("amp")==0){
			this->mMotion_amp = position;
			this->mController->SetParam(this->theta, this->height, this->speed);
			Eigen::VectorXd target_position = this->mController->GetTargetPosition();
			this->mTargetPos = target_position;
			if(this->mCurFrame==0){
				this->init_pos = position.segment<3>(3);
			}

			this->mMotion_amp[3]-=init_pos[0];
			this->mMotion_amp[5]-=init_pos[2];
		}
	
	}

}

void
MainInterface::
motion(int mx, int my) 
{
	if ((drag_mouse_l==0) &&(drag_mouse_r==0))
		return;

	auto& io = ImGui::GetIO();
	if (drag_mouse_r==1)
	{
		if(!io.WantCaptureMouse)
			mCamera->Translate(mx,my,last_mouse_x,last_mouse_y);
	}
	else if (drag_mouse_l==1)
	{
		if(!io.WantCaptureMouse)
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
		last_mouse_x = mx;
		last_mouse_y = my;

		if (button == GLUT_LEFT_BUTTON)
			drag_mouse_l = 1;
		else if(button == GLUT_RIGHT_BUTTON)
			drag_mouse_r = 1;

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
	ResetParam();
	this->play_amp = true;
	this->mController->Reset(false);
	this->SetFrame(this->mCurFrame);

}

void
MainInterface::
ResetParam(){
	this->height = 0.0f;
	this->theta = 0.0f;
	this->speed = 1.0f;
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

 //    if (key == 'q'){
 //    	if(mCurFrame>0) 
	// 		this->mCurFrame--;
	// 	else
	// 		this->mCurFrame=0;
 //    }
	// if (key == 'p')
 //    	if(mCurFrame<mTotalFrame-1) 
	// 		this->mCurFrame++;
	// 	else
	// 		this->mCurFrame=mTotalFrame-1;
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


	if(on_animation && (play_amp || play_sim || play_bvh)){
        this->mCurFrame++;
        SetFrame(this->mCurFrame);  	
    }
    
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