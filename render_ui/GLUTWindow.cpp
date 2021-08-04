
#include "GLUTWindow.h"
#include <iostream>

std::vector<GLUTWindow*> GLUTWindow::mWindows;
std::vector<int> GLUTWindow::mWinIDs;

// Data
static double       g_Time = 0.0f;
static bool         g_MousePressed[3] = { false, false, false };
static float        g_MouseWheel = 0.0f;
static GLuint       g_FontTexture = 0;


GLUTWindow::
GLUTWindow()
	:mCamera(new Camera()),drag_mouse_l(0),drag_mouse_r(0),last_mouse_y(0),last_mouse_x(0),mDisplayTimeout(1.0/30.0)
{
	this->frame_no = 0;
}
GLUTWindow::
~GLUTWindow()
{

}

void
GLUTWindow::
GLInitWindow(const char* _name)
{

	mWindows.push_back(this);
	#ifdef __FREEGLUT_EXT_H__
    	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	#endif
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE);
	// glutInitDisplayMode(GLUT_DEPTH |GLUT_DOUBLE | GLUT_RGBA | GLUT_STENCIL);
	//glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE | GLUT_ACCUM);
	glutInitWindowSize(1920, 1080);
	glutInitWindowPosition(0, 0);
	mWinIDs.push_back(glutCreateWindow("Physics-based Character Controller"));

	GLenum err=glewInit();
	if(err!=GLEW_OK) {
	// Problem: glewInit failed, something is seriously wrong.
		std::cout << "glewInit failed: " << glewGetErrorString(err) << std::endl;
		exit(1);
	}

	    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    ImGui::StyleColorsDark();

    // Setup Dear ImGui style
    //ImGui::StyleColorsClassic();
    ImGui_ImplGLUT_Init();
	glutDisplayFunc(DisplayEvent);
	glutReshapeFunc(ReshapeEvent);
	glutKeyboardFunc(KeyboardEvent);
	glutSpecialFunc(SKeyboardEvent);
	glutMouseFunc(MouseEvent);
	
	glutMotionFunc(MotionEvent);
	glutTimerFunc(mDisplayTimeout, TimerEvent, 0);
    ImGui_ImplOpenGL2_Init();

	GLinitEnvironment();



}


void 
GLUTWindow::
GLinitEnvironment()
{
	float  light0_position[] = { 10.0, 10.0, 10.0, 1.0 };
	float  light0_diffuse[] = { 0.8, 0.8, 0.8, 1.0 };
	float  light0_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	float  light0_ambient[] = { 0.1, 0.1, 0.1, 1.0 };
	glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light0_specular);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
	glEnable(GL_LIGHT0);

	glEnable(GL_LIGHTING);

	glEnable(GL_COLOR_MATERIAL);

	glEnable(GL_DEPTH_TEST);

	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);

	// sky color
	glClearColor(0.95, 0.95, 0.95, 0.0);

	//ani_start = clock();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);

	glEnable(GL_FOG);
	GLfloat fogColor[] = {0.9,0.9,0.9,1};
	glFogfv(GL_FOG_COLOR,fogColor);
	glFogi(GL_FOG_MODE,GL_LINEAR);
	glFogf(GL_FOG_DENSITY,0.05);
	glFogf(GL_FOG_START,20.0);
	glFogf(GL_FOG_END,60.0);
}


void 
GLUTWindow::
glRecordInitialize()
{
	RecorderConfig cfg;
    cfg.m_triple_buffering = 1;
    cfg.m_record_audio = 1;
    cfg.m_width = 1920;
    cfg.m_height = 1080;
    cfg.m_video_format = OGR_VF_VP8;
    cfg.m_audio_format = OGR_AF_VORBIS;
    cfg.m_audio_bitrate = 112000;
    cfg.m_video_bitrate = 200000;
    cfg.m_record_fps = 30;
    cfg.m_record_jpg_quality = 90;
    ogrInitConfig(&cfg);
    ogrRegReadPixelsFunction(glReadPixels);
    ogrRegPBOFunctions(glGenBuffers, glBindBuffer, glBufferData,glDeleteBuffers, glMapBuffer, glUnmapBuffer);

    ogrSetSavedName("Render");
}

inline GLUTWindow*
GLUTWindow::
current()
{
	int id = glutGetWindow();
	for (int i = 0; i < mWinIDs.size(); i++)
	{
		if (mWinIDs.at(i) == id) {
			return mWindows.at(i);
		}
	}
	std::cout << "An unknown error occurred!" << std::endl;
	exit(0);
}


void
GLUTWindow::
DisplayEvent()
{
	current()->display();
}
void
GLUTWindow::
KeyboardEvent(unsigned char key,int x,int y)
{
	current()->keyboard(key,x,y);
	ImGui_ImplGLUT_KeyboardFunc(key,x,y);
}
void
GLUTWindow::
SKeyboardEvent(int key,int x,int y)
{
	current()->skeyboard(key,x,y);
	ImGui_ImplGLUT_SpecialFunc(key,x,y);
}
void
GLUTWindow::
MouseEvent(int button, int state, int x, int y)
{
	current()->mouse(button,state,x,y);
	ImGui_ImplGLUT_MouseFunc(button,state,x,y);
}
void
GLUTWindow::
MotionEvent(int x, int y)
{
	current()->motion(x,y);
	ImGui_ImplGLUT_MotionFunc(x,y);
}
void
GLUTWindow::
ReshapeEvent(int w, int h)
{
	current()->reshape(w,h);
	ImGui_ImplGLUT_ReshapeFunc(w,h);
}
void
GLUTWindow::
TimerEvent(int value)
{
	current()->Timer(value);
}