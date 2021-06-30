
#ifdef  WIN32
	#include <windows.h>
#endif
#include <getopt.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/glut.h>
#include <vector>
#include <string>

#include <experimental/filesystem>

#include "MainInterface.h"

int main(int argc, char ** argv)
{
	std::cout<<"S : Play"<<std::endl;
	std::cout<<"ESC : exit"<<std::endl;

	std::string ppo="", bvh="", amp="";
       static struct option long_options[] =
    {
    {"bvh", required_argument,        0, '1'},
    {"ppo", required_argument,        0, '2'},
    {"amp", required_argument,        0, '3'},
    }; 
  	int option_index = 0;
  	while (1)
    {
  		auto c = getopt_long (argc, argv, "1:23::",
                       long_options, &option_index);
      	if (c == -1)
        	break;
      	switch (c)
      	{
	           case 0:
	             break;

	           case '1':
	             bvh= optarg;
	             break;

	           case '2':
	              ppo = optarg;

              case '3':
                amp = optarg;
	   			  break; 
       	}
    }

	glutInit(&argc, argv);	
	MainInterface* interface = new MainInterface(bvh, ppo,amp);
    interface->GLInitWindow("Motion Control");
	glutMainLoop();
	return 0;
}
