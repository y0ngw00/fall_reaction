#ifndef __DEEP_PHYSICS_H__
#define __DEEP_PHYSICS_H__
#include "Controller.h"
#include "ReferenceManager.h"
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
#include <Eigen/Core>

#include <queue>
namespace DPhy
{
	class Controller;
}
namespace py = pybind11;
class SimEnv
{
public:
	
	SimEnv(int num_slaves, std::string ref, std::string training_path);
	//For general properties
	int GetNumState(){ return mNumState;}
	int GetNumAction(){	return mNumAction;}
	int GetNumPose(){ return mNumPose;}
	int GetNumFeature(){ return mNumFeature;}

	//For each slave
	void Step(int id);
	void Reset(int id,bool RSI);
	py::tuple IsNanAtTerminal(int id);

	py::array_t<double> GetState(int id);
	py::array_t<double> GetFeature(int id);
	void SetAction(py::array_t<double> np_array,int id);
	double GetReward(int id);
	py::array_t<double> GetRewardByParts(int id);

	//For controlling all slaves
	void Steps();
	void Resets(bool RSI);
	py::array_t<double> GetStates();

	py::array_t<double> GetExpertPoses();
	py::array_t<double> GetAgentPoses();
	void SetActions(py::array_t<double> np_array);
	py::list GetRewardLabels();
	py::array_t<double> GetRewards();
	py::array_t<double> GetRewardsByParts();

	double GetPhaseLength(){return mReferenceManager->GetPhaseLength();}
	int GetDOF(){ return mReferenceManager->GetDOF();}


private:
	std::vector<DPhy::Controller*> mSlaves;
	DPhy::ReferenceManager* mReferenceManager;

	int mNumSlaves;
	int mNumState;
	int mNumAction;
	int mNumFeature;
	int mNumPose;
	
	std::string mPath;
};


#endif