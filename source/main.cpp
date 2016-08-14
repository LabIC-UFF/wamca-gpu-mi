/**
 * @file	main.cpp
 *
 * @brief	Application entry point.
 *
 * @author	Eyder Rios
 * @date	2015-05-28
 */

#include <unistd.h>
//#include <papi/papi.h>
#include <iostream>

#include "WamcaExperiment.hpp"

using namespace std;


int main(int argc, char **argv)
{

	bool costTour = false;
	bool distRound = false;
	bool coordShift = false;
	string instance_path = "./instances/01_berlin52.tsp";

	MLProblem problem(costTour, distRound, coordShift);
	problem.load(instance_path.c_str());

	WAMCAExperiment exper(problem);

	exper.runExperiment();

    l4printf(">>>> BUILT AT %s %s\n",__DATE__,__TIME__);

    return 0;
}
