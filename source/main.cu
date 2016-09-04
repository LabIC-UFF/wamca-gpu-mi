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

/*!
 * Initialize environment.
 */
void
envInit()
{
    int     error;

    // Initializes log system
    logInit();

    cudaDeviceProp  prop;
    int             count;

    // Set thread GPU
    cudaSetDevice(0);

    // Detect CUDA driver and GPU devices
    switch(cudaGetDeviceCount(&count)) {
    case cudaSuccess:
        for(int d;d < count;d++) {
            if(cudaGetDeviceProperties(&prop,d) == cudaSuccess) {
                if(prop.major < 2)
                    WARNING("Device '%s' is not suitable to this application. Device capability %d.%d < 2.0\n",
                            prop.name,prop.major,prop.minor);
            }
            lprintf("GPU ok!\n");
        }
        break;
    case cudaErrorNoDevice:
        WARNING("No GPU Devices detected.");
        break;
    case cudaErrorInsufficientDriver:
        WARNING("No CUDA driver installed.");
        break;
    default:
        EXCEPTION("Unknown error detecting GPU devices.");
    }
}

int main(int argc, char **argv)
{
	envInit();

	bool costTour = true;
	bool distRound = false;
	bool coordShift = false;
	//string instance_path = "./instances/01_berlin52.tsp";
	//string instance_path = "./instances/02_kroD100.tsp";
	//string instance_path = "./instances/03_pr226.tsp";
	//string instance_path = "./instances/04_lin318.tsp";
	//string instance_path = "./instances/05_TRP-S500-R1.tsp";
	//string instance_path = "./instances/06_d657.tsp";
	//string instance_path = "./instances/07_rat784.tsp";
	string instance_path = "./instances/08_TRP-S1000-R1.tsp";

	//string instance_path = "./instances/08_TRP-S1000-R1.tsp";

	MLProblem problem(costTour, distRound, coordShift);
	problem.load(instance_path.c_str());


	int seed = 100; // 0: random
	WAMCAExperiment exper(problem, seed);
	exper.runWAMCA2016();


    l4printf(">>>> BUILT AT %s %s\n",__DATE__,__TIME__);

    lprintf("finished successfully\n");

    return 0;
}
