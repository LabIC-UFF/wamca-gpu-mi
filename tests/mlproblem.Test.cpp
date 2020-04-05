#include <gtest/gtest.h>

// system
#include <limits>

#include "mlproblem.h"

using namespace std;


TEST(mlproblem_test, TestSeedMTrand_seed0_unlimited)
{
   bool costTour = true;
	bool distRound = false;
	bool coordShift = false;
	string instance_path = "../instances/01_berlin52.tsp";
	//string instance_path = "./instances/02_kroD100.tsp";
	//string instance_path = "./instances/03_pr226.tsp";
	//string instance_path = "./instances/04_lin318.tsp";
	//string instance_path = "./instances/05_TRP-S500-R1.tsp";
	//string instance_path = "./instances/06_d657.tsp";
	//string instance_path = "./instances/07_rat784.tsp";
	//string instance_path = "./instances/08_TRP-S1000-R1.tsp";

	MLProblem problem(costTour, distRound, coordShift);
	problem.load(instance_path.c_str());
   EXPECT_EQ(problem.size, 53);
}
