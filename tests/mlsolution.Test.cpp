#include <gtest/gtest.h>

// system
#include <limits>

#include "mlsolution.h"

using namespace std;

class TestBerlin52
{
private:
	static MLProblem *problem;

public:
	static MLProblem &getProblem()
	{
		if (!problem)
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

			problem = new MLProblem(costTour, distRound, coordShift);
			problem->load(instance_path.c_str());
		}
		return *problem;
	}
};
MLProblem * TestBerlin52::problem{nullptr};

TEST(mlsolution_test, CreateMLSolution_Berlin52)
{
	MLProblem& problem = TestBerlin52::getProblem();
	//
	MLSolution *solDevice = new MLSolution(problem, cudaHostAllocDefault);
	EXPECT_EQ(solDevice->clientCount, 0);
}

TEST(mlsolution_test, CreateMLSolution_B52_Random)
{
	MLProblem& problem = TestBerlin52::getProblem();
	//
	MLSolution *solDevice = new MLSolution(problem, cudaHostAllocDefault);
	EXPECT_EQ(solDevice->clientCount, 0);

	MTRandom rng;
   	rng.seed(0);

	solDevice->random(rng, 0.50);
	EXPECT_EQ(solDevice->clientCount, 53);
}

TEST(mlsolution_test, CreateMLSolution_B52_Random_to_GPU)
{
	MLProblem& problem = TestBerlin52::getProblem();
	//
	MLSolution *solDevice = new MLSolution(problem, cudaHostAllocDefault);
	EXPECT_EQ(solDevice->clientCount, 0);

	MTRandom rng;
   	rng.seed(0);

	solDevice->random(rng, 0.50);
	EXPECT_EQ(solDevice->clientCount, 53);
	// 
	solDevice->update();
	solDevice->ldsUpdate();
	//
	uint valor1 = solDevice->costCalc();
	EXPECT_EQ(valor1, 494309);
}
