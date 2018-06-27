#ifndef __wamca_dvnd_h
#define __wamca_dvnd_h

#include <iostream>
#include <ctime>
#include <map>
#include <vector>

#include <stdio.h>
#include <unistd.h>

#include "WamcaExperiment.hpp"
#include "dvnd.cuh"

using namespace std;

void envInit(int deviceNumber);

extern "C" unsigned int applyMoves(char * file, int *solution, unsigned int solutionSize, unsigned int useMoves = 0, unsigned short *ids = NULL,
		unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL);
WAMCAExperiment * getExperiment(MLProblem * problem, unsigned int hostCode = 0, int seed = 500);
extern "C" int getNoConflictMoves(unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL,
		int *selectedMoves = NULL, int *impValue = NULL, bool maximize = false);
MLProblem * getProblem(char * file, unsigned int hostCode = 0);
MLSolution* getSolution(MLProblem * problem, int *solution, unsigned int solutionSize);
void move64ToVectors(MLMove64 *moves, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL, unsigned int size = 0);
void removeProblem(MLProblem * problem);
void removeExperiment(WAMCAExperiment * exper);
MLMove64 * vectorsToMove64(unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL);
MLMove * vectorsToMove(unsigned int useMoves = 0, unsigned short *ids = NULL, unsigned int *is = NULL, unsigned int *js = NULL, int *costs = NULL);

#endif
