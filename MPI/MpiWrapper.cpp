#include "MpiWrapper.hpp"

#include <mpi.h>
#include <iostream>

using namespace std;

MpiWrapper::MpiWrapper()
{
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
}

MpiWrapper::~MpiWrapper()
{
  MPI_Finalize();
}

int MpiWrapper::getWorldRank()
{
  return worldRank;
}

int MpiWrapper::getWorldSize()
{
  return worldSize;
}

pair<vector<int>, vector<int>> MpiWrapper::calculateDisplacements(int totalSize)
{
  vector<int> sendCounts;
  vector<int> displacements;

  int initSize = totalSize / worldSize;
  for (int i = 0; i < worldSize; ++i)
  {
    sendCounts.push_back(initSize);
  }
  int sum = worldSize * initSize;
  int i = 0;
  while (sum != totalSize)
  {
    ++sendCounts.at(i % totalSize);
    ++i;
    ++sum;
  }

  int startIndex = 0;
  for (int i = 0; i < worldSize; ++i)
  {
    displacements.push_back(startIndex);
    startIndex += sendCounts.at(i);
  }

  return make_pair(sendCounts, displacements);
}
