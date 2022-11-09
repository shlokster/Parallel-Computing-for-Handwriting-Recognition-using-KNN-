#pragma once

#include <utility>
#include <vector>

namespace {
    using namespace std;
}

class MpiWrapper
{
public:
    MpiWrapper();
    ~MpiWrapper();

    int getWorldRank();
    int getWorldSize();

    pair<vector<int>, vector<int>> calculateDisplacements(int totalSize);
private:
    int worldRank{0};
    int worldSize{0};
};
