#include "ElasticPig.hpp"
#include <chrono>

// g++ main.cpp -o main.exe -O2

int main()
{
    ElasticPig::Solver solver;
    std::cout << "here " << std::endl;
    solver.ReadMeshInit("testBar2.txt");
    std::cout << "here " << std::endl;
    solver.InitMassVol();
    solver.InitBaseBC();
    solver.InitMatrix();


    for (int iStep = 1; iStep <= 1000; iStep++)
    {
        auto ret = solver.TimeStepImplicit(0.1, 1e-2, 100);
        std::cout << "Step " << iStep << " iters used: " << ret.nStep << " convergence: " << ret.thRel << std::endl;
    }

    return 0;
}