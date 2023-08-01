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

    ElasticPig::ModelMeshInterpolator interpolator;
    ElasticPig::TVecs modelPoints;
    modelPoints.resize(3, 3);
    modelPoints(Eigen::all, 0) << 0, 0, 0;
    modelPoints(Eigen::all, 1) << 0, 0, 2.5;
    modelPoints(Eigen::all, 2) << 0, 0, 5;
    interpolator.assignModelPoints(modelPoints);
    interpolator.buildMappingIndex(ElasticPig::TVecCloud{solver.getNodes()});

    for (int iStep = 1; iStep <= 1000; iStep++)
    {
        interpolator.interpolateForce2Mesh(solver.getFExt());
        auto ret = solver.TimeStepImplicit(0.1, 1e-2, 100);
        interpolator.interpolateDisp2Model(solver.getU());
        std::cout << "Step " << iStep << " iters used: " << ret.nStep << " convergence: " << ret.thRel
                  << " pos 2 " << interpolator.getModelDisp()(Eigen::all, 2).transpose() << std::endl;
    }

    return 0;
}