#pragma once

#include <cstdint>
#include <vector>
#include "eigen-3.4.0/Eigen/Dense"
#include <iostream>
#include <fstream>
#include <string>
#include "eigen-3.4.0/Eigen/SparseLU"
#include <exception>
#include <array>
#include <functional>

namespace ElasticPig
{
    using num = double;
    using ind = std::int32_t;
    using TVecs = Eigen::Matrix<num, 3, Eigen::Dynamic>;
    using TFlag = Eigen::Matrix<num, 3, Eigen::Dynamic>;
    using TTets = Eigen::Matrix<ind, 4, Eigen::Dynamic>;
    using TDof = Eigen::Vector<num, Eigen::Dynamic>;
    using TVecsLocal = Eigen::Matrix<num, 3, 4>;
    using TJac = Eigen::Matrix<num, 3, 3>;

    class BadMeshException : public std::exception
    {
        virtual const char *what() const throw()
        {
            return "Mesh is Bad";
        }
    } BadMesh;

    static const TVecsLocal DiBj{
        {-1, 1, 0, 0},
        {-1, 0, 1, 0},
        {-1, 0, 0, 1},
    };

    class Solver
    {
        TVecs nodes;
        TVecs u, v, f, fExt;
        TFlag flags;
        TVecs mass;
        TVecs invMass;

        TTets elems;

        Eigen::Vector<num, Eigen::Dynamic> vol;
        std::vector<TJac> JacS;
        std::vector<TJac> invJasS;

        Eigen::Matrix<num, Eigen::Dynamic, Eigen::Dynamic> KMat_Dense;
        Eigen::SparseMatrix<num> KMat, IMat;
        Eigen::SparseLU<decltype(KMat)> LUSolver;

        bool useDense = false;

        num eta = 1e-5;
        Eigen::Matrix<num, 6, 6> C = Eigen::Matrix<num, 6, 6>::Identity() * 1e2;

    public:
        ind getNumElem()
        {
            return elems.cols();
        }

        ind getNumNode()
        {
            return nodes.cols();
        }

        void ReadMeshInit(const std::string &fName)
        {
            std::cout << fName << std::endl;
            std::ifstream file;
            file.open(fName);

            if (!file)
                std::abort();
            int nNode, nElem;
            file >> nNode >> nElem;
            std::cout << nNode << " " << nElem << std::endl;

            nodes.setZero(3, nNode);
            elems.setConstant(4, nElem, -1);
            u = v = fExt = f = nodes;
           

            for (auto &i : nodes.reshaped())
                file >> i;
            for (auto &i : elems.reshaped())
                file >> i;
            elems.array() -= 1; // to zero-based

            v(1, Eigen::all) = nodes(2, Eigen::all) * 0.1;// init condition
        }

        void InitMassVol()
        {
            vol.resize(this->getNumElem());
            mass.setZero(3, this->getNumNode());
            JacS.resize(this->getNumElem());
            invJasS.resize(this->getNumElem());

            for (ind iE = 0; iE < this->getNumElem(); iE++)
            {
                auto elem2node = elems(Eigen::all, iE);
                TVecsLocal coords = nodes(Eigen::all, elem2node);
                TJac Dx_i_Dxii_j = coords * DiBj.transpose();
                num JDet = Dx_i_Dxii_j.determinant();
                vol[iE] = JDet / 6.;
                JacS[iE] = Dx_i_Dxii_j;
                auto dec = Dx_i_Dxii_j.fullPivLu();
                if (!dec.isInvertible())
                    throw BadMesh;
                invJasS[iE] = dec.inverse();
                mass(Eigen::all, elem2node).array() += vol[iE] / 4.;
            }
            invMass = mass.array().inverse();
        }

        void InitBaseBC()
        {
            flags.setConstant(3, this->getNumNode(), 1);
            for (ind iN = 0; iN < this->getNumNode(); iN++)
            {
                if (std::abs(nodes(2, iN) - 0.0) < 1e-5)
                    flags(Eigen::all, iN).setConstant(0);
            }
        }

        void InitMatrix()
        {
            if (useDense)
                KMat_Dense.resize(this->getNumNode() * 3, this->getNumNode() * 3);
            else
            {
                std::vector<Eigen::Triplet<num>> triplets;
                triplets.reserve((3 * 4 * this->getNumElem()));
                for (ind iE = 0; iE < this->getNumElem(); iE++)
                {
                    auto elem2node = elems(Eigen::all, iE);
                    Eigen::Matrix<ind, 3, 4> iDofs;
                    iDofs(0, Eigen::all) = elem2node.transpose().array() * 3 + 0;
                    iDofs(1, Eigen::all) = elem2node.transpose().array() * 3 + 1;
                    iDofs(2, Eigen::all) = elem2node.transpose().array() * 3 + 2;
                    for (auto iDof : iDofs.reshaped())
                        for (auto jDof : iDofs.reshaped())
                            triplets.push_back(Eigen::Triplet<num>(iDof, jDof, 1));
                }
                KMat.resize(this->getNumNode() * 3, this->getNumNode() * 3);
                KMat.setFromTriplets(triplets.begin(), triplets.end());
                LUSolver.analyzePattern(KMat);

                IMat.resize(this->getNumNode() * 3, this->getNumNode() * 3);
                IMat.setIdentity();
            }
        }

        void ConstructRHSJacobi(TVecs &ff, TVecs &uc, TVecs &vc, bool getJacobi)
        {
            ff = (fExt.array() * mass.array()).matrix();
            ff -= (v.array() * mass.array() * eta).matrix();

            if (getJacobi)
            {
                if (useDense)
                    KMat_Dense.setZero();
                else
                    KMat.operator*=(0.0);
            }

            for (ind iE = 0; iE < this->getNumElem(); iE++)
            {
                auto elem2node = elems(Eigen::all, iE);

                Eigen::Matrix<ind, 3, 4> iDofs;
                iDofs(0, Eigen::all) = elem2node.transpose().array() * 3 + 0;
                iDofs(1, Eigen::all) = elem2node.transpose().array() * 3 + 1;
                iDofs(2, Eigen::all) = elem2node.transpose().array() * 3 + 2;
                // TVecsLocal coords_0 = nodes(Eigen::all, elem2node);
                TVecsLocal coords_d = uc(Eigen::all, elem2node);
                Eigen::Matrix<num, 3, 4> flagsLocal = flags(Eigen::all, elem2node);

                // TJac Dx0_i_Dxii_j = coords_0 * DiBj.transpose();
                // TJac Du_i_Dxii_j = coords_d * DiBj.transpose();
                Eigen::Matrix<num, 4, 3>
                    B0 = DiBj.transpose() * invJasS[iE];
                TJac Du_i_Dx0_j = coords_d * B0;

                TJac E_Green = (Du_i_Dx0_j.transpose() + Du_i_Dx0_j +
                                Du_i_Dx0_j.transpose() * Du_i_Dx0_j) *
                               0.5;
                Eigen::Vector<num, 6> E6_Green = E_Green.reshaped()({0, 4, 8, 1, 5, 2});

                using tp = std::pair<int, int>;
                static const std::array<tp, 6> i62ij{
                    tp{0, 0}, tp{1, 1}, tp{2, 2},
                    tp{0, 1}, tp{1, 2}, tp{2, 0}};

                Eigen::Matrix<num, 12, 6> dE6_du;
                for (int i6 = 0; i6 < 6; i6++)
                {
                    int i = i62ij[i6].first;
                    int j = i62ij[i6].second;
                    Eigen::Matrix<num, 3, 4> dE6_duP;
                    dE6_duP.setZero();
                    dE6_duP(i, Eigen::all) += B0(Eigen::all, j).transpose();
                    dE6_duP(j, Eigen::all) += B0(Eigen::all, i).transpose();
                    dE6_duP += (coords_d * B0(Eigen::all, j)) * B0(Eigen::all, i).transpose();
                    dE6_duP += (coords_d * B0(Eigen::all, i)) * B0(Eigen::all, j).transpose();
                    dE6_du(Eigen::all, i6) = dE6_duP.reshaped();
                }
                dE6_du.array() *= 0.5;

                Eigen::Vector<num, 12> FLocal = (E6_Green.transpose() * C * dE6_du.transpose()).transpose();
                // FLocal.array() *= flagsLocal.reshaped().array();
                (ff.reshaped()(iDofs.reshaped())).array() -= FLocal.array() * vol[iE];

                if (getJacobi)
                {
                    Eigen::Matrix<num, 12, 12> KLocal = dE6_du * C * dE6_du.transpose();
                    num KLM = KLocal.array().abs().maxCoeff() + 1e-100;
                    for (int iD = 0; iD < 12; iD++)
                        if (flagsLocal.reshaped()[iD] == 0)
                        {
                            KLocal(Eigen::all, iD).setZero();
                            KLocal(iD, Eigen::all).setZero();
                            KLocal(iD, iD) = KLM;
                        }
                    // std::cout << iDofs << std::endl;
                    // std::cout
                    //     << KLocal << std::endl;
                    // std::abort();
                    if (useDense)
                        KMat_Dense(iDofs.reshaped(), iDofs.reshaped()) -= KLocal * vol[iE];
                    else
                        for (int iD = 0; iD < 12; iD++)
                            for (int jD = 0; jD < 12; jD++)
                                KMat.coeffRef(iDofs.reshaped()[iD], iDofs.reshaped()[jD]) -=
                                    KLocal(iD, jD) * vol[iE];
                }
            }
            ff.array() *= flags.array();
        }

        struct TSRet
        {
            int nStep = 0;
            num thRel = 1.;
        };

        TSRet TimeStepImplicit(
            num dt, num threshold = 1e-2, int nStep = 10, num alpha = 0.0,
            num maxDU = 0.5, num maxDV = 50)
        {
            this->ConstructRHSJacobi(f, u, v, true);
            // std::cout << this->KMat_Dense << std::endl;
            // std::abort();

            TVecs uold = u;
            TVecs vold = v;
            TVecs fold = f;
            TSRet ret;

            num duBase = 1e100;
            // std::cout << u(1, 53) << std::endl;

            for (int iter = 1; iter <= nStep; iter++)
            {
                TDof R1 = alpha * vold.reshaped() + (1 - alpha) * v.reshaped() - (u - uold).reshaped() * (1. / dt);
                TDof R2 = ((alpha * fold.reshaped() + (1 - alpha) * f.reshaped()).array() * invMass.reshaped().array()).matrix() -
                          (v - vold).reshaped() * (1. / dt);
                num JCB_A = -1. / dt;
                num JCB_B = 1 - alpha;
                num JCB_C = -1. / dt - eta * (1 - alpha);
                // std::cout << R1 << "=============" << std::endl;
                // std::cout << R2 << std::endl;
                // std::abort();

                TDof du, dv;

                if (useDense)
                {
                    decltype(KMat_Dense) JCB_N = (-JCB_B * invMass.reshaped().asDiagonal()) * KMat_Dense;
                    TDof NR1 = -JCB_N * R1;
                    decltype(KMat_Dense) Mat1 =
                        JCB_A * JCB_C * decltype(KMat_Dense)::Identity(KMat_Dense.rows(), KMat_Dense.cols()) +
                        JCB_B * JCB_N;
                    dv = Mat1.partialPivLu().solve(JCB_A * R2 - NR1);
                    du = JCB_N.partialPivLu().solve(NR1 + JCB_B * JCB_N * dv) * (-1. / JCB_A);
                    // std::cout << dv << "=============" << std::endl;
                    // std::cout << du << std::endl;
                    // std::abort();
                }
                else
                {
                    decltype(KMat) JCB_N = (-JCB_B * invMass.reshaped().asDiagonal()) * KMat;
                    // std::cout << JCB_N << std::endl;
                    // std::abort();
                    TDof NR1 = -JCB_N * R1;
                    decltype(KMat) Mat1 =
                        JCB_A * JCB_C * IMat +
                        JCB_B * JCB_N;
                    LUSolver.factorize(Mat1);
                    dv = LUSolver.solve(JCB_A * R2 - NR1);
                    LUSolver.factorize(JCB_N);
                    du = LUSolver.solve(NR1 + JCB_B * JCB_N * dv) * (-1. / JCB_A);
                }

                num dum = du.array().abs().maxCoeff() + 1e-100;
                num dvm = dv.array().abs().maxCoeff() + 1e-100;
                num damping = std::min({1., maxDU / dum, maxDV / dvm}) * 0.9;

                u -= du.reshaped(3, u.cols()) * damping;
                v -= dv.reshaped(3, v.cols()) * damping;
                u.array() *= flags.array();
                v.array() *= flags.array();

                // std::cout << dum << " " << damping << std::endl;
                // std::abort();

                if (iter == 1)
                    duBase = dum;
                else
                {
                    if (dum <= duBase * threshold || iter == nStep)
                    {
                        ret.nStep = iter;
                        ret.thRel = dum / duBase;
                        return ret;
                    }
                }

                this->ConstructRHSJacobi(f, u, v, true);
            }
            return ret;
        }
    };
}