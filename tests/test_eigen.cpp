#include <iostream>
#include <Eigen/Dense>
#include <random_generation.hpp>

int main( int argc, char** argv )
{
    Eigen::MatrixXd A;
    const size_t N (5);
    A.resize(N,N);
    for (size_t i=0; i<N; ++i)
        for(size_t j=0; j<=i; ++j)
            A(i,j) = getRandIn(0.0, 1.0, false);
    Eigen::MatrixXd I;
    I.resize(N,N);
    I.setIdentity();
    A += N * I;
    std::cout<<"A :\n"<<A<<std::endl<<std::endl;
    Eigen::MatrixXd savA = A.selfadjointView<Eigen::Lower>();
    std::cout<<"selfadjoint View of A :\n"<<savA<<std::endl<<std::endl;
    Eigen::MatrixXd LofA = savA.llt().matrixL();
    std::cout<<"LLT(A) matrix L :\n"<<LofA<<std::endl<<std::endl;
    std::cout<<"L * L^T :\n"<<LofA * LofA.transpose()<<std::endl;
    return 0;
}
