#include <iostream>

#include <gp_regression/gp_regressor.h>

using namespace gp_regression;
using namespace std;

int main( int argc, char** argv )
{
        /*****  Generate INPUT data  ******************************************/
        std::cout << "Generate INPUT data for a sphere..." << std::endl;
        Data cloud;
        int ni = 6;
        int nj = 6;
        for(int i = 0; i < ni; ++i)
        {
                for(int j = 0; j < nj; ++j)
                {
                        // on
                        cloud.coord_x.push_back(1.0*cos(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_y.push_back(1.2*sin(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_z.push_back(1.4*sin(2*3.1416*j/nj));
                        cloud.label.push_back(0.0);
                        // inside
                        cloud.coord_x.push_back(0.5*cos(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_y.push_back(0.5*sin(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_z.push_back(0.5*sin(2*3.1416*j/nj));
                        cloud.label.push_back(-1.0);
                        // outside
                        cloud.coord_x.push_back(2.0*cos(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_y.push_back(2.0*sin(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_z.push_back(2.0*sin(2*3.1416*j/nj));
                        cloud.label.push_back(1.0);
                }
        }

        /*****  Create the model  *********************************************/
        std::cout << "Create the model..." << std::endl;
        Model sphere;
        GaussianRegressor regresor;
        
        regresor.create(cloud, sphere);

        cout << "Model points: " << endl;
        cout << sphere.P << endl << endl;
        cout << "Model labels (pre): " << endl;
        cout << sphere.Y << endl << endl;
        // cout << "Model Kpp: " << endl;
        // cout << sphere.Kpp << endl << endl;
        // cout << "Model Kppdiff: " << endl;
        // cout << sphere.Kppdiff << endl << endl;
        cout << "Model Normal: " << endl;
        cout << sphere.N << endl << endl;
        cout << "Model Tx: " << endl;
        cout << sphere.Tx << endl << endl;
        cout << "Model Ty: " << endl;
        cout << sphere.Ty << endl << endl;


        /*****  Query the model with a point  *********************************/
        std::cout << "Query the model with a point" << std::endl;
        Data query;
        query.coord_x = vector<double>(1, 0.68);
        query.coord_y = vector<double>(1, 0.5);
        query.coord_z = vector<double>(1, 0.5);

        regresor.estimate(sphere, query);

        cout << "Query value: " << endl;
        cout << query.coord_x.at(0) << " " << query.coord_y.at(0) << " " 
                        << query.coord_z.at(0) << endl << endl;
        cout << "Query label (post): " << endl;
        cout << query.label.at(0) << endl << endl;
        // cout << "Query label covariance: " << endl;
        // cout << query.ev_x.at(0) << " " << query.ev_y.at(0) 
                        // << " " << query.ev_z.at(0) 
                        // << endl << endl;

        /*****  Query the model with a INPUT point  ***************************/
        std::cout << "Query the model with a INPUT point" << std::endl;
        query.label.clear();
        query.coord_x = vector<double>(1, 0.1);
        query.coord_y = vector<double>(1, 0.3);
        query.coord_z = vector<double>(1, 0.5);

        regresor.estimate(sphere, query);

        cout << "Query same input: " << endl;
        cout << query.coord_x.at(0) << " " << query.coord_y.at(0) << " " 
                        << query.coord_z.at(0) << endl << endl;
        cout << "Query label (post): " << endl;
        cout << query.label.at(0) << endl << endl;
        // cout << "Query label covariance: " << endl;
        // cout << query.ev_x.at(0) << " " << query.ev_y.at(0) 
                        // << " " << query.ev_z.at(0) 
                        // << endl << endl;

        /*****  Test the Tangent Basis generator  *****************************/
        std::cout << "Test the Tangent Basis generator" << std::endl;
        Eigen::Vector3d z(1.0, 1.0, 1.0);
        std::cout << "z: " << std::endl << z << std::endl;

        Eigen::Vector3d x, y;

        computeTangentBasis(z, x, y);
        std::cout << "x: " << std::endl << x << std::endl;
        std::cout << "y: " << std::endl << y << std::endl;
        std::cout << "x*z: " << std::endl << x.dot(z) << std::endl;
        std::cout << "y*z: " << std::endl << y.dot(z) << std::endl;
        std::cout << "x*y: " << std::endl << x.dot(y) << std::endl;

        return 0;
}
