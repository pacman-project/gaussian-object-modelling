#include <iostream>

#include <gp_regression/gp_regressors.h>
#include <gp_regression/gp_projector.hpp>

using namespace gp_regression;
using namespace std;

int main( int argc, char** argv )
{
        /*****  Generate INPUT data  ******************************************/
        std::cout << "Generate INPUT data for a sphere..." << std::endl;
        Data cloud;
        int ni = 10;
        int nj = 9;
        for(int i = 0; i < ni; ++i)
        {
                for(int j = 0; j < nj; ++j)
                {
                        // on
                        cloud.coord_x.push_back(10.0*cos(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_y.push_back(10.0*sin(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_z.push_back(10.0*sin(2*3.1416*j/nj));
                        cloud.label.push_back(0.0);
                        // outside
                        cloud.coord_x.push_back(20.0*cos(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_y.push_back(20.0*sin(2*3.1416*i/ni)*cos(2*3.1416*j/nj));
                        cloud.coord_z.push_back(20.0*sin(2*3.1416*j/nj));
                        cloud.label.push_back(1.0);
                }
        }

        // inside
        cloud.coord_x.push_back(1.0*cos(2*3.1416*0/ni)*cos(2*3.1416*0/nj));
        cloud.coord_y.push_back(1.0*sin(2*3.1416*0/ni)*cos(2*3.1416*0/nj));
        cloud.coord_z.push_back(1.0*sin(2*3.1416*0/nj));
        cloud.label.push_back(-1.0);

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
        std::vector<double> f,v;

        regresor.evaluate(sphere, query, f, v);

        cout << "Query value x: " << endl;
        cout << query.coord_x.at(0) << " " << query.coord_y.at(0) << " " 
                        << query.coord_z.at(0) << endl << endl;
        cout << "Query function f(x): " << endl;
        cout << f.at(0) << endl << endl;
        cout << "Query confidence V(f(x)): " << endl;
        cout << v.at(0) << endl << endl;

        /*****  Query the model with a point  *********************************/
        std::cout << "Query the model with a point on the surface" << std::endl;
        f.clear();
        v.clear();
        query.coord_x = vector<double>(1, 10.0*cos(2*3.1416*23/77)*cos(2*3.1416*27/77));
        query.coord_y = vector<double>(1, 10.0*sin(2*3.1416*23/77)*cos(2*3.1416*27/77));
        query.coord_z = vector<double>(1, 10.0*sin(2*3.1416*27/77));
        Eigen::MatrixXd N;
        Eigen::MatrixXd Tx;
        Eigen::MatrixXd Ty;

        regresor.evaluate(sphere, query, f, v, N, Tx, Ty);

        cout << "Query value x: " << endl;
        cout << query.coord_x.at(0) << " " << query.coord_y.at(0) << " "
                        << query.coord_z.at(0) << endl << endl;
        cout << "Query function f(x): " << endl;
        cout << f.at(0) << endl << endl;
        cout << "Query confidence V(f(x)): " << endl;
        cout << v.at(0) << endl << endl;

        cout << "Query normal N(f(x)): " << endl;
        cout << N.row(0) << endl << endl;
        cout << "Query tangent X Tx(f(x)): " << endl;
        cout << Tx.row(0) << endl << endl;
        cout << "Query tangent Ty(f(x)): " << endl;
        cout << Ty.row(0) << endl << endl;

        /*****  Query the model with a INPUT point  ***************************/
        std::cout << "Query the model with the same INPUT" << std::endl;
        cloud.label.clear();
        f.clear();
        v.clear();

        regresor.evaluate(sphere, cloud, f, v);

        cout << "Query function f(x): " << endl;
        for(int i = 0; i < f.size(); ++i)
        {
                cout << f.at(i) << " / ";
        }
        cout << endl << endl;

        cout << "Query confidence V(f(x)): " << endl;
        for(int i = 0; i < f.size(); ++i)
        {
                cout << v.at(i) << " /  ";
        }
        cout << endl << endl;

        /*****  Test the Tangent Basis generator  *****************************/
//        std::cout << "Test the Tangent Basis generator" << std::endl;
//        Eigen::Vector3d z(1.0, 1.0, 1.0);
//        std::cout << "z: " << std::endl << z << std::endl << std::endl;

//        Eigen::Vector3d x, y;

//        regresor.computeTangentBasis(z, x, y);
//        std::cout << "x: " << std::endl << x << std::endl << std::endl;
//        std::cout << "y: " << std::endl << y << std::endl << std::endl;
//        std::cout << "x*z: " << std::endl << x.dot(z) << std::endl << std::endl;
//        std::cout << "y*z: " << std::endl << y.dot(z) << std::endl << std::endl;
//        std::cout << "x*y: " << std::endl << x.dot(y) << std::endl << std::endl;

        /*****  Test the GPProjector  *****************************/
        cloud.label.clear();
        f.clear();
        v.clear();
        Eigen::Vector3d center;
        center(0) = query.coord_x.at(0);
        center(1) = query.coord_y.at(0);
        center(2) = query.coord_z.at(0);

        GPProjector<Gaussian> projector;

        Chart chart;
        projector.generateChart(sphere, center, 1.0, chart);

        cout << "Chart info: " << endl << endl;
        cout << "Center: " << chart.C << endl << endl;
        cout << "N: " << chart.N << endl << endl;
        cout << "Tx: " << chart.Tx << endl << endl;
        cout << "Ty: " << chart.Ty << endl << endl;
        cout << "Size: " << chart.R << endl << endl;

        // TEST Projection
        // point in the model
        Eigen::Vector3d init_center(10.0*cos(2*3.1416*3/ni)*cos(2*3.1416*2/nj),
                               10.0*sin(2*3.1416*3/ni)*cos(2*3.1416*2/nj),
                               10.0*sin(2*3.1416*2/nj));

        // create the chart there
        Chart init_chart;
        projector.generateChart(sphere, center, 1.0, init_chart);

        // dummy sample in the chart
        Eigen::Vector3d point_in_tangent;
        point_in_tangent = init_chart.C + 10.0*init_chart.Tx + 10.0*init_chart.Ty;

        // project it
        Eigen::Vector3d projected_point;
        projector.project(sphere, init_chart, point_in_tangent, projected_point);

        // check value
        Data result;

        result.coord_x.push_back(point_in_tangent(0));
        result.coord_y.push_back(point_in_tangent(1));
        result.coord_z.push_back(point_in_tangent(2));

        result.coord_x.push_back(projected_point(0));
        result.coord_y.push_back(projected_point(1));
        result.coord_z.push_back(projected_point(2));

        std::vector<double> f_projection, v_projection;

        regresor.evaluate(sphere, result, f_projection, v_projection);

        cout << "Tangent point x: " << endl;
        cout << result.coord_x.at(0) << " " << result.coord_y.at(0) << " "
                        << result.coord_z.at(0) << endl << endl;
        cout << "Function at tangent point f(x): " << endl;
        cout << f_projection.at(0) << endl << endl;

        cout << "Projected point x: " << endl;
        cout << result.coord_x.at(1) << " " << result.coord_y.at(1) << " "
                        << result.coord_z.at(1) << endl << endl;
        cout << "Function at projected point f(x): " << endl;
        cout << f_projection.at(1) << endl << endl;

        return 0;
}
