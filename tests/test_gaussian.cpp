#include <iostream>

#include <gp_regression/gp_regressor.h>

using namespace gp_regression;
using namespace std;

int main( int argc, char** argv )
{
        /*****  Generate Input data  ******************************************/
        Data cloud;
        cloud.coord_x = vector<double>(1, 0.1);
        cloud.coord_y = vector<double>(1, 0.3);
        cloud.coord_z = vector<double>(1, 0.5);
        cloud.label = vector<double>(1, 0.0);
        cloud.coord_x.push_back(0.3);
        cloud.coord_y.push_back(-0.1);
        cloud.coord_z.push_back(-0.4);
        cloud.label.push_back(-1.0);
        cloud.coord_x.push_back(0.25);
        cloud.coord_y.push_back(0.02);
        cloud.coord_z.push_back(-0.8);
        cloud.label.push_back(1.0);
        cloud.coord_x.push_back(-0.1);
        cloud.coord_y.push_back(0.01);
        cloud.coord_z.push_back(0.3);
        cloud.label.push_back(1.0);

        /*****  Create the model  *********************************************/
        Model mug;
        GaussianRegressor regresor;
        
        regresor.create(cloud, mug);

        cout << "Model points: " << endl;
        cout << mug.P << endl << endl;
        cout << "Model labels (pre): " << endl;
        cout << mug.Y << endl << endl;
        cout << "Model Kpp: " << endl;
        cout << mug.Kpp << endl << endl;

        /*****  Query the model with a point  *********************************/
        Data query;
        query.coord_x = vector<double>(1, 0.68);
        query.coord_y = vector<double>(1, 0.5);
        query.coord_z = vector<double>(1, 0.5);

        regresor.estimate(mug, query);

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
        query.label.clear();
        query.coord_x = vector<double>(1, 0.1);
        query.coord_y = vector<double>(1, 0.3);
        query.coord_z = vector<double>(1, 0.5);

        regresor.estimate(mug, query);

        cout << "Query same input: " << endl;
        cout << query.coord_x.at(0) << " " << query.coord_y.at(0) << " " 
                        << query.coord_z.at(0) << endl << endl;
        cout << "Query label (post): " << endl;
        cout << query.label.at(0) << endl << endl;
        // cout << "Query label covariance: " << endl;
        // cout << query.ev_x.at(0) << " " << query.ev_y.at(0) 
                        // << " " << query.ev_z.at(0) 
                        // << endl << endl;

        return 0;
}
