#include <iostream>
#include <time.h>

#include "gp/SampleSet.h"
#include "gp/GaussianProcess.h"

using namespace gp;
using namespace std;

/* Random generators */
template <typename Type> const Type random() {
	return static_cast <Type> (rand()) / static_cast <Type> (RAND_MAX);
};
int main( int argc, char** argv )
{
//	test random number generator
//	for (size_t i = 0; i < 100; ++i)
//		printf("Random[%lu] = %f\n", i, random<double>());
//	return 0;
        /*****  Global variables  ******************************************/
        size_t N_sur = 30, N_ext = 30, N_int = 1;
 	const Real surRho = 0.05, extRho = 0.4;
	const Real surRhoSqr = surRho*surRho, extRhoSqr = extRho*extRho;
       double noise = 0.001;
        const double PI = numeric_const<double>::PI;

        /* initialize random seed: */
  	srand (time(NULL));

  	const bool prtInit = true;
  	const bool prtPreds = true;

        /*****  Generate Input data  ******************************************/
        Vec3Seq cloud;
        Vec targets;
        printf("Generate input data %lu\n", N_sur + N_ext + N_int);
        if (prtInit)
        	printf("Cloud points %lu:\n", N_sur);
        for (size_t i = 0; i < N_sur; ++i) {
        	const Real theta = 2 * PI*random<double>() - PI;
		const Real colatitude = (PI / 2) - (2 * PI*random<double>() - PI);
		const Real z = (2 * random<double>()*surRho - surRho) * cos(colatitude);
		Vec3 point(sin(theta)*sqrt(surRhoSqr - z*z), cos(theta)*sqrt(surRhoSqr - z*z), z);

        	point += Vec3(2*noise*random<double>() - noise, 2*noise*random<double>() - noise, 2*noise*random<double>() - noise);
        	cloud.push_back(point);

        	double y = point.magnitudeSqr() - surRhoSqr;
        	if (prtInit) cout << "points[" << i << "] th(" << theta <<") = <" << point.x << " " << point.y << " " << point.z << ">" << " targets[" << i << "] = " << y << endl;
        	targets.push_back(y);
       	}

       	if (prtInit)
       		printf("\nExternal points %lu:\n", N_ext);
       	for (size_t i = 0; i < N_ext; ++i) {
        	const Real theta = 2 * PI*random<double>() - PI;
		const Real colatitude = (PI / 2) - (2 * PI*random<double>() - PI);
		const Real z = (2 * random<double>()*extRho - extRho) * cos(colatitude);
		Vec3 point(sin(theta)*sqrt(extRhoSqr - z*z), cos(theta)*sqrt(extRhoSqr - z*z), z);
        	point += Vec3(2*noise*random<double>() - noise, 2*noise*random<double>() - noise, 2*noise*random<double>() - noise);
        	cloud.push_back(point);

        	double y = point.magnitudeSqr() - surRhoSqr;
        	if (prtInit)
        		if (prtInit) cout << "points[" << i << "] th(" << theta <<") = <" << point.x << " " << point.y << " " << point.z << ">" << " targets[" << i << "] = " << y << endl;
        	targets.push_back(y);
       	}

       	if (prtInit)
       		printf("\nInternal point(s) %lu:\n", N_int);
       	Vec3 zero; zero.setZero();
       	cloud.push_back(zero);
       	targets.push_back(zero.magnitudeSqr() - surRhoSqr);
       	if (prtInit) cout << "points[" << 1 << "] th(" << 0 <<") = <" << zero.x << " " << zero.y << " " << zero.z << ">" << " targets[" << 1 << "] = " << zero.magnitudeSqr() - surRhoSqr << endl << endl;

       	/*****  Create the model  *********************************************/
       	SampleSet::Ptr trainingData(new SampleSet(cloud, targets));
//#define gaussianReg
#define cov_thin_plate
#ifdef laplaceReg
	LaplaceRegressor::Desc laplaceDesc;
	laplaceDesc.covTypeDesc.inputDim = trainingData->rows();
	laplaceDesc.noise = noise;
	LaplaceRegressor::Ptr gp = laplaceDesc.create();
	printf("Laplace Regressor created %s\n", gp->getName().c_str());
#endif
#ifdef gaussianReg
	GaussianRegressor::Desc guassianDesc;
	guassianDesc.covTypeDesc.inputDim = trainingData->rows();
	guassianDesc.noise = noise;
	guassianDesc.covTypeDesc.length = 0.03;
	guassianDesc.covTypeDesc.sigma = 0.015;
	guassianDesc.optimise = false;	
	GaussianRegressor::Ptr gp = guassianDesc.create();
	printf("Gaussian Regressor created %s\n", gp->getName().c_str());
#endif
#ifdef cov_thin_plate
	ThinPlateRegressor::Desc covDesc;
	covDesc.initialLSize = trainingData->rows();
	covDesc.covTypeDesc.inputDim = trainingData->cols();
	covDesc.covTypeDesc.noise = 0.0001;
	double l = 0.0;
	for (size_t i = 0; i < cloud.size(); ++i) {
		for (size_t j = i; j < cloud.size(); ++j) {
			const double d = cloud[i].distance(cloud[j]);
			if (l < d) l = d;
		}
	}
	covDesc.covTypeDesc.length = l;
	covDesc.optimise = false;
	printf("ThinPlate cov: inputDim=%lu, parmDim=%lu, R=%f noise=%f\n", covDesc.covTypeDesc.inputDim, covDesc.covTypeDesc.paramDim,
		covDesc.covTypeDesc.length, covDesc.covTypeDesc.noise);
	ThinPlateRegressor::Ptr gp = covDesc.create();
	printf("%s Regressor created\n", gp->getName().c_str());
#endif
   	
        gp->set(trainingData);
        //ThinPlateRegressor gp(&trainingData);


        /*****  Query the model with a point  *********************************/
        Vec3 q(cloud[0]);
        const double qf = gp->f(q);
        const double qVar = gp->var(q);

        if (prtPreds) cout << "y = " << targets[0] << " -> qf = " << qf << " qVar = " << qVar << endl << endl;

        /*****  Query the model with new points  *********************************/
	const size_t testSize = 10;
	const double range = 1.0;
	Vec3Seq x_star; x_star.resize(testSize);
	Vec y_star; y_star.resize(testSize);
	for (size_t i = 0; i < testSize; ++i) {
		const double theta = 2*PI*random<double>() - PI;
        	const double z = 2*random<double>() - 1;
        	Vec3 point(sin(theta)*sqrt(1-z*z), cos(theta)*sqrt(1-z*z), z);
        	point += Vec3(2*noise*random<double>() - noise, 2*noise*random<double>() - noise, 2*noise*random<double>() - noise);

        	// save the point for later
        	x_star[i] = point;
        	// compute the y value
        	double y = point.magnitudeSqr() - 1;
        	// save the y value for later
        	y_star[i] = y;

        	// gp estimate of y value
        	const double f_star = gp->f(point);
		const double v_star = gp->var(point);
        	if (prtPreds)
        		cout << "f(x_star) = " << y << " -> f_star = " << f_star << " v_star = " << v_star << endl;
	}
	cout << endl;

	/*****  Evaluate points and normals  *********************************************/
	printf("Evaluate %lu points and normals\n", testSize);
	std::vector<Real> fx, varx;
	Eigen::MatrixXd normals, tx, ty;
	Vec3Seq nn; nn.resize(testSize);
//		points.clear(); points.resize(testSize);
	Real evalError = 0.0;
	gp->evaluate(x_star, fx, varx, normals, tx, ty);
	for (size_t i = 0; i < testSize; ++i) {
		//points[i] = x_star[i];
		nn[i] = Vec3(normals(i, 0), normals(i, 1), normals(i, 2));
		evalError += std::pow(fx[i] - y_star[i], 2);
		if (prtPreds)
			printf("Evaluate[%lu]: f(x_star) = %f -> f_star = %f v_star = %f normal=[%f %f %f]\n", i, y_star[i], fx[i], varx[i], normals(i,0), normals(i,1), normals(i,2));
	}
	if (prtPreds)
		printf("Evaluate(): error=%f avg=%f\n\n", evalError, evalError / testSize);


	/*****  Add point to the model  *********************************************/
	gp->add_patterns(x_star, y_star);

	for (size_t i = 0; i < testSize; ++i) {
		const double q2f = gp->f(x_star[i]);
		const double q2Var = gp->var(x_star[i]);

		if (prtPreds)
			cout << "y = " << y_star[i] << " -> q2f = " << q2f << " q2Var = " << q2Var << endl;

	}




//        /*****  Create the model  *********************************************/
//        Model mug;
//        GaussianRegressor regresor;
//
//        regresor.create(cloud, mug);

//        cout << "Model points: " << endl;
//        cout << mug.P << endl << endl;
//        cout << "Model labels (pre): " << endl;
//        cout << mug.Y << endl << endl;
//        cout << "Model Kpp: " << endl;
//        cout << mug.Kpp << endl << endl;

//        /*****  Query the model with a point  *********************************/
//        Data query;
//        query.coord_x = vector<double>(1, 0.68);
//        query.coord_y = vector<double>(1, 0.5);
//        query.coord_z = vector<double>(1, 0.5);

//        regresor.estimate(mug, query);

//        cout << "Query value: " << endl;
//        cout << query.coord_x.at(0) << " " << query.coord_y.at(0) << " "
//                        << query.coord_z.at(0) << endl << endl;
//        cout << "Query label (post): " << endl;
//        cout << query.label.at(0) << endl << endl;
//        // cout << "Query label covariance: " << endl;
//        // cout << query.ev_x.at(0) << " " << query.ev_y.at(0)
//                        // << " " << query.ev_z.at(0)
//                        // << endl << endl;

//        /*****  Query the model with a INPUT point  ***************************/
//        query.label.clear();
//        query.coord_x = vector<double>(1, 0.1);
//        query.coord_y = vector<double>(1, 0.3);
//        query.coord_z = vector<double>(1, 0.5);

//        regresor.estimate(mug, query);

//        cout << "Query same input: " << endl;
//        cout << query.coord_x.at(0) << " " << query.coord_y.at(0) << " "
//                        << query.coord_z.at(0) << endl << endl;
//        cout << "Query label (post): " << endl;
//        cout << query.label.at(0) << endl << endl;
//        // cout << "Query label covariance: " << endl;
//        // cout << query.ev_x.at(0) << " " << query.ev_y.at(0)
//                        // << " " << query.ev_z.at(0)
//                        // << endl << endl;

        return 0;
}
