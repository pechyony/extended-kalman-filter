#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
    */
	
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
		std::cout << "Wrong sizes" << std::endl;
		return rmse;
	}

	vector<VectorXd> residuals_square;
	VectorXd residual(4);

	//accumulate squared residuals
	int n = estimations.size();
	for (int i = 0; i < n; ++i) {
		residual = estimations[i] - ground_truth[i];
		residuals_square.push_back(residual.array() * residual.array());
	}

	//calculate the mean
	for (int i = 0; i < n; i++) {
		rmse += residuals_square[i];
	}
	rmse /= n;

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}


