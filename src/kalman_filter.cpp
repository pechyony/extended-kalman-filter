#include <iostream>
#include "kalman_filter.h"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {
	
	x_ = VectorXd(4);
	x_ << 0, 0, 0, 0;

	P_ = MatrixXd(4, 4);
	P_ << 0, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0;

	Q_ = MatrixXd(4, 4);
	Q_ << 0, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0;

	F_ = MatrixXd(4, 4);
	F_ << 1, 0, 1, 0,
    	  0, 1, 0, 1,
		  0, 0, 1, 0,
		  0, 0, 0, 1;

	I_ = MatrixXd::Identity(4, 4);
}

KalmanFilter::~KalmanFilter() {}

/**
* Set state vector and convariance matrix of Kalman filter
*/
void KalmanFilter::SetState(const VectorXd& x, const MatrixXd& P) {
	x_ = x;
	P_ = P;
}

/**
* Return current state
*/
VectorXd KalmanFilter::GetState() {
	return x_;
}

/**
* Return covariance matrix of the current state
*/
MatrixXd KalmanFilter::GetCovariance() {
	return P_;
}

/**
* Predicts the state and the state covariance
* using the process model
* @param delta_T Time between k and k+1 in microseconds
*/
void KalmanFilter::Predict(long long delta_T_) {
	// convert from microseconds to seconds
	float delta_T = delta_T_ / 1000000.0;

	// Update matrix F
	F_(0, 2) = delta_T;
	F_(1, 3) = delta_T;
 
	// Update matrix Q
	Q_(0, 0) = pow(delta_T, 4) * noise_ax / 4;
	Q_(0, 2) = pow(delta_T, 3) * noise_ax / 2;
	Q_(1, 1) = pow(delta_T, 4) * noise_ay / 4;
	Q_(1, 3) = pow(delta_T, 3) * noise_ay / 2;
	Q_(2, 0) = pow(delta_T, 3) * noise_ax / 2;
	Q_(2, 2) = pow(delta_T, 2) * noise_ax;
	Q_(3, 1) = pow(delta_T, 3) * noise_ay / 2;
	Q_(3, 3) = pow(delta_T, 2) * noise_ay;

	// prediction of state and its covariance matrix
	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

/**
* Updates the state by using standard/extended Kalman Filter equations
* @param z The measurement at k+1
* @param R Covariance matrix of z
* @param H Projection matrix from state to measurement space (standard Kalman filter) / Jacobian matrix (extended Kalman filter)
* @param projected_state Current state, projected to measurement space
*/
void KalmanFilter::Update(const VectorXd &z, const MatrixXd &R, const MatrixXd &H, const VectorXd &projected_state) {
    /**
     * update the state by using Kalman Filter equations
    */
	VectorXd y = z - projected_state;
	MatrixXd Ht = H.transpose();
	MatrixXd S = H * P_ * Ht + R;
	MatrixXd Si = S.inverse();
	MatrixXd K = P_ * Ht * Si;
	
	x_ += K * y;
	P_ = (I_ - K * H) * P_;
}


