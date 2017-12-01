#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

#define PI 3.1415926

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  
  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
	          0, 1, 0, 0;

  H_radar_ = MatrixXd(3, 4);
}

/**
 * Destructor
 */
FusionEKF::~FusionEKF() {}

/**
 * Normalize angle to be in the range [-pi,pi]
 * @param phi Measurement angle
 */
double FusionEKF::AdjustAngle(double phi) {
	if (phi > PI) {
		phi -= 2 * PI;
	}
	else if (phi < -PI) {
		phi += 2 * PI;
	}
	return phi;
}

/**
 * Normalize angles rho and phi so that |phi-phi_hat| <= pi
 * @param rho Measurement angle
 * @param phi Predicted angle
 */
void FusionEKF::AlignAngles(double &phi, double &phi_hat) {
	if (abs(phi - phi_hat) > PI)
		// |phi| > PI/2 or |phi_hat| > PI/2
		if (phi > PI / 2)
			phi_hat += 2 * PI;
		else if (phi < -PI / 2)
			phi_hat -= 2 * PI;
		else if (phi_hat > PI / 2)
			phi_hat += 2 * PI;
		else
			phi_hat -= 2 * PI;
}

/**
 * Initialize Kalman filter with the first measurement, create the covariance matrix
 * @param measurement_pack First measurement
 */
void FusionEKF::InitKF(const MeasurementPackage &measurement_pack) {
	
	previous_timestamp_ = measurement_pack.timestamp_;
	VectorXd x = VectorXd(4);
	MatrixXd P = MatrixXd(4, 4);

	cout << "Init" << endl;

	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

		// initialize with radar measurement		
		cout << "Radar" << endl;
		cout << measurement_pack.raw_measurements_ << endl;

		// normalize angle to be in [-pi,pi] range
		double phi = AdjustAngle(measurement_pack.raw_measurements_(1));

		// convert radar measurement from polar to cartesian coordinates 
		x << measurement_pack.raw_measurements_(0) * cos(phi),
			 -measurement_pack.raw_measurements_(0) * sin(phi),
			 0, 0;
	
		// initialize covariance matrix
		P << R_radar_(0, 0), 0, 0, 0,
			 0, R_radar_(1, 1), 0, 0,
			 0, 0, 100, 0,
			 0, 0, 0, 100;
	}
	else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
		
		// initialize with lidar measurement
		cout << "Laser" << endl;
		cout << measurement_pack.raw_measurements_ << endl;

		// initialize state and covariance matrix
		x << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), 0, 0;
		P << R_laser_(0, 0), 0, 0, 0,
			 0, R_laser_(1, 1), 0, 0,
			 0, 0, 100, 0,
			 0, 0, 0, 100;
	}

	ekf_.SetState(x, P);
	cout << "Init State" << endl;
	cout << ekf_.GetState() << endl;

    // done initializing, no need to predict or update
	is_initialized_ = true;
}

/**
 * Project state vector to the space of radar measurements
 * @param x State vector
 */
VectorXd FusionEKF::ProjectRadarSpace(const VectorXd &x) {

	VectorXd projected_state = VectorXd(3);

	// get current state
	VectorXd state = ekf_.GetState();
	double px = state(0);
	double py = state(1);
	double vx = state(2);
	double vy = state(3);

	// compute projection
	projected_state(0) = sqrt(px*px+py*py);
	projected_state(1) = atan2(py, px);
	projected_state(2) = (px * vx + py * vy) / projected_state(0);
	
	return projected_state;
}

/**
 * Compute Jacobian (H_radar_ matrix) of the linear approximation of ProjectRadarSpace function
 * @param x State vector
 */
bool FusionEKF::ComputeRadarJacobian(const VectorXd &x) {
	
	// recover state parameters
	double px = x(0);
	double py = x(1);
	double vx = x(2);
	double vy = x(3);

	// pre-compute a set of terms to avoid repeated calculation
	double c1 = px*px + py*py;
	double c2 = sqrt(c1);
	double c3 = (c1*c2);

	// check division by zero
	if (fabs(c1) < 0.0001 || fabs(c2) < 0.0001 || fabs(c3) < 0.0001) {
		cout << "CalculateJacobian () - Error - Division by Zero. Cannot update the state." << endl;
		return false;
	}

	// compute the Jacobian matrix
	H_radar_ << (px / c2), (py / c2), 0, 0,
		        -(py / c1), (px / c1), 0, 0,
		        py*(vx*py - vy*px) / c3, px*(px*vy - py*vx) / c3, px / c2, py / c2;
		
	return true;
}

/**
 * Update Kalman filter with the new measurement
 * @param measurement_pack Measurement, can be from either radar or lidar.
 */
void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
	
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
	  InitKF(measurement_pack);
	  return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  
  cout << "Delta t: " << measurement_pack.timestamp_ - previous_timestamp_ << endl;
  ekf_.Predict(measurement_pack.timestamp_-previous_timestamp_);
  
  cout << "Predicted state" << endl;
  cout << ekf_.GetState() << endl;

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   * Use the sensor type to perform the update step.
   */

  bool updated = true;
  VectorXd x = ekf_.GetState();

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // radar update
	  cout << "Radar measurement" << endl;
	  cout << measurement_pack.raw_measurements_ << endl;

	  // adjust measurement angle to be in [-pi,pi] range
	  VectorXd adjusted_measurement = measurement_pack.raw_measurements_;
	  adjusted_measurement(1) = AdjustAngle(adjusted_measurement(1));

	  // project state vector to the measurement space
	  VectorXd projected_state = ProjectRadarSpace(x);
	  cout << "Projected State" << endl;
	  cout << projected_state << endl;

	  // normalize angles so that |adjusted_measurement(1) - projected_state(1)| <= pi
	  // this normalization is needed when the smallest angle between phi and phi_hat contains the negative side of x axis
	  AlignAngles(adjusted_measurement(1), projected_state(1));

	  // update H_radar_ matrix
	  if (ComputeRadarJacobian(x))
		  // if Jacobian was computer successfully then update state and its covariance matrix
		  ekf_.Update(adjusted_measurement, R_radar_, H_radar_, projected_state);
	  else
		  updated = false;
  } else {
	  // lidar update
	  cout << "Laser measurement" << endl;
	  cout << measurement_pack.raw_measurements_ << endl;

	  ekf_.Update(measurement_pack.raw_measurements_, R_laser_, H_laser_, H_laser_ * x);
  }

  // record last update time
  if (updated)
      previous_timestamp_ = measurement_pack.timestamp_;

  // print the output
  cout << "x_ = " << ekf_.GetState() << endl;
  cout << "P_ = " << ekf_.GetCovariance() << endl;
}
