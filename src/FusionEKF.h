#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class FusionEKF {
public:
  /**
   * Constructor
   */
  FusionEKF();

  /**
   * Destructor
   */
  virtual ~FusionEKF();

  /**
   * Update Kalman filter with the new measurement
   * @param measurement_pack Measurement, can be from either radar or lidar.
   */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
   * Kalman Filter object, implement prediction and update formulas
   */
  KalmanFilter ekf_;

private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  // tool object used to compute RMSE
  Tools tools;

  // covariance matrices of the measurenments of lidar and radar
  MatrixXd R_laser_;
  MatrixXd R_radar_;

  // projection matrix of lidar
  MatrixXd H_laser_;

  // Jacobian matrix of radar
  MatrixXd H_radar_;

  /**
    * Initialize Kalman filter with the first measurement, create the covariance matrix
	* @param measurement_pack First measurement
	*/
  void InitKF(const MeasurementPackage &measurement_pack);

  /**
    * Project state vector to the space of radar measurements
	* @param x State vector
	*/
  VectorXd ProjectRadarSpace(const VectorXd &x);

  /**
    * Compute Jacobian (H_radar_ matrix) of the linear approximation of ProjectRadarSpace function
	* @param x State vector
	*/
  bool ComputeRadarJacobian(const VectorXd &x);

  /**
    * Normalize angle to be in the range [-pi,pi]
	* @param phi Measurement angle
	*/
  double AdjustAngle(double phi);

  /**
    * Normalize angles rho and phi so that |phi-phi_hat| <= pi. 
	* @param rho Measurement angle
	* @param phi Predicted angle
	*/
  void AlignAngles(double &phi, double &phi_hat);
};

#endif /* FusionEKF_H_ */
