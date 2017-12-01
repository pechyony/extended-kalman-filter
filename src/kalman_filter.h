#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// variance of acceleration in x and y axes (we assume acceleration has zero mean)
const float noise_ax = 9;
const float noise_ay = 9;

class KalmanFilter {
public:
  /**
   * Constructor
   */
  KalmanFilter();

  /**
   * Destructor
   */
  virtual ~KalmanFilter();

  /**
   * Set state vector and convariance matrix of Kalman filter
   */
  void SetState(const VectorXd &x, const MatrixXd &P);

  /**
   * Return current state
   */
  VectorXd GetState();

  /**
  * Return covariance matrix of the current state
  */
  MatrixXd GetCovariance();

  /**
   * Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in microseconds
   */
  void Predict(long long delta_T);

  /**
   * Updates the state by using standard/extended Kalman Filter equations
   * @param z The measurement at k+1
   * @param R Covariance matrix of z
   * @param H Projection matrix from state to measurement space (standard Kalman filter) / Jacobian matrix (extended Kalman filter)
   * @param projected_state Current state, projected to measurement space
   */
  void Update(const VectorXd &z, const MatrixXd &R, const MatrixXd &H, const VectorXd &projected_state);

private:
	// state vector
	VectorXd x_;

	// state covariance matrix
	MatrixXd P_;

	// state transition matrix
	MatrixXd F_;

	// process covariance matrix
	MatrixXd Q_;

	// identity matrix
	MatrixXd I_;
};

#endif /* KALMAN_FILTER_H_ */
