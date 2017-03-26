#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

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
  H_laser_ = MatrixXd(2, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0,      0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0,    0,
              0,    9e-4, 0,
              0,    0,    0.09;

  ekf_.x_ = VectorXd(4);
  ekf_.P_ = MatrixXd(4,4);
  ekf_.F_ = MatrixXd(4,4);
  ekf_.Q_ = MatrixXd(4,4);

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  ekf_.P_ << 1, 0, 0,   0,
             0, 1, 0,   0,
             0, 0, 1e3, 0,
             0, 0, 0,   1e3;

  ekf_.F_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;

  ekf_.Q_ << 0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0,
             0, 0, 0, 0;

  noise_ax_ = 9;
  noise_ay_ = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "EKF: " << endl;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho     = measurement_pack.raw_measurements_[0];
      float phi     = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_ << rho * cos(phi), rho * sin(phi), rho_dot * cos(phi), rho_dot * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1e6;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_(0,2) = dt;
	ekf_.F_(1,3) = dt;

	ekf_.Q_(0,0) = (pow(dt,4)/4) * noise_ax_;
	ekf_.Q_(0,2) = (pow(dt,3)/2) * noise_ax_;
	ekf_.Q_(1,1) = (pow(dt,4)/4) * noise_ay_;
	ekf_.Q_(1,3) = (pow(dt,3)/2) * noise_ay_;
	ekf_.Q_(2,0) = (pow(dt,3)/2) * noise_ax_;
	ekf_.Q_(2,2) = (pow(dt,2)/1) * noise_ax_;
	ekf_.Q_(3,1) = (pow(dt,3)/2) * noise_ay_;
	ekf_.Q_(3,3) = (pow(dt,2)/1) * noise_ay_;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  VectorXd pred = VectorXd(3);

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    Tools t;
    ekf_.H_ = t.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;

    double px, py, vx, vy, rho, phi, rho_dot;

    px = ekf_.x_[0];
    py = ekf_.x_[1];
    vx = ekf_.x_[2];
    vy = ekf_.x_[3];

    rho     = sqrt(px * px + py * py);
    phi     = fabs(px)  > 1e-4 ? atan(py / px) : 0;
    rho_dot = fabs(rho) > 1e-4 ? (px * vx + py * vy) / rho : 0;

    pred << rho, phi, rho_dot;
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    pred = ekf_.H_ * ekf_.x_;
  }

  ekf_.Update(measurement_pack.raw_measurements_ - pred);

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
