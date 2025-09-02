/**
 * @file test_spline.cpp
 * @author Yanwei Du (yanwei.du@gatech.edu)
 * @brief None
 * @version 0.1
 * @date 09-01-2025
 * @copyright Copyright (c) 2025
 */

#include <cmath>
#include <deque>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <vector>

#include "sim/BsplineSE3.h"
#include "utils/dataset_reader.h"

using namespace ov_core;

namespace internal {
struct StampedPose {
  double timestamp, tx, ty, tz, qx, qy, qz, qw;

  StampedPose(double _timestamp, const Eigen::Matrix3d &R, const Eigen::Vector3d &t) {
    timestamp = _timestamp;
    tx = t.x();
    ty = t.y();
    tz = t.z();
    Eigen::Quaterniond quat(R);
    qx = quat.x();
    qy = quat.y();
    qz = quat.z();
    qw = quat.w();
  }

  friend std::ostream &operator<<(std::ostream &os, const StampedPose &s) {
    os << std::fixed;
    os << std::setprecision(10) << s.timestamp << " " << std::setprecision(6) << s.tx << " " << s.ty << " " << s.tz << " " << s.qx << " "
       << s.qy << " " << s.qz << " " << s.qw;
    return os;
  }

  static std::string header() { return "# timestamp tx ty tz qx qy qz qw"; }
};

void save(const std::string &filename, const std::vector<StampedPose> &poses) {
  std::ofstream myfile(filename);
  myfile << StampedPose::header() << "\n";
  for (const auto &p : poses) {
    myfile << p << "\n";
  }
  myfile.close();
}
} // namespace internal

int main(int argc, char **argv) {

  std::string data_root = "/mnt/IVALAB/rosbags/tsrb/";
  std::vector<std::string> seqnames{
      "20241012", "20250330", "20250331", "20250530", "20250619", "new/msf/two_loops", "new/msf/one_big_loop", "new/msf/inspection",
  };

  for (const auto &seq_name : seqnames) {
    std::cout << "Processing " << seq_name << "..." << "\n";
    std::string filepath = data_root + "/" + seq_name + "/slam_toolbox/slam_toolbox_KeyFrameTrajectory.txt";
    // Load samples.
    std::vector<Eigen::VectorXd> samples;
    DatasetReader::load_simulated_trajectory(filepath, samples);
    if (samples.size() < 10u) {
      std::cout << "Insufficient samples: " << samples.size() << "\n";
      continue;
    }
    double start_t = samples.front()(0);
    double end_t = samples.back()(0);
    double dt = 1.0 / 100.0;

    // Fit splines.
    BsplineSE3 spline;
    spline.feed_trajectory(samples);
    Eigen::Matrix3d R, R_cam, R_bc;
    Eigen::Vector3d p, p_cam, p_bc;
    R_bc << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    p_bc << 0.125, 0.040, 0.50;
    std::vector<internal::StampedPose> body_poses;
    std::vector<internal::StampedPose> cam_poses;
    while (start_t <= end_t) {
      bool success = spline.get_pose(start_t, R, p);
      if (success) {
        body_poses.emplace_back(start_t, R, p);
        R_cam = R * R_bc;
        p_cam = R * p_bc + p;
        cam_poses.emplace_back(start_t, R_cam, p_cam);
      }
      start_t += dt;
    }

    // Save poses.
    {
      std::string outfile = data_root + "/" + seq_name + "/slam_toolbox/gt_pose_body.txt";
      internal::save(outfile, body_poses);

      outfile = data_root + "/" + seq_name + "/slam_toolbox/gt_pose_cam0.txt";
      internal::save(outfile, cam_poses);
    }
    std::cout << "Done!" << std::endl;
  }
  return 0;
}