/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Offline, ROS-free AIGP dataset feeder for OpenVINS. This executable links
 * against OpenVINS, which is licensed under GPL-3.0-or-later.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "types/Landmark.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace {

struct ImuRow {
  double timestamp;
  Eigen::Vector3d wm;
  Eigen::Vector3d am;
};

struct CameraRow {
  double timestamp;
  std::string filename;
  long source_frame_id;
};

std::string join_path(const std::string &left, const std::string &right) {
  if (left.empty()) {
    return right;
  }
  if (left.back() == '/') {
    return left + right;
  }
  return left + "/" + right;
}

constexpr double kFixedContrastWindowS = 2.0;
constexpr int kFixedContrastTargetMedian = 48;
constexpr double kFixedContrastMinGamma = 0.55;

cv::Mat decode_red_channel(const std::string &image_path) {
  const cv::Mat color_image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (color_image.empty()) {
    return cv::Mat();
  }
  cv::Mat red_image;
  // OpenCV decodes color images in BGR order, so channel 2 is red.
  cv::extractChannel(color_image, red_image, 2);
  return red_image;
}

std::vector<cv::Point> guide_cone_polygon_for(const cv::Mat &image) {
  const auto scaled_x = [&image](double x) {
    return static_cast<int>(std::lround(x * image.cols));
  };
  const auto scaled_y = [&image](double y) {
    return static_cast<int>(std::lround(y * image.rows));
  };
  return {
      cv::Point(scaled_x(285.0 / 640.0), scaled_y(130.0 / 360.0)),
      cv::Point(scaled_x(355.0 / 640.0), scaled_y(130.0 / 360.0)),
      cv::Point(scaled_x(590.0 / 640.0), image.rows - 1),
      cv::Point(scaled_x(50.0 / 640.0), image.rows - 1),
  };
}

cv::Mat equalize_red_from_side_regions(const cv::Mat &red_image) {
  cv::Mat histogram_mask(red_image.rows, red_image.cols, CV_8UC1,
                         cv::Scalar(255));
  cv::fillConvexPoly(histogram_mask, guide_cone_polygon_for(red_image),
                     cv::Scalar(0));
  std::array<std::uint64_t, 256> histogram{};
  std::uint64_t pixel_count = 0;
  for (int row = 0; row < red_image.rows; ++row) {
    const auto *pixels = red_image.ptr<std::uint8_t>(row);
    const auto *included = histogram_mask.ptr<std::uint8_t>(row);
    for (int col = 0; col < red_image.cols; ++col) {
      if (included[col] != 0) {
        ++histogram[pixels[col]];
        ++pixel_count;
      }
    }
  }
  if (pixel_count == 0) {
    throw std::runtime_error(
        "side-region histogram contains no included pixels");
  }

  int first_nonzero = 0;
  while (first_nonzero < 256 && histogram[first_nonzero] == 0) {
    ++first_nonzero;
  }
  cv::Mat lut(1, 256, CV_8UC1, cv::Scalar(0));
  if (first_nonzero >= 255 || histogram[first_nonzero] == pixel_count) {
    for (int intensity = 0; intensity < 256; ++intensity) {
      lut.at<std::uint8_t>(0, intensity) =
          static_cast<std::uint8_t>(intensity);
    }
  } else {
    const std::uint64_t cdf_min = histogram[first_nonzero];
    const double scale = 255.0 / static_cast<double>(pixel_count - cdf_min);
    std::uint64_t cumulative = 0;
    for (int intensity = first_nonzero + 1; intensity < 256; ++intensity) {
      cumulative += histogram[intensity];
      lut.at<std::uint8_t>(0, intensity) = static_cast<std::uint8_t>(
          std::lround(std::max(0.0, std::min(255.0, cumulative * scale))));
    }
  }
  cv::Mat equalized;
  cv::LUT(red_image, lut, equalized);
  return equalized;
}

cv::Mat build_fixed_red_contrast_lut(const std::string &replay_dir,
                                     const std::vector<CameraRow> &camera_rows) {
  std::array<std::uint64_t, 256> histogram{};
  std::uint64_t pixel_count = 0;
  std::size_t frame_count = 0;
  const double end_timestamp =
      camera_rows.front().timestamp + kFixedContrastWindowS;
  for (const auto &camera : camera_rows) {
    if (camera.timestamp > end_timestamp) {
      break;
    }
    const std::string image_path =
        join_path(join_path(replay_dir, "cam0"), camera.filename);
    const cv::Mat red_image = decode_red_channel(image_path);
    if (red_image.empty()) {
      throw std::runtime_error(
          "unable to decode fixed-contrast calibration image: " + image_path);
    }
    for (int row = 0; row < red_image.rows; ++row) {
      const auto *pixels = red_image.ptr<std::uint8_t>(row);
      for (int col = 0; col < red_image.cols; ++col) {
        ++histogram[pixels[col]];
      }
    }
    pixel_count += static_cast<std::uint64_t>(red_image.total());
    ++frame_count;
  }
  if (frame_count == 0 || pixel_count == 0) {
    throw std::runtime_error(
        "fixed red contrast has no images in its calibration window");
  }

  const std::uint64_t median_rank = (pixel_count + 1) / 2;
  std::uint64_t cumulative = 0;
  int median_intensity = 0;
  for (int intensity = 0; intensity < 256; ++intensity) {
    cumulative += histogram[static_cast<std::size_t>(intensity)];
    if (cumulative >= median_rank) {
      median_intensity = intensity;
      break;
    }
  }

  double gamma = 1.0;
  if (median_intensity > 0 && median_intensity < 255) {
    gamma = std::log(kFixedContrastTargetMedian / 255.0) /
            std::log(median_intensity / 255.0);
    gamma = std::max(kFixedContrastMinGamma, std::min(1.0, gamma));
  }
  cv::Mat lut(1, 256, CV_8UC1);
  for (int intensity = 0; intensity < 256; ++intensity) {
    const double normalized = intensity / 255.0;
    const double mapped =
        255.0 * std::pow(normalized, gamma);
    lut.at<std::uint8_t>(0, intensity) = static_cast<std::uint8_t>(
        std::lround(std::max(0.0, std::min(255.0, mapped))));
  }
  std::cout << "Fixed red contrast: " << frame_count << " frames over "
            << kFixedContrastWindowS << " s, source median "
            << median_intensity << ", target median "
            << kFixedContrastTargetMedian << ", gamma " << gamma << "\n";
  return lut;
}

std::vector<std::string> split_csv(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

std::vector<ImuRow> read_imu(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open IMU CSV: " + path);
  }
  std::string line;
  std::getline(input, line);
  if (line != "timestamp,wx,wy,wz,ax,ay,az") {
    throw std::runtime_error("unexpected IMU CSV header: " + line);
  }

  std::vector<ImuRow> rows;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const auto fields = split_csv(line);
    if (fields.size() != 7) {
      throw std::runtime_error("invalid IMU CSV row: " + line);
    }
    ImuRow row;
    row.timestamp = std::stod(fields[0]);
    row.wm << std::stod(fields[1]), std::stod(fields[2]), std::stod(fields[3]);
    row.am << std::stod(fields[4]), std::stod(fields[5]), std::stod(fields[6]);
    if (!rows.empty() && row.timestamp <= rows.back().timestamp) {
      throw std::runtime_error("IMU timestamps are not strictly increasing");
    }
    rows.push_back(row);
  }
  if (rows.empty()) {
    throw std::runtime_error("IMU CSV contains no rows");
  }
  return rows;
}

std::vector<CameraRow> read_camera(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("unable to open camera CSV: " + path);
  }
  std::string line;
  std::getline(input, line);
  if (line != "timestamp,filename,source_frame_id,source_sim_time_ns") {
    throw std::runtime_error("unexpected camera CSV header: " + line);
  }

  std::vector<CameraRow> rows;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const auto fields = split_csv(line);
    if (fields.size() != 4) {
      throw std::runtime_error("invalid camera CSV row: " + line);
    }
    CameraRow row;
    row.timestamp = std::stod(fields[0]);
    row.filename = fields[1];
    row.source_frame_id = std::stol(fields[2]);
    if (!rows.empty() && row.timestamp <= rows.back().timestamp) {
      throw std::runtime_error("camera timestamps are not strictly increasing");
    }
    rows.push_back(row);
  }
  if (rows.empty()) {
    throw std::runtime_error("camera CSV contains no rows");
  }
  return rows;
}

void write_header(std::ofstream &output) {
  output << "timestamp,q_GtoI_x,q_GtoI_y,q_GtoI_z,q_GtoI_w,"
            "p_IinG_x,p_IinG_y,p_IinG_z,v_IinG_x,v_IinG_y,v_IinG_z,"
            "bias_g_x,bias_g_y,bias_g_z,bias_a_x,bias_a_y,bias_a_z\n";
}

void write_state(std::ofstream &output, const std::shared_ptr<ov_msckf::State> &state) {
  const Eigen::VectorXd q = state->_imu->quat();
  const Eigen::VectorXd p = state->_imu->pos();
  const Eigen::VectorXd v = state->_imu->vel();
  const Eigen::VectorXd bg = state->_imu->bias_g();
  const Eigen::VectorXd ba = state->_imu->bias_a();
  output << std::setprecision(17) << state->_timestamp;
  for (int index = 0; index < 4; ++index) output << ',' << q(index);
  for (int index = 0; index < 3; ++index) output << ',' << p(index);
  for (int index = 0; index < 3; ++index) output << ',' << v(index);
  for (int index = 0; index < 3; ++index) output << ',' << bg(index);
  for (int index = 0; index < 3; ++index) output << ',' << ba(index);
  output << '\n';
}

void write_diagnostics_header(std::ofstream &output) {
  output << "camera_timestamp,source_frame_id,initialized,state_timestamp,"
            "state_advanced,msckf_features_used,slam_features_in_state,"
            "active_tracks,msckf_lost_candidates,msckf_marginal_candidates,"
            "msckf_maxtrack_candidates,msckf_candidates_before_limit,"
            "msckf_candidates_after_limit,msckf_input_features,"
            "msckf_input_measurements,msckf_rejected_too_few,"
            "msckf_rejected_triangulation,msckf_triangulation_bad_condition,"
            "msckf_triangulation_depth_too_near,"
            "msckf_triangulation_depth_too_far,"
            "msckf_triangulation_invalid_numeric,msckf_triangulation_other,"
            "msckf_rejected_refinement,msckf_refinement_depth_too_near,"
            "msckf_refinement_depth_too_far,msckf_refinement_baseline_ratio,"
            "msckf_refinement_invalid_numeric,msckf_refinement_other,"
            "msckf_chi2_tested,msckf_rejected_chi2,"
            "msckf_geometry_valid_features,msckf_post_chi2_features,"
            "msckf_rejected_selection_limit,msckf_selected_features,"
            "msckf_accepted_features,"
            "msckf_accepted_measurements,msckf_chi2_ratio_mean,"
            "msckf_chi2_ratio_max,position_norm,speed,bias_gyro_norm,"
            "bias_accel_norm\n";
}

void write_diagnostics(std::ofstream &output, const CameraRow &camera,
                       bool initialized, double state_timestamp_before,
                       const std::shared_ptr<ov_msckf::VioManager> &system) {
  const auto state = system->get_state();
  double active_timestamp = -1.0;
  std::unordered_map<size_t, Eigen::Vector3d> active_positions;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks;
  system->get_active_tracks(active_timestamp, active_positions, active_tracks);
  const auto msckf = system->get_last_msckf_update_diagnostics();

  output << std::setprecision(17) << camera.timestamp << ','
         << camera.source_frame_id << ',' << (initialized ? 1 : 0) << ','
         << state->_timestamp << ','
         << (state->_timestamp > state_timestamp_before ? 1 : 0) << ','
         << system->get_good_features_MSCKF().size() << ','
         << state->_features_SLAM.size() << ',' << active_tracks.size() << ','
         << msckf.lost_features << ',' << msckf.marginal_features << ','
         << msckf.maxtrack_features << ',' << msckf.candidates_before_limit
         << ',' << msckf.candidates_after_limit << ',' << msckf.input_features
         << ',' << msckf.input_measurements << ','
         << msckf.rejected_too_few_measurements << ','
         << msckf.rejected_triangulation << ','
         << msckf.triangulation_bad_condition << ','
         << msckf.triangulation_depth_too_near << ','
         << msckf.triangulation_depth_too_far << ','
         << msckf.triangulation_invalid_numeric << ','
         << msckf.triangulation_other << ','
         << msckf.rejected_refinement << ','
         << msckf.refinement_depth_too_near << ','
         << msckf.refinement_depth_too_far << ','
         << msckf.refinement_baseline_ratio << ','
         << msckf.refinement_invalid_numeric << ','
         << msckf.refinement_other << ',' << msckf.chi2_tested << ','
         << msckf.rejected_chi2 << ',' << msckf.geometry_valid_features << ','
         << msckf.post_chi2_features << ','
         << msckf.rejected_selection_limit << ',' << msckf.selected_features
         << ',' << msckf.accepted_features << ','
         << msckf.accepted_measurements << ',' << msckf.chi2_ratio_mean()
         << ',' << msckf.chi2_ratio_max;
  if (initialized) {
    output << ',' << state->_imu->pos().norm() << ',' << state->_imu->vel().norm()
           << ',' << state->_imu->bias_g().norm() << ','
           << state->_imu->bias_a().norm();
  } else {
    output << ",,,,";
  }
  output << '\n';
}

void write_feature_geometry_header(std::ofstream &output) {
  output << "camera_timestamp,source_frame_id,feature_id,result,"
            "first_u_px,first_v_px,last_u_px,last_v_px,mean_u_px,mean_v_px,"
            "min_u_px,max_u_px,min_v_px,max_v_px,"
            "triangulation_success,refinement_success,chi2_tested,"
            "selected_for_update,selection_rank,selection_grid_row,"
            "selection_grid_col,accepted,"
            "triangulation_failure_reason,refinement_failure_reason,"
            "observation_count,track_duration_s,max_parallax_deg,"
            "max_camera_baseline_m,condition_number,linear_depth_m,"
            "linear_range_m,refined_depth_m,refined_range_m,chi2_ratio\n";
}

void write_feature_geometry(
    std::ofstream &output, const CameraRow &camera,
    const std::shared_ptr<ov_msckf::VioManager> &system) {
  const auto diagnostics = system->get_last_msckf_update_diagnostics();
  for (const auto &feature : diagnostics.feature_geometry) {
    output << std::setprecision(17) << camera.timestamp << ','
           << camera.source_frame_id << ',' << feature.feature_id << ','
           << static_cast<int>(feature.result) << ',' << feature.first_u_px
           << ',' << feature.first_v_px << ',' << feature.last_u_px << ','
           << feature.last_v_px << ',' << feature.mean_u_px << ','
           << feature.mean_v_px << ',' << feature.min_u_px << ','
           << feature.max_u_px << ',' << feature.min_v_px << ','
           << feature.max_v_px << ','
           << (feature.triangulation_success ? 1 : 0) << ','
           << (feature.refinement_success ? 1 : 0) << ','
           << (feature.chi2_tested ? 1 : 0) << ','
           << (feature.selected_for_update ? 1 : 0) << ','
           << feature.selection_rank << ',' << feature.selection_grid_row
           << ',' << feature.selection_grid_col << ','
           << (feature.accepted ? 1 : 0) << ','
           << feature.triangulation_failure_reason << ','
           << feature.refinement_failure_reason << ','
           << feature.observation_count << ',' << feature.track_duration_s
           << ',' << feature.max_parallax_deg << ','
           << feature.max_camera_baseline_m << ',' << feature.condition_number
           << ',' << feature.linear_depth_m << ',' << feature.linear_range_m
           << ',' << feature.refined_depth_m << ',' << feature.refined_range_m
           << ',' << feature.chi2_ratio << '\n';
  }
}

void write_slam_feature_header(std::ofstream &output) {
  output << "camera_timestamp,source_frame_id,feature_id,tracked_in_current_frame,"
            "u_px,v_px,depth_m,anchor_camera_id,unique_camera_id,"
            "anchor_clone_timestamp,update_fail_count,should_marg\n";
}

void write_slam_features(
    std::ofstream &output, const CameraRow &camera,
    const std::shared_ptr<ov_msckf::VioManager> &system) {
  const auto state = system->get_state();
  double active_timestamp = -1.0;
  std::unordered_map<size_t, Eigen::Vector3d> active_positions;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks;
  system->get_active_tracks(active_timestamp, active_positions, active_tracks);

  for (const auto &entry : state->_features_SLAM) {
    const auto &landmark = entry.second;
    const auto tracked = active_tracks.find(entry.first);
    output << std::setprecision(17) << camera.timestamp << ','
           << camera.source_frame_id << ',' << entry.first << ','
           << (tracked != active_tracks.end() ? 1 : 0);
    if (tracked != active_tracks.end()) {
      output << ',' << tracked->second(0) << ',' << tracked->second(1) << ','
             << tracked->second(2);
    } else {
      output << ",,,";
    }
    output << ',' << landmark->_anchor_cam_id << ','
           << landmark->_unique_camera_id << ','
           << landmark->_anchor_clone_timestamp << ','
           << landmark->update_fail_count << ','
           << (landmark->should_marg ? 1 : 0) << '\n';
  }
}

int run_live_stream(const std::string &config_path,
                    const std::string &camera_image_mode) {
  if (camera_image_mode != "grayscale" && camera_image_mode != "red") {
    throw std::runtime_error(
        "live CAMERA_IMAGE_MODE must be grayscale or red");
  }

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  ov_msckf::VioManagerOptions params;
  params.print_and_load(parser);
  params.num_opencv_threads = 0;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  auto system = std::make_shared<ov_msckf::VioManager>(params);
  if (!parser->successful()) {
    throw std::runtime_error(
        "OpenVINS could not parse all live configuration fields");
  }

  std::string line;
  double last_state_timestamp = -1.0;
  std::size_t imu_count = 0;
  std::size_t camera_count = 0;
  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream header(line);
    char kind = '\0';
    header >> kind;
    if (kind == 'Q') {
      break;
    }
    if (kind == 'I') {
      ov_core::ImuData message;
      header >> message.timestamp >> message.wm(0) >> message.wm(1) >>
          message.wm(2) >> message.am(0) >> message.am(1) >> message.am(2);
      if (!header || !std::isfinite(message.timestamp) ||
          !message.wm.allFinite() || !message.am.allFinite()) {
        throw std::runtime_error("invalid live IMU message: " + line);
      }
      system->feed_measurement_imu(message);
      ++imu_count;
      continue;
    }
    if (kind != 'C') {
      throw std::runtime_error("unknown live message: " + line);
    }

    double timestamp = 0.0;
    long frame_id = -1;
    std::size_t jpeg_size = 0;
    header >> timestamp >> frame_id >> jpeg_size;
    if (!header || !std::isfinite(timestamp) || jpeg_size == 0 ||
        jpeg_size > 4000000) {
      throw std::runtime_error("invalid live camera header: " + line);
    }
    std::vector<unsigned char> jpeg(jpeg_size);
    std::cin.read(reinterpret_cast<char *>(jpeg.data()),
                  static_cast<std::streamsize>(jpeg.size()));
    if (std::cin.gcount() != static_cast<std::streamsize>(jpeg.size())) {
      throw std::runtime_error("truncated live JPEG payload");
    }
    // The Python writer terminates each binary payload with one newline.
    char delimiter = '\0';
    std::cin.get(delimiter);
    if (delimiter != '\n') {
      throw std::runtime_error("live JPEG payload is missing delimiter");
    }

    cv::Mat color = cv::imdecode(jpeg, cv::IMREAD_COLOR);
    if (color.empty()) {
      throw std::runtime_error("unable to decode live JPEG");
    }
    cv::Mat image;
    if (camera_image_mode == "red") {
      cv::extractChannel(color, image, 2);
    } else {
      cv::cvtColor(color, image, cv::COLOR_BGR2GRAY);
    }

    ov_core::CameraData message;
    message.timestamp = timestamp;
    message.sensor_ids.push_back(0);
    message.images.push_back(image);
    message.masks.push_back(
        cv::Mat::zeros(image.rows, image.cols, CV_8UC1));
    system->feed_measurement_camera(message);
    ++camera_count;

    if (!system->initialized()) {
      if (camera_count % 30 == 0) {
        std::cout << "AIGP_VIO_WAIT camera=" << camera_count
                  << " imu=" << imu_count << '\n' << std::flush;
      }
      continue;
    }
    const auto state = system->get_state();
    if (state->_timestamp <= last_state_timestamp) {
      continue;
    }
    const Eigen::VectorXd q = state->_imu->quat();
    const Eigen::VectorXd p = state->_imu->pos();
    const Eigen::VectorXd v = state->_imu->vel();
    std::cout << std::setprecision(17) << "AIGP_VIO_STATE "
              << state->_timestamp;
    for (int index = 0; index < 4; ++index) {
      std::cout << ' ' << q(index);
    }
    for (int index = 0; index < 3; ++index) {
      std::cout << ' ' << p(index);
    }
    for (int index = 0; index < 3; ++index) {
      std::cout << ' ' << v(index);
    }
    std::cout << '\n' << std::flush;
    last_state_timestamp = state->_timestamp;
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc >= 3 && std::string(argv[1]) == "--live") {
    try {
      return run_live_stream(argv[2], argc >= 4 ? argv[3] : "red");
    } catch (const std::exception &error) {
      std::cerr << "ERROR: " << error.what() << '\n';
      return 2;
    }
  }
  if (argc < 2 || argc > 7) {
    std::cerr << "usage: run_vq1_dataset REPLAY_DIRECTORY [OUTPUT_CSV] "
                 "[DIAGNOSTICS_CSV] [MASK_TOP_ROWS] [MASK_GUIDE_CONE] "
                 "[CAMERA_IMAGE_MODE]\n";
    return 64;
  }

  const std::string replay_dir = argv[1];
  const std::string output_path =
      argc >= 3 ? argv[2] : join_path(replay_dir, "openvins_estimate.csv");
  const std::string diagnostics_path =
      argc >= 4 ? argv[3]
                : join_path(replay_dir, "openvins_update_diagnostics.csv");
  const std::string feature_geometry_path =
      join_path(replay_dir, "openvins_feature_geometry.csv");
  const std::string slam_feature_path =
      join_path(replay_dir, "openvins_slam_feature_diagnostics.csv");
  const int mask_top_rows = argc >= 5 ? std::stoi(argv[4]) : 0;
  const int mask_guide_cone_value = argc >= 6 ? std::stoi(argv[5]) : 0;
  const std::string camera_image_mode = argc >= 7 ? argv[6] : "grayscale";
  if (mask_top_rows < 0) {
    std::cerr << "ERROR: MASK_TOP_ROWS must be nonnegative\n";
    return 64;
  }
  if (mask_guide_cone_value != 0 && mask_guide_cone_value != 1) {
    std::cerr << "ERROR: MASK_GUIDE_CONE must be 0 or 1\n";
    return 64;
  }
  if (camera_image_mode != "grayscale" && camera_image_mode != "red" &&
      camera_image_mode != "red_fixed" &&
      camera_image_mode != "red_sidehist") {
    std::cerr << "ERROR: CAMERA_IMAGE_MODE must be grayscale, red, or "
                 "red_fixed, or red_sidehist\n";
    return 64;
  }
  const bool mask_guide_cone = mask_guide_cone_value == 1;

  try {
    const auto imu_rows = read_imu(join_path(replay_dir, "imu.csv"));
    const auto camera_rows = read_camera(join_path(replay_dir, "cam0/data.csv"));
    cv::Mat fixed_red_contrast_lut;
    if (camera_image_mode == "red_fixed") {
      fixed_red_contrast_lut =
          build_fixed_red_contrast_lut(replay_dir, camera_rows);
    }
    const std::string config_path =
        join_path(replay_dir, "config/estimator_config.yaml");

    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
    std::string verbosity = "INFO";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);

    ov_msckf::VioManagerOptions params;
    params.print_and_load(parser);
    params.num_opencv_threads = 0;
    params.use_multi_threading_pubs = false;
    params.use_multi_threading_subs = false;
    auto system = std::make_shared<ov_msckf::VioManager>(params);
    if (!parser->successful()) {
      throw std::runtime_error("OpenVINS could not parse all configuration fields");
    }

    std::ofstream output(output_path, std::ios::trunc);
    if (!output) {
      throw std::runtime_error("unable to create estimate CSV: " + output_path);
    }
    write_header(output);

    std::ofstream diagnostics(diagnostics_path, std::ios::trunc);
    if (!diagnostics) {
      throw std::runtime_error("unable to create diagnostics CSV: " +
                               diagnostics_path);
    }
    write_diagnostics_header(diagnostics);

    std::ofstream feature_geometry(feature_geometry_path, std::ios::trunc);
    if (!feature_geometry) {
      throw std::runtime_error("unable to create feature geometry CSV: " +
                               feature_geometry_path);
    }
    write_feature_geometry_header(feature_geometry);

    std::ofstream slam_features(slam_feature_path, std::ios::trunc);
    if (!slam_features) {
      throw std::runtime_error("unable to create SLAM feature diagnostics CSV: " +
                               slam_feature_path);
    }
    write_slam_feature_header(slam_features);

    std::size_t imu_index = 0;
    std::size_t initialized_camera_index = 0;
    std::size_t estimate_rows = 0;
    std::size_t bracketed_camera_rows = 0;
    double bracket_lead_sum_s = 0.0;
    double bracket_lead_max_s = 0.0;
    double last_written_timestamp = -1.0;
    const double camera_imu_offset_s = params.calib_camimu_dt;

    const auto feed_imu_row = [&system](const ImuRow &row) {
      ov_core::ImuData message;
      message.timestamp = row.timestamp;
      message.wm = row.wm;
      message.am = row.am;
      system->feed_measurement_imu(message);
    };

    for (std::size_t camera_index = 0; camera_index < camera_rows.size(); ++camera_index) {
      const auto &camera = camera_rows[camera_index];
      const double effective_imu_timestamp =
          camera.timestamp + camera_imu_offset_s;
      while (imu_index < imu_rows.size() &&
             imu_rows[imu_index].timestamp <= effective_imu_timestamp) {
        feed_imu_row(imu_rows[imu_index]);
        ++imu_index;
      }
      // OpenVINS interpolates to camera time plus calib_camimu_dt using two
      // IMU samples. Buffer against that effective timestamp so a nonzero
      // time offset cannot force extrapolation.
      // Buffer one strictly newer sample before feeding the image so its
      // Propagator never has to extrapolate the last two measurements.
      if (imu_index >= imu_rows.size()) {
        throw std::runtime_error(
            "camera has no strictly newer IMU sample at t=" +
            std::to_string(effective_imu_timestamp));
      }
      const double bracket_lead_s =
          imu_rows[imu_index].timestamp - effective_imu_timestamp;
      if (bracket_lead_s <= 0.0) {
        throw std::runtime_error("internal error: IMU bracket is not in the future");
      }
      feed_imu_row(imu_rows[imu_index]);
      ++imu_index;
      ++bracketed_camera_rows;
      bracket_lead_sum_s += bracket_lead_s;
      if (bracket_lead_s > bracket_lead_max_s) {
        bracket_lead_max_s = bracket_lead_s;
      }

      const std::string image_path =
          join_path(join_path(replay_dir, "cam0"), camera.filename);
      cv::Mat image;
      if (camera_image_mode == "red" || camera_image_mode == "red_fixed" ||
          camera_image_mode == "red_sidehist") {
        const cv::Mat red_image = decode_red_channel(image_path);
        if (!red_image.empty()) {
          if (camera_image_mode == "red_fixed") {
            cv::LUT(red_image, fixed_red_contrast_lut, image);
          } else if (camera_image_mode == "red_sidehist") {
            image = equalize_red_from_side_regions(red_image);
          } else {
            image = red_image;
          }
        }
      } else {
        image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
      }
      if (image.empty()) {
        throw std::runtime_error("unable to decode camera image: " + image_path);
      }
      if (mask_top_rows >= image.rows) {
        throw std::runtime_error(
            "MASK_TOP_ROWS must be smaller than the camera height");
      }
      cv::Mat feature_mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
      // The simulator's translucent blue guide cone changes while the vehicle
      // is physically stationary. Exclude its screen-fixed envelope before
      // both initialization and tracking so it cannot violate the static-world
      // assumption. Coordinates scale from the audited 640x360 raster.
      if (mask_guide_cone) {
        cv::fillConvexPoly(feature_mask, guide_cone_polygon_for(image),
                           cv::Scalar(255));
      }
      // Preserve the historical top-row experiment's baseline initializer;
      // unlike the guide-cone exclusion, that mask activates only afterward.
      if (mask_top_rows > 0 && system->initialized()) {
        feature_mask.rowRange(0, mask_top_rows).setTo(cv::Scalar(255));
      }
      ov_core::CameraData message;
      message.timestamp = camera.timestamp;
      message.sensor_ids.push_back(0);
      message.images.push_back(image);
      message.masks.push_back(feature_mask);
      const double state_timestamp_before = system->get_state()->_timestamp;
      system->feed_measurement_camera(message);

      write_diagnostics(diagnostics, camera, system->initialized(),
                        state_timestamp_before, system);
      write_feature_geometry(feature_geometry, camera, system);
      if (system->initialized()) {
        write_slam_features(slam_features, camera, system);
      }

      if (system->initialized()) {
        if (initialized_camera_index == 0) {
          initialized_camera_index = camera_index + 1;
          std::cout << "OpenVINS initialized at camera " << (camera_index + 1)
                    << "/" << camera_rows.size() << ", t="
                    << system->initialized_time() << " s\n";
        }
        const auto state = system->get_state();
        if (state->_timestamp > last_written_timestamp) {
          write_state(output, state);
          last_written_timestamp = state->_timestamp;
          ++estimate_rows;
        }
      }

      if ((camera_index + 1) % 100 == 0 || camera_index + 1 == camera_rows.size()) {
        std::cout << "Processed camera " << (camera_index + 1) << "/"
                  << camera_rows.size() << ", IMU " << imu_index << "/"
                  << imu_rows.size() << ", bracketed "
                  << bracketed_camera_rows << "\n";
      }
    }

    output.close();
    diagnostics.close();
    feature_geometry.close();
    slam_features.close();
    if (!system->initialized() || estimate_rows == 0) {
      std::cerr << "ERROR: OpenVINS did not initialize; no estimates were produced\n";
      return 3;
    }
    if (bracketed_camera_rows != camera_rows.size()) {
      std::cerr << "ERROR: not every camera update had a future IMU bracket\n";
      return 4;
    }
    std::cout << "PASS: wrote " << estimate_rows << " estimates to " << output_path
              << " and per-camera update diagnostics to " << diagnostics_path
              << " and per-feature geometry to " << feature_geometry_path
              << " and persistent-SLAM feature diagnostics to "
              << slam_feature_path
              << " (camera image mode " << camera_image_mode
              << "; initialized at source camera index " << initialized_camera_index
              << "; future-IMU brackets " << bracketed_camera_rows << "/"
              << camera_rows.size() << ", mean lead "
              << (1000.0 * bracket_lead_sum_s / bracketed_camera_rows)
              << " ms, max lead " << (1000.0 * bracket_lead_max_s) << " ms)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
  }
}
