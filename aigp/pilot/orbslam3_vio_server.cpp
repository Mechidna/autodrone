// ORB-SLAM3 VIO TCP server for orbslam3_vio_bridge.py.
//
// This is the real server entrypoint: it links against ORB-SLAM3, keeps one
// ORB_SLAM3::System alive, accepts line-delimited JSON requests, decodes the
// image/IMU samples, calls TrackMonocular(..., imu_samples), and returns JSON.
//
// Build with aigp/pilot/orbslam3_vio_server_CMakeLists.txt after ORB-SLAM3 is
// checked out and built.

#include <arpa/inet.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>

#include "System.h"
#include "ImuTypes.h"

namespace {

constexpr const char *kDefaultHost = "127.0.0.1";
constexpr int kDefaultPort = 45530;
constexpr int kTrackingSystemNotReady = -1;
constexpr int kTrackingNoImagesYet = 0;
constexpr int kTrackingNotInitialized = 1;
constexpr int kTrackingOk = 2;
constexpr int kTrackingRecentlyLost = 3;
constexpr int kTrackingLost = 4;
constexpr int kTrackingOkKlt = 5;
volatile std::sig_atomic_t g_should_stop = 0;

struct ServerConfig {
    std::string host = kDefaultHost;
    int port = kDefaultPort;
    std::string vocabulary_path;
    std::string settings_path;
    bool use_viewer = false;
};

struct ImuRequestDiagnostics {
    int count = 0;
    double first_t = 0.0;
    double last_t = 0.0;
    double mean_dt = 0.0;
    double max_dt = 0.0;
    Eigen::Vector3d accel_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_mean = Eigen::Vector3d::Zero();
    double accel_norm_mean = 0.0;
    double accel_dynamic_mean = 0.0;
    double gyro_norm_max = 0.0;
};

void handle_signal(int) {
    g_should_stop = 1;
}

void close_fd(int fd) {
    if (fd >= 0) {
        ::close(fd);
    }
}

std::string format_vec3(const Eigen::Vector3d &v, int precision = 3) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(precision);
    out << "(" << v.x() << "," << v.y() << "," << v.z() << ")";
    return out.str();
}

const char *json_bool(bool value) {
    return value ? "true" : "false";
}

void print_settings_diagnostics(const std::string &settings_path) {
    cv::FileStorage fs(settings_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "orbslam3 diagnostics: failed to open settings for diagnostics: "
                  << settings_path << std::endl;
        return;
    }

    cv::Mat tbc;
    fs["IMU.T_b_c1"] >> tbc;
    if (tbc.empty()) {
        std::cerr << "orbslam3 diagnostics: IMU.T_b_c1 missing from settings"
                  << std::endl;
        return;
    }

    cv::Mat tbc64;
    tbc.convertTo(tbc64, CV_64F);
    if (tbc64.rows != 4 || tbc64.cols != 4) {
        std::cerr << "orbslam3 diagnostics: IMU.T_b_c1 has unexpected shape "
                  << tbc64.rows << "x" << tbc64.cols << std::endl;
        return;
    }

    cv::Mat rbc = tbc64(cv::Rect(0, 0, 3, 3));
    cv::Mat tbc_translation = tbc64(cv::Rect(3, 0, 1, 3));
    std::ostringstream matrix;
    matrix.setf(std::ios::fixed);
    matrix.precision(6);
    matrix << "[";
    for (int r = 0; r < tbc64.rows; ++r) {
        if (r > 0) {
            matrix << ";";
        }
        for (int c = 0; c < tbc64.cols; ++c) {
            if (c > 0) {
                matrix << ",";
            }
            matrix << tbc64.at<double>(r, c);
        }
    }
    matrix << "]";

    Eigen::Vector3d translation(
        tbc_translation.at<double>(0, 0),
        tbc_translation.at<double>(1, 0),
        tbc_translation.at<double>(2, 0)
    );
    std::ostringstream determinant;
    determinant.setf(std::ios::fixed);
    determinant.precision(6);
    determinant << cv::determinant(rbc);

    std::cout
        << "orbslam3 diagnostics settings: IMU.T_b_c1="
        << matrix.str()
        << " Rdet=" << determinant.str()
        << " t_b_c=" << format_vec3(translation, 6)
        << std::endl;
}

std::optional<ImuRequestDiagnostics> summarize_imu_request(
    const std::vector<ORB_SLAM3::IMU::Point> &imu
) {
    if (imu.empty()) {
        return std::nullopt;
    }

    ImuRequestDiagnostics diag;
    diag.count = static_cast<int>(imu.size());
    diag.first_t = imu.front().t;
    diag.last_t = imu.back().t;

    double dt_sum = 0.0;
    int dt_count = 0;
    Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
    double accel_norm_sum = 0.0;
    for (std::size_t i = 0; i < imu.size(); ++i) {
        Eigen::Vector3d accel = imu[i].a.cast<double>();
        Eigen::Vector3d gyro = imu[i].w.cast<double>();
        accel_sum += accel;
        gyro_sum += gyro;
        accel_norm_sum += accel.norm();
        diag.gyro_norm_max = std::max(diag.gyro_norm_max, gyro.norm());
        if (i > 0) {
            double dt = imu[i].t - imu[i - 1].t;
            dt_sum += dt;
            diag.max_dt = std::max(diag.max_dt, dt);
            ++dt_count;
        }
    }
    diag.accel_mean = accel_sum / static_cast<double>(imu.size());
    diag.gyro_mean = gyro_sum / static_cast<double>(imu.size());
    diag.accel_norm_mean = accel_norm_sum / static_cast<double>(imu.size());
    diag.mean_dt = dt_count > 0 ? dt_sum / static_cast<double>(dt_count) : 0.0;

    double accel_dynamic_sum = 0.0;
    for (const auto &sample : imu) {
        Eigen::Vector3d accel = sample.a.cast<double>();
        accel_dynamic_sum += (accel - diag.accel_mean).norm();
    }
    diag.accel_dynamic_mean = accel_dynamic_sum / static_cast<double>(imu.size());
    return diag;
}

std::string json_escape(const std::string &input) {
    std::string out;
    out.reserve(input.size());
    for (char ch : input) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

int parse_int_arg(const char *value, const char *name) {
    char *end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        std::ostringstream oss;
        oss << "invalid " << name << ": " << value;
        throw std::runtime_error(oss.str());
    }
    if (parsed < 1 || parsed > 65535) {
        std::ostringstream oss;
        oss << name << " must be within [1, 65535]";
        throw std::runtime_error(oss.str());
    }
    return static_cast<int>(parsed);
}

ServerConfig parse_args(int argc, char **argv) {
    ServerConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = parse_int_arg(argv[++i], "port");
        } else if (arg == "--vocabulary" && i + 1 < argc) {
            config.vocabulary_path = argv[++i];
        } else if (arg == "--settings" && i + 1 < argc) {
            config.settings_path = argv[++i];
        } else if (arg == "--viewer") {
            config.use_viewer = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0]
                << " --vocabulary ORBvoc.txt --settings orbslam3.yaml"
                << " [--host 127.0.0.1] [--port 45530] [--viewer]\n";
            std::exit(0);
        } else {
            std::ostringstream oss;
            oss << "unknown or incomplete argument: " << arg;
            throw std::runtime_error(oss.str());
        }
    }
    if (config.vocabulary_path.empty()) {
        throw std::runtime_error("--vocabulary is required");
    }
    if (config.settings_path.empty()) {
        throw std::runtime_error("--settings is required");
    }
    return config;
}

int create_listen_socket(const ServerConfig &config) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("socket failed: ") + std::strerror(errno));
    }

    int opt = 1;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        close_fd(fd);
        throw std::runtime_error(std::string("setsockopt failed: ") + std::strerror(errno));
    }

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(config.port));
    if (::inet_pton(AF_INET, config.host.c_str(), &addr.sin_addr) != 1) {
        close_fd(fd);
        throw std::runtime_error("host must be an IPv4 address");
    }

    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        close_fd(fd);
        throw std::runtime_error(std::string("bind failed: ") + std::strerror(errno));
    }
    if (::listen(fd, 8) < 0) {
        close_fd(fd);
        throw std::runtime_error(std::string("listen failed: ") + std::strerror(errno));
    }
    return fd;
}

std::optional<std::string> read_json_line(int client_fd) {
    std::string line;
    char ch = '\0';
    while (!g_should_stop) {
        ssize_t n = ::recv(client_fd, &ch, 1, 0);
        if (n == 0) {
            if (line.empty()) {
                return std::nullopt;
            }
            return line;
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == ECONNRESET || errno == ECONNABORTED) {
                return std::nullopt;
            }
            throw std::runtime_error(std::string("recv failed: ") + std::strerror(errno));
        }
        if (ch == '\n') {
            return line;
        }
        if (ch != '\r') {
            line.push_back(ch);
        }
    }
    return std::nullopt;
}

bool send_all(int client_fd, const std::string &data) {
    const char *cursor = data.data();
    std::size_t remaining = data.size();
    while (remaining > 0) {
        ssize_t n = ::send(client_fd, cursor, remaining, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        cursor += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

std::optional<std::string> extract_string_field(
    const std::string &json,
    const std::string &field_name
) {
    std::string needle = "\"" + field_name + "\":\"";
    std::size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos += needle.size();
    std::size_t end = json.find('"', pos);
    if (end == std::string::npos) {
        return std::nullopt;
    }
    return json.substr(pos, end - pos);
}

std::optional<double> extract_number_field(
    const std::string &json,
    const std::string &field_name
) {
    std::string pattern = "\"" + field_name + R"("\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?))";
    std::regex re(pattern);
    std::smatch match;
    if (!std::regex_search(json, match, re)) {
        return std::nullopt;
    }
    return std::stod(match[1].str());
}

std::optional<std::vector<double>> extract_number_array_field(
    const std::string &json,
    const std::string &field_name
) {
    std::string pattern = "\"" + field_name + R"("\s*:\s*\[([^\]]*)\])";
    std::regex re(pattern);
    std::smatch match;
    if (!std::regex_search(json, match, re)) {
        return std::nullopt;
    }

    std::vector<double> values;
    std::string body = match[1].str();
    std::regex num_re(R"(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
    for (
        std::sregex_iterator it(body.begin(), body.end(), num_re), end;
        it != end;
        ++it
    ) {
        values.push_back(std::stod((*it)[0].str()));
    }
    return values;
}

std::vector<std::string> extract_imu_objects(const std::string &json) {
    std::vector<std::string> objects;
    std::string needle = "\"imu\":[";
    std::size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return objects;
    }
    pos += needle.size();

    int array_depth = 1;
    int object_depth = 0;
    std::size_t object_start = std::string::npos;
    for (std::size_t i = pos; i < json.size(); ++i) {
        char ch = json[i];
        if (ch == '[') {
            ++array_depth;
        } else if (ch == ']') {
            --array_depth;
            if (array_depth == 0) {
                break;
            }
        } else if (ch == '{') {
            if (object_depth == 0) {
                object_start = i;
            }
            ++object_depth;
        } else if (ch == '}') {
            --object_depth;
            if (object_depth == 0 && object_start != std::string::npos) {
                objects.push_back(json.substr(object_start, i - object_start + 1));
                object_start = std::string::npos;
            }
        }
    }
    return objects;
}

std::string base64_decode(const std::string &input) {
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> table(256, -1);
    for (int i = 0; i < static_cast<int>(alphabet.size()); ++i) {
        table[static_cast<unsigned char>(alphabet[i])] = i;
    }

    std::string out;
    int val = 0;
    int valb = -8;
    for (unsigned char ch : input) {
        if (ch == '=') {
            break;
        }
        if (table[ch] == -1) {
            continue;
        }
        val = (val << 6) + table[ch];
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

cv::Mat decode_image_from_request(const std::string &request) {
    auto encoded = extract_string_field(request, "data");
    if (!encoded.has_value()) {
        throw std::runtime_error("request frame.data missing");
    }
    std::string jpeg = base64_decode(*encoded);
    std::vector<uchar> bytes(jpeg.begin(), jpeg.end());
    cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("failed to decode request JPEG");
    }
    return image;
}

std::vector<ORB_SLAM3::IMU::Point> imu_points_from_request(const std::string &request) {
    std::vector<ORB_SLAM3::IMU::Point> points;
    for (const std::string &object : extract_imu_objects(request)) {
        auto accel = extract_number_array_field(object, "accel_xyz");
        auto gyro = extract_number_array_field(object, "gyro_xyz");
        auto time_usec = extract_number_field(object, "time_usec");
        if (!accel.has_value() || !gyro.has_value() || !time_usec.has_value()) {
            continue;
        }
        if (accel->size() != 3 || gyro->size() != 3) {
            continue;
        }
        double t = *time_usec / 1e6;
        points.emplace_back(
            static_cast<float>((*accel)[0]),
            static_cast<float>((*accel)[1]),
            static_cast<float>((*accel)[2]),
            static_cast<float>((*gyro)[0]),
            static_cast<float>((*gyro)[1]),
            static_cast<float>((*gyro)[2]),
            t
        );
    }
    return points;
}

std::optional<std::string> validate_imu_points(
    const std::vector<ORB_SLAM3::IMU::Point> &imu
) {
    if (imu.empty()) {
        return "orbslam3_empty_imu_window";
    }

    double previous_t = -std::numeric_limits<double>::infinity();
    for (std::size_t idx = 0; idx < imu.size(); ++idx) {
        const auto &sample = imu[idx];
        if (!std::isfinite(sample.t)) {
            return "orbslam3_nonfinite_imu_timestamp";
        }
        if (sample.t <= previous_t) {
            return "orbslam3_nonmonotonic_imu_timestamp";
        }
        previous_t = sample.t;

        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(sample.a[axis]) || !std::isfinite(sample.w[axis])) {
                return "orbslam3_nonfinite_imu_sample";
            }
        }
    }
    return std::nullopt;
}

double frame_timestamp_s_from_request(const std::string &request) {
    if (auto timestamp_ns = extract_number_field(request, "timestamp_ns")) {
        return *timestamp_ns / 1e9;
    }
    if (auto wall_time = extract_number_field(request, "wall_time")) {
        return *wall_time;
    }
    throw std::runtime_error("request frame timestamp missing");
}

std::string error_response(const std::string &message) {
    std::ostringstream out;
    out
        << "{"
        << "\"valid\":false,"
        << "\"status\":\"lost\","
        << "\"source\":\"orb_slam3_cpp\","
        << "\"failure_reason\":\"" << json_escape(message) << "\""
        << "}\n";
    return out.str();
}

std::string tracking_state_name(int tracking_state) {
    switch (tracking_state) {
        case kTrackingSystemNotReady: return "system_not_ready";
        case kTrackingNoImagesYet: return "no_images_yet";
        case kTrackingNotInitialized: return "initializing";
        case kTrackingOk: return "tracking";
        case kTrackingRecentlyLost: return "recently_lost";
        case kTrackingLost: return "lost";
        case kTrackingOkKlt: return "tracking";
        default: return "unknown";
    }
}

bool tracking_state_is_valid(int tracking_state) {
    return tracking_state == kTrackingOk || tracking_state == kTrackingOkKlt;
}

std::string failure_reason_for_tracking_state(int tracking_state) {
    switch (tracking_state) {
        case kTrackingSystemNotReady: return "orbslam3_system_not_ready";
        case kTrackingNoImagesYet: return "orbslam3_no_images_yet";
        case kTrackingNotInitialized: return "orbslam3_not_initialized";
        case kTrackingRecentlyLost: return "orbslam3_recently_lost";
        case kTrackingLost: return "orbslam3_lost";
        case kTrackingOk:
        case kTrackingOkKlt:
            return "";
        default: return "orbslam3_unknown_tracking_state";
    }
}

std::string tracking_response(
    const Sophus::SE3f &t_cw,
    int tracking_state,
    int tracked_features,
    int request_count,
    double timestamp_s,
    const ORB_SLAM3::OrbSlam3Diagnostics &diagnostics
) {
    Sophus::SE3f t_wc = t_cw.inverse();
    Eigen::Vector3f p = t_wc.translation();
    Eigen::Quaternionf q(t_wc.rotationMatrix());
    q.normalize();

    std::string status = tracking_state_name(tracking_state);
    std::string failure_reason = failure_reason_for_tracking_state(tracking_state);
    bool valid = tracking_state_is_valid(tracking_state);
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z())) {
        status = "lost";
        valid = false;
        failure_reason = "orbslam3_nonfinite_pose";
    }

    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(9);
    out
        << "{"
        << "\"valid\":" << (valid ? "true" : "false") << ","
        << "\"status\":\"" << status << "\","
        << "\"source\":\"orb_slam3_cpp\","
        << "\"timestamp_s\":" << timestamp_s << ","
        << "\"pos_neu\":[" << p.x() << "," << p.y() << "," << p.z() << "],"
        << "\"vel_neu\":[0.0,0.0,0.0],"
        << "\"quat_xyzw\":[" << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "],"
        << "\"tracked_features\":" << tracked_features << ","
        << "\"inlier_features\":" << tracked_features << ","
        << "\"outlier_features\":0,"
        << "\"keyframes\":" << diagnostics.keyframes << ","
        << "\"map_points\":" << diagnostics.map_points << ","
        << "\"map_count\":" << diagnostics.map_count << ","
        << "\"lived_keyframes\":" << diagnostics.lived_keyframes << ","
        << "\"lived_map_points\":" << diagnostics.lived_map_points << ","
        << "\"request_count\":" << request_count << ","
        << "\"imu_initialized\":" << json_bool(diagnostics.imu_initialized) << ","
        << "\"inertial_ba1\":" << json_bool(diagnostics.inertial_ba1) << ","
        << "\"inertial_ba2\":" << json_bool(diagnostics.inertial_ba2) << ","
        << "\"bad_imu\":" << json_bool(diagnostics.bad_imu) << ","
        << "\"local_mapper_initializing\":" << json_bool(diagnostics.local_mapper_initializing) << ","
        << "\"keyframes_in_queue\":" << diagnostics.keyframes_in_queue << ","
        << "\"local_mapper_init_time_s\":" << diagnostics.local_mapper_init_time_s << ","
        << "\"local_mapper_matches_inliers\":" << diagnostics.local_mapper_matches_inliers << ","
        << "\"tracker_matches_inliers\":" << diagnostics.tracker_matches_inliers << ","
        << "\"current_frame_id\":" << diagnostics.current_frame_id << ","
        << "\"initial_frame_id\":" << diagnostics.initial_frame_id << ","
        << "\"current_frame_keypoints\":" << diagnostics.current_frame_keypoints << ","
        << "\"initial_frame_keypoints\":" << diagnostics.initial_frame_keypoints << ","
        << "\"mono_ready_to_initialize\":" << json_bool(diagnostics.mono_ready_to_initialize) << ","
        << "\"created_map\":" << json_bool(diagnostics.created_map) << ","
        << "\"max_frames\":" << diagnostics.max_frames << ","
        << "\"frames_to_reset_imu\":" << diagnostics.frames_to_reset_imu << ","
        << "\"initializer_reason\":\"" << json_escape(diagnostics.initializer_reason) << "\","
        << "\"initializer_matches\":" << diagnostics.initializer_matches << ","
        << "\"initializer_triangulated\":" << diagnostics.initializer_triangulated << ","
        << "\"initializer_tracked_map_points\":" << diagnostics.initializer_tracked_map_points << ","
        << "\"initializer_median_depth\":" << diagnostics.initializer_median_depth << ","
        << "\"initializer_elapsed_s\":" << diagnostics.initializer_elapsed_s << ","
        << "\"initializer_reconstruct_success\":" << json_bool(diagnostics.initializer_reconstruct_success) << ","
        << "\"tracking_state\":" << tracking_state << ","
        << "\"tracking_state_name\":\"" << status << "\","
        << "\"failure_reason\":\"" << failure_reason << "\","
        << "\"reset_counter\":0"
        << "}\n";
    return out.str();
}

void handle_client(int client_fd, ORB_SLAM3::System &slam) {
    int request_count = 0;
    while (!g_should_stop) {
        std::optional<std::string> request;
        try {
            request = read_json_line(client_fd);
        } catch (const std::exception &exc) {
            std::cerr << "client read failed: " << exc.what() << std::endl;
            return;
        }
        if (!request.has_value()) {
            return;
        }
        ++request_count;

        try {
            cv::Mat image = decode_image_from_request(*request);
            std::vector<ORB_SLAM3::IMU::Point> imu = imu_points_from_request(*request);
            std::optional<ImuRequestDiagnostics> imu_diag = summarize_imu_request(imu);
            double timestamp_s = frame_timestamp_s_from_request(*request);
            if (auto imu_error = validate_imu_points(imu)) {
                std::cerr
                    << "request " << request_count
                    << " rejected image=" << image.cols << "x" << image.rows
                    << " imu=" << imu.size()
                    << " t=" << timestamp_s
                    << " reason=" << *imu_error
                    << std::endl;
                throw std::runtime_error(*imu_error);
            }

            Sophus::SE3f t_cw = slam.TrackMonocular(image, timestamp_s, imu);
            int tracking_state = slam.GetTrackingState();
            int tracked_features = static_cast<int>(slam.GetTrackedKeyPointsUn().size());
            ORB_SLAM3::OrbSlam3Diagnostics slam_diag = slam.GetDiagnostics();

            std::cout
                << "request " << request_count
                << " image=" << image.cols << "x" << image.rows
                << " imu=" << imu.size()
                << " t=" << timestamp_s
                << " tracking_state=" << tracking_state
                << " tracked_features=" << tracked_features
                << " kf=" << slam_diag.keyframes
                << " mp=" << slam_diag.map_points
                << " maps=" << slam_diag.map_count
                << " imu_init=" << (slam_diag.imu_initialized ? 1 : 0)
                << " ba1=" << (slam_diag.inertial_ba1 ? 1 : 0)
                << " ba2=" << (slam_diag.inertial_ba2 ? 1 : 0)
                << " bad_imu=" << (slam_diag.bad_imu ? 1 : 0)
                << " lm_init=" << (slam_diag.local_mapper_initializing ? 1 : 0)
                << " kf_queue=" << slam_diag.keyframes_in_queue
                << " init_t=" << slam_diag.local_mapper_init_time_s
                << " lm_inliers=" << slam_diag.local_mapper_matches_inliers
                << " trk_inliers=" << slam_diag.tracker_matches_inliers
                << " frame_id=" << slam_diag.current_frame_id
                << " init_frame_id=" << slam_diag.initial_frame_id
                << " frame_kp=" << slam_diag.current_frame_keypoints
                << " init_kp=" << slam_diag.initial_frame_keypoints
                << " ready_init=" << (slam_diag.mono_ready_to_initialize ? 1 : 0)
                << " created_map=" << (slam_diag.created_map ? 1 : 0)
                << " max_frames=" << slam_diag.max_frames
                << " frames_to_reset_imu=" << slam_diag.frames_to_reset_imu
                << " init_reason=" << slam_diag.initializer_reason
                << " init_matches=" << slam_diag.initializer_matches
                << " init_triangulated=" << slam_diag.initializer_triangulated
                << " init_tracked_mp=" << slam_diag.initializer_tracked_map_points
                << " init_median_depth=" << slam_diag.initializer_median_depth
                << " init_elapsed=" << slam_diag.initializer_elapsed_s
                << " init_reconstruct=" << (slam_diag.initializer_reconstruct_success ? 1 : 0);
            if (imu_diag.has_value()) {
                std::cout
                    << " imu_t=[" << imu_diag->first_t << "," << imu_diag->last_t << "]"
                    << " img_minus_imu_last=" << (timestamp_s - imu_diag->last_t)
                    << " imu_dt_mean=" << imu_diag->mean_dt
                    << " imu_dt_max=" << imu_diag->max_dt
                    << " accel_mean=" << format_vec3(imu_diag->accel_mean)
                    << " accel_norm_mean=" << imu_diag->accel_norm_mean
                    << " accel_dyn_mean=" << imu_diag->accel_dynamic_mean
                    << " gyro_mean=" << format_vec3(imu_diag->gyro_mean)
                    << " gyro_norm_max=" << imu_diag->gyro_norm_max;
            }
            std::cout << std::endl;

            if (!send_all(
                    client_fd,
                    tracking_response(
                        t_cw,
                        tracking_state,
                        tracked_features,
                        request_count,
                        timestamp_s,
                        slam_diag
                    )
                )) {
                std::cerr << "send failed: " << std::strerror(errno) << std::endl;
                return;
            }
        } catch (const std::exception &exc) {
            std::cerr << "request failed: " << exc.what() << std::endl;
            if (!send_all(client_fd, error_response(exc.what()))) {
                return;
            }
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        ServerConfig config = parse_args(argc, argv);
        std::cout << "loading ORB-SLAM3 vocabulary/settings..." << std::endl;
        print_settings_diagnostics(config.settings_path);
        ORB_SLAM3::System slam(
            config.vocabulary_path,
            config.settings_path,
            ORB_SLAM3::System::IMU_MONOCULAR,
            config.use_viewer
        );

        int listen_fd = create_listen_socket(config);
        std::cout
            << "orbslam3 VIO server listening on "
            << config.host << ":" << config.port
            << std::endl;

        while (!g_should_stop) {
            sockaddr_in client_addr {};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = ::accept(
                listen_fd,
                reinterpret_cast<sockaddr *>(&client_addr),
                &client_len
            );
            if (client_fd < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(
                    std::string("accept failed: ") + std::strerror(errno)
                );
            }

            char addr_buf[INET_ADDRSTRLEN] = {};
            ::inet_ntop(
                AF_INET,
                &client_addr.sin_addr,
                addr_buf,
                sizeof(addr_buf)
            );
            std::cout
                << "client connected from "
                << addr_buf << ":" << ntohs(client_addr.sin_port)
                << std::endl;

            handle_client(client_fd, slam);
            close_fd(client_fd);
            std::cout << "client disconnected" << std::endl;
        }

        slam.Shutdown();
        close_fd(listen_fd);
        return 0;
    } catch (const std::exception &exc) {
        std::cerr << "error: " << exc.what() << std::endl;
        return 1;
    }
}
