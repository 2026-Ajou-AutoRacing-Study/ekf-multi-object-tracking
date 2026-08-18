#ifndef VEHICLE_ORIENTATION_CANONICALIZER_HPP_
#define VEHICLE_ORIENTATION_CANONICALIZER_HPP_

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace vehicle_orientation {

struct Result {
    double yaw{0.0};
    bool flipped{false};
    double speed_mps{0.0};
    double motion_yaw_error_rad{0.0};
};

inline double NormalizeAngle(const double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

class Canonicalizer {
public:
    Canonicalizer() = default;

    Result Apply(
            const int track_id,
            const double yaw,
            const double velocity_x,
            const double velocity_y,
            const bool is_vehicle,
            const bool enabled,
            const double minimum_speed_mps,
            const double flip_enter_error_rad,
            const double flip_exit_error_rad) {
        const double speed = std::hypot(velocity_x, velocity_y);
        bool flipped = flipped_by_track_[track_id];
        double error = 0.0;

        if (!enabled || !is_vehicle) {
            flipped = false;
            flipped_by_track_[track_id] = false;
        } else if (speed >= minimum_speed_mps) {
            const double motion_yaw = std::atan2(velocity_y, velocity_x);
            error = std::abs(NormalizeAngle(yaw - motion_yaw));
            if (!flipped && error >= flip_enter_error_rad) {
                flipped = true;
            } else if (flipped && error <= flip_exit_error_rad) {
                flipped = false;
            }
            flipped_by_track_[track_id] = flipped;
        }

        return Result{
            NormalizeAngle(yaw + (flipped ? M_PI : 0.0)),
            flipped,
            speed,
            error,
        };
    }

    void RetainOnly(const std::unordered_set<int>& active_track_ids) {
        for (auto iterator = flipped_by_track_.begin();
             iterator != flipped_by_track_.end();) {
            if (active_track_ids.count(iterator->first) == 0U) {
                iterator = flipped_by_track_.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

private:
    std::unordered_map<int, bool> flipped_by_track_;
};

}  // namespace vehicle_orientation

#endif  // VEHICLE_ORIENTATION_CANONICALIZER_HPP_
