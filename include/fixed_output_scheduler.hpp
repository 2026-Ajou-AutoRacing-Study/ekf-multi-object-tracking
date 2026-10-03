#pragma once

#include <algorithm>
#include <cmath>

namespace tracking_output {
// The nominal slot owns cadence, not the prediction timestamp. A measurement
// can legitimately be newer than a missed slot while still older than now.
inline bool claim_slot(const double now, const double rate, double & next_slot,
                       double & prediction_time) {
    if (!std::isfinite(now) || !std::isfinite(rate) || rate <= 0.0) return false;
    const double period = 1.0 / rate;
    if (next_slot <= 0.0 || now + period < next_slot) next_slot = now;
    if (now + 1.0e-9 < next_slot) return false;
    const double missed = std::floor((now - next_slot) / period);
    next_slot += (std::max(0.0, missed) + 1.0) * period;
    prediction_time = now;
    return true;
}

inline bool valid_prediction_age(const double age, const double maximum_age) {
    return std::isfinite(age) && std::isfinite(maximum_age) &&
           age >= -1.0e-6 && age <= maximum_age;
}
}  // namespace tracking_output
