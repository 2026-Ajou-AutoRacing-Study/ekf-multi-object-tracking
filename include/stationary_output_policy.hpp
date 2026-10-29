#pragma once

#include <cmath>

namespace stationary_output_policy {
// Match avoidance ownership boundaries; no timer or sample-count debounce.
inline bool UpdateDynamic(bool previous, double speed, double enter, double exit) {
  if (!std::isfinite(speed) || speed >= enter) return true;
  if (speed <= exit) return false;
  return previous;
}
}  // namespace stationary_output_policy
