#pragma once

#include <cstddef>

namespace tracking_diagnostics {

// Discovery stays cheap: a recorder can subscribe after the tracker starts.
// Detailed trace allocation/serialization still requires a live subscriber.
inline bool ShouldAdvertise(bool explicitly_enabled, bool auto_on_subscribe) {
    return explicitly_enabled || auto_on_subscribe;
}

inline bool ShouldCollect(bool explicitly_enabled, bool auto_on_subscribe,
                          std::size_t subscriber_count) {
    return ShouldAdvertise(explicitly_enabled, auto_on_subscribe) && subscriber_count > 0;
}

}  // namespace tracking_diagnostics
