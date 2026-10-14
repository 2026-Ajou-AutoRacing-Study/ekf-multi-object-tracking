#ifndef AJOU_DETECTION_INPUT_RECEIPT_HPP
#define AJOU_DETECTION_INPUT_RECEIPT_HPP

#include <cstdint>
#include <cmath>

namespace detection_input {

inline bool IsNewerStamp(double incoming, double latest, bool has_latest) {
    return std::isfinite(incoming) && (!has_latest || incoming > latest);
}

// All accesses belong under the caller's packet mutex. Completion of an old
// snapshot must never acknowledge a newer callback's latest-value packet.
class Receipt {
public:
    bool HasPending() const { return pending_; }
    std::uint64_t Generation() const { return generation_; }
    void Commit() { ++generation_; pending_ = true; }
    bool Acknowledge(std::uint64_t processed_generation) {
        if (!pending_ || processed_generation != generation_) return false;
        pending_ = false;
        return true;
    }
private:
    std::uint64_t generation_{0};
    bool pending_{false};
};

} // namespace detection_input
#endif
