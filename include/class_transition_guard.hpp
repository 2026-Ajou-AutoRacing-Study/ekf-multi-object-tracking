#ifndef CLASS_TRANSITION_GUARD_HPP_
#define CLASS_TRANSITION_GUARD_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

// Independent of ROS and the tracker header so State can be a TrackStruct member.
// Class IDs follow mc_mot::ObjectClass: UNKNOWN=0, CAR=1, TRUCK=2.
namespace class_transition_guard {

using Dimensions = std::array<double, 3>;  // length, width, height in metres
constexpr std::size_t kMaximumPendingVotes = 32U;

struct Config {
    bool enabled{false};
    unsigned int required_consecutive{3U};
    double maximum_gap_sec{0.5};
    double shape_ratio{1.8};
    double minimum_dimension_m{0.25};
};

enum class Phase { Idle, Pending, Released };

struct State {
    Phase phase{Phase::Idle};
    int source_family{-1};
    int candidate_family{-1};
    Dimensions candidate_dimensions{{0.0, 0.0, 0.0}};
    double last_stamp{0.0};
    std::array<int, kMaximumPendingVotes> class_votes{};
    std::size_t vote_count{0U};

    // Also call on a track-slot reset and an unassociated input frame. A later
    // reacquisition must not inherit evidence collected before disconnection.
    void Reset() {
        phase = Phase::Idle;
        source_family = -1;
        candidate_family = -1;
        candidate_dimensions = {{0.0, 0.0, 0.0}};
        last_stamp = 0.0;
        class_votes.fill(0);
        vote_count = 0U;
    }
};

struct Decision {
    // Replay these labels, in this order, through the original class EMA. No
    // confidence weighting, probability overwrite or permanent class lock.
    std::array<int, kMaximumPendingVotes> class_votes{};
    std::size_t vote_count{0U};
    bool hold_shape{false};
    bool pending{false};
    bool released{false};
};

inline int Family(const int classification) {
    if (classification == 0) return 0;
    if (classification == 1 || classification == 2) return 1;
    return -1;
}

inline bool ValidConfig(const Config& config) {
    return config.required_consecutive >= 2U &&
        config.required_consecutive <= kMaximumPendingVotes &&
        std::isfinite(config.maximum_gap_sec) && config.maximum_gap_sec > 0.0 &&
        std::isfinite(config.shape_ratio) && config.shape_ratio > 1.0 &&
        std::isfinite(config.minimum_dimension_m) && config.minimum_dimension_m > 0.0;
}

inline bool ValidDimensions(const Dimensions& dimensions, const double minimum) {
    for (const double value : dimensions) {
        if (!std::isfinite(value) || value < minimum) return false;
    }
    return true;
}

// Sort footprint axes to avoid treating a length/width convention swap as a
// physical discontinuity. A large change along any valid axis is sufficient;
// this is not an IoU, volume, position-innovation or association gate.
inline double ShapeRatio(const Dimensions& first, const Dimensions& second) {
    const Dimensions first_sorted{{std::max(first[0], first[1]),
                                   std::min(first[0], first[1]), first[2]}};
    const Dimensions second_sorted{{std::max(second[0], second[1]),
                                    std::min(second[0], second[1]), second[2]}};
    double ratio = 1.0;
    for (std::size_t axis = 0U; axis < first_sorted.size(); ++axis) {
        ratio = std::max(ratio, std::max(first_sorted[axis], second_sorted[axis]) /
                                 std::min(first_sorted[axis], second_sorted[axis]));
    }
    return ratio;
}

inline Decision PassThrough(const int observed_class) {
    Decision decision;
    decision.class_votes[0] = observed_class;
    decision.vote_count = 1U;
    return decision;
}

// Call once for each associated measurement, before the dimension/class EMA.
// `mature` is the caller's existing confirmed-track predicate, not a speed gate.
// During pending, retain only the prior dimensions: the EKF position and all
// other ordinary measurement updates must still run. On release, apply the
// ordinary dimension EMA once and replay class_votes through the original EMA.
inline Decision Evaluate(
        const Config& config,
        State& state,
        const bool mature,
        const int representative_class,
        const Dimensions& track_dimensions,
        const int observed_class,
        const Dimensions& observed_dimensions,
        const double stamp) {
    const Decision passthrough = PassThrough(observed_class);
    const int source_family = Family(representative_class);
    const int candidate_family = Family(observed_class);
    const bool guarded_families = source_family >= 0 && candidate_family >= 0 &&
                                  source_family != candidate_family;

    // Invalid or tiny dimensions have no trustworthy relative-shape evidence.
    // Fall back to the unchanged EMA rather than permanently blocking recovery.
    if (!config.enabled || !ValidConfig(config) || !mature ||
        !std::isfinite(stamp) || !guarded_families ||
        !ValidDimensions(track_dimensions, config.minimum_dimension_m) ||
        !ValidDimensions(observed_dimensions, config.minimum_dimension_m)) {
        state.Reset();
        return passthrough;
    }

    if (state.phase != Phase::Idle) {
        const double gap = stamp - state.last_stamp;
        const bool invalid_history = !std::isfinite(gap) || gap <= 0.0 ||
            gap > config.maximum_gap_sec || state.source_family != source_family ||
            state.candidate_family != candidate_family ||
            !ValidDimensions(state.candidate_dimensions, config.minimum_dimension_m) ||
            ShapeRatio(state.candidate_dimensions, observed_dimensions) >= config.shape_ratio;
        if (invalid_history) state.Reset();
    }

    // Three alpha=.2 votes from a certain UNKNOWN leave CAR at .488. Keep an
    // admitted family/shape eligible while the unchanged EMA catches up; do not
    // start another confirmation cycle only because the representative lags.
    if (state.phase == Phase::Released) {
        state.last_stamp = stamp;
        return passthrough;
    }

    if (state.phase == Phase::Idle) {
        if (ShapeRatio(track_dimensions, observed_dimensions) < config.shape_ratio) {
            return passthrough;
        }
        state.phase = Phase::Pending;
        state.source_family = source_family;
        state.candidate_family = candidate_family;
        state.candidate_dimensions = observed_dimensions;
        state.vote_count = 0U;
    }

    // State is normally internal, but defensively reject corrupt or incompatible
    // counts rather than indexing beyond the bounded evidence buffer.
    if (state.vote_count >= kMaximumPendingVotes ||
        state.vote_count >= config.required_consecutive) {
        state.Reset();
        return passthrough;
    }
    state.class_votes[state.vote_count++] = observed_class;
    state.last_stamp = stamp;

    Decision decision;
    if (state.vote_count < config.required_consecutive) {
        decision.hold_shape = true;
        decision.pending = true;
        return decision;
    }

    decision.class_votes = state.class_votes;
    decision.vote_count = state.vote_count;
    decision.released = true;
    state.phase = Phase::Released;
    // The bounded history is no longer needed. Family/shape and timestamp alone
    // maintain continuity until the representative changes or evidence resets.
    state.class_votes.fill(0);
    state.vote_count = 0U;
    return decision;
}

}  // namespace class_transition_guard

#endif  // CLASS_TRANSITION_GUARD_HPP_
