#pragma once

#include <cstdint>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>

#include "algorithm/ekf_multi_object_tracking.hpp"

namespace tracking_diagnostics {

inline std::string Escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else out << c;
    }
    return out.str();
}

inline void Number(std::ostream& out, const double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
}

template <std::size_t N>
inline void Array(std::ostream& out, const std::array<double, N>& values) {
    out << '[';
    for (std::size_t i = 0; i < N; ++i) {
        if (i) out << ',';
        Number(out, values[i]);
    }
    out << ']';
}

inline std::string TrackUuid(const int track_id) {
    std::ostringstream out;
    out << "414a4f550000000000000000" << std::hex << std::setfill('0') << std::setw(8) << static_cast<std::uint32_t>(track_id);
    return out.str();
}

inline const char* SnapshotPhase(const std::string& event) {
    if (event == "unmatched_candidate") return "pre_association_update";
    if (event == "matched") return "post_measurement_update";
    if (event == "birth_suppressed") return "post_matched_updates";
    if (event == "birth") return "post_birth";
    if (event == "delete") return "before_lifecycle_reset";
    return "post_miss_count_update";
}

inline const char* CandidateScope(const std::string& event) {
    if (event == "unmatched_candidate") return "nearest_existing_track_only";
    if (event == "matched") return "chosen_assignment";
    if (event == "birth_suppressed") return "first_duplicate_guard_blocker";
    if (event == "birth") return "unmatched_measurement";
    return "lifecycle_track";
}

inline std::string ToJson(const mc_mot::AssociationDiagnostics& trace,
                          const std::uint64_t input_generation,
                          const std::uint32_t input_sequence,
                          const std::string& world_frame,
                          const double publication_stamp) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << "{\"schema_version\":1,\"event\":\"tracker_association_update\",\"measurement_stamp_sec\":";
    Number(out, trace.measurement_stamp);
    out << ",\"publication_stamp_sec\":";
    Number(out, publication_stamp);
    out << ",\"input_generation\":" << input_generation << ",\"input_seq\":" << input_sequence
        << ",\"world_frame\":\"" << Escape(world_frame) << "\",\"input_count\":" << trace.input_count
        << ",\"processed_count\":" << trace.processed_count
        << ",\"omitted_rows\":" << trace.omitted_rows
        << ",\"truncated\":" << (trace.omitted_rows ? "true" : "false")
        << ",\"collection_scope\":\"bounded_chosen_and_nearest_only_not_exhaustive\",\"rows\":[";
    for (std::size_t i = 0; i < trace.rows.size(); ++i) {
        const auto& row = trace.rows[i];
        if (i) out << ',';
        out << "{\"event\":\"" << Escape(row.event) << "\",\"reason\":\"" << Escape(row.reason)
            << "\",\"snapshot_phase\":\"" << SnapshotPhase(row.event)
            << "\",\"candidate_scope\":\"" << CandidateScope(row.event)
            << "\",\"assignment_index_phase\":\"post_assignment\",\"measurement_index\":" << row.measurement_index << ",\"track_id\":" << row.track_id
            << ",\"track_uuid\":";
        if (row.track_id >= 0) out << '"' << TrackUuid(row.track_id) << '"';
        else out << "null";
        out << ",\"track_assigned_measurement_index\":" << row.track_assigned_measurement_index
            << ",\"measurement_class\":" << row.measurement_class << ",\"track_class\":" << row.track_class
            << ",\"confirmed\":" << (row.confirmed ? "true" : "false")
            << ",\"associated\":" << (row.associated ? "true" : "false")
            << ",\"detection_count\":" << row.detection_count << ",\"distance_m\":";
        Number(out, row.distance_m);
        out << ",\"association_cost\":"; Number(out, row.association_cost);
        out << ",\"ranking_cost\":"; Number(out, row.ranking_cost);
        out << ",\"prior_measurement_age_sec\":"; Number(out, row.prior_measurement_age_sec);
        out << ",\"last_measurement_stamp_sec\":"; Number(out, row.last_measurement_time);
        out << ",\"last_measurement_age_sec\":"; Number(out, trace.measurement_stamp-row.last_measurement_time);
        out << ",\"initialization_stamp_sec\":"; Number(out, row.initialization_time);
        out << ",\"track_score\":"; Number(out, row.score);
        out << ",\"measurement_score\":"; Number(out, row.measurement_score);
        out << ",\"center_xyz\":"; Array(out, row.center);
        out << ",\"dimensions_lwh\":"; Array(out, row.dimensions);
        out << ",\"velocity_xy\":"; Array(out, row.velocity);
        out << ",\"class_votes\":"; Array(out, row.class_votes);
        out << '}';
    }
    out << "]}";
    return out.str();
}

}  // namespace tracking_diagnostics
