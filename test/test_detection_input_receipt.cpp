#include <gtest/gtest.h>
#include "detection_input_receipt.hpp"
#include <future>
#include <mutex>
#include <thread>
#include <limits>

TEST(DetectionInputReceipt, OlderOrDuplicateCallbackCannotOverwriteNewerPacket) {
    EXPECT_TRUE(detection_input::IsNewerStamp(0.0, 0.0, false));
    EXPECT_TRUE(detection_input::IsNewerStamp(2.0, 1.0, true));
    EXPECT_FALSE(detection_input::IsNewerStamp(1.0, 2.0, true));
    EXPECT_FALSE(detection_input::IsNewerStamp(2.0, 2.0, true));
    EXPECT_FALSE(detection_input::IsNewerStamp(std::numeric_limits<double>::quiet_NaN(), 2.0, true));
}

TEST(DetectionInputReceipt, StartsEmptyAndAcknowledgesOnlyCurrentSnapshot) {
    detection_input::Receipt receipt;
    EXPECT_FALSE(receipt.HasPending());
    EXPECT_FALSE(receipt.Acknowledge(0));
    receipt.Commit();
    const auto token = receipt.Generation();
    EXPECT_TRUE(receipt.HasPending());
    EXPECT_TRUE(receipt.Acknowledge(token));
    EXPECT_FALSE(receipt.HasPending());
    EXPECT_FALSE(receipt.Acknowledge(token));
}

TEST(DetectionInputReceipt, OldCompletionCannotClearNewCallback) {
    detection_input::Receipt receipt;
    receipt.Commit();
    const auto old = receipt.Generation();
    receipt.Commit();
    EXPECT_FALSE(receipt.Acknowledge(old));
    EXPECT_TRUE(receipt.HasPending());
    EXPECT_TRUE(receipt.Acknowledge(receipt.Generation()));
}

TEST(DetectionInputReceipt, RepeatedCallbacksRemainLatestValueNotFifo) {
    detection_input::Receipt receipt;
    receipt.Commit();
    const auto old = receipt.Generation();
    for (int i = 0; i < 100; ++i) receipt.Commit();
    EXPECT_FALSE(receipt.Acknowledge(old));
    EXPECT_EQ(receipt.Generation(), old + 100);
    EXPECT_TRUE(receipt.Acknowledge(receipt.Generation()));
}

TEST(DetectionInputReceipt, FailedConversionDoesNotCommitOrErasePendingPacket) {
    detection_input::Receipt receipt;
    receipt.Commit();
    const auto token = receipt.Generation();
    // A failed local candidate conversion never reaches the commit section.
    EXPECT_TRUE(receipt.HasPending());
    EXPECT_EQ(receipt.Generation(), token);
    EXPECT_TRUE(receipt.Acknowledge(token));
}

TEST(DetectionInputReceipt, CallerLockedConcurrentCallbackSurvivesWorkerCompletion) {
    detection_input::Receipt receipt;
    std::mutex mutex;
    std::promise<void> snapshot_taken, callback_committed;
    auto callback_ready = callback_committed.get_future();
    auto snapshot_ready = snapshot_taken.get_future();
    receipt.Commit();
    auto worker = std::async(std::launch::async, [&] {
        std::uint64_t token;
        { std::lock_guard<std::mutex> lock(mutex); token = receipt.Generation(); }
        snapshot_taken.set_value();
        callback_ready.wait();
        std::lock_guard<std::mutex> lock(mutex);
        return receipt.Acknowledge(token);
    });
    snapshot_ready.wait();
    { std::lock_guard<std::mutex> lock(mutex); receipt.Commit(); }
    callback_committed.set_value();
    EXPECT_FALSE(worker.get());
    EXPECT_TRUE(receipt.HasPending());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
