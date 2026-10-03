#include "fixed_output_scheduler.hpp"
#include <gtest/gtest.h>
#include <limits>

TEST(FixedOutputScheduler, FreshMeasurementAfterSlotNeverStarvesOutput) {
    double next = 100.0, target = 0.0;
    for (int i=0; i<30; ++i) {
        const double slot = 100.0 + i/30.0;
        const double measurement = slot + .003;
        const double now = slot + .005;
        EXPECT_FALSE(tracking_output::valid_prediction_age(slot-measurement,.5));
        ASSERT_TRUE(tracking_output::claim_slot(now,30,next,target));
        EXPECT_DOUBLE_EQ(target,now);
        EXPECT_TRUE(tracking_output::valid_prediction_age(target-measurement,.5));
        EXPECT_DOUBLE_EQ(measurement,slot+.003);  // no observation retimestamping
    }
}
TEST(FixedOutputScheduler, StaleAndFutureStatesRemainRejected) {
    EXPECT_TRUE(tracking_output::valid_prediction_age(.5,.5));
    EXPECT_FALSE(tracking_output::valid_prediction_age(.50001,.5));
    EXPECT_FALSE(tracking_output::valid_prediction_age(-.001,.5));
    EXPECT_FALSE(tracking_output::valid_prediction_age(std::numeric_limits<double>::quiet_NaN(),.5));
}
TEST(FixedOutputScheduler, DuplicateCallsAndDelayedLoopsDoNotBurst) {
    double next=0,target=0;
    ASSERT_TRUE(tracking_output::claim_slot(100,30,next,target));
    EXPECT_FALSE(tracking_output::claim_slot(100,30,next,target));
    EXPECT_FALSE(tracking_output::claim_slot(100.01,30,next,target));
    EXPECT_TRUE(tracking_output::claim_slot(100.51,30,next,target));
    EXPECT_DOUBLE_EQ(target,100.51);
    EXPECT_GT(next,100.51);
    EXPECT_FALSE(tracking_output::claim_slot(100.51,30,next,target));
}
TEST(FixedOutputScheduler, BackwardClockResetsCadenceNotFreshness) {
    double next=200,target=0;
    EXPECT_TRUE(tracking_output::claim_slot(100,30,next,target));
    EXPECT_DOUBLE_EQ(target,100);
    EXPECT_GT(next,100);
    EXPECT_FALSE(tracking_output::valid_prediction_age(target-200,.5));
    EXPECT_FALSE(tracking_output::claim_slot(100,0,next,target));
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
