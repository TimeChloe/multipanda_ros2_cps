#include <cstdint>
#include <optional>
#include <thread>

#include <gtest/gtest.h>

#include "cps_controllers/async_request_gate.hpp"
#include "cps_controllers/latest_value_mailbox.hpp"

using cps_controllers::AsyncRequestGate;

TEST(AsyncRequestGate, ResultMustBeConsumedBeforeAnotherRequest)
{
  AsyncRequestGate gate;
  EXPECT_TRUE(gate.canPublish());
  gate.published(1);
  // The worker may be computing, publishing, or have a ready output. All
  // three states must block another request until control finishes handoff.
  EXPECT_FALSE(gate.canPublish());
  gate.consumed(1);
  EXPECT_TRUE(gate.canPublish());
}

TEST(AsyncRequestGate, WorkerDiscardReleasesRequestWithoutAnOutput)
{
  AsyncRequestGate gate;
  gate.published(41);
  std::thread worker([&gate] { gate.discardedByWorker(41); });
  worker.join();
  EXPECT_TRUE(gate.canPublish());
  gate.published(42);
  EXPECT_FALSE(gate.canPublish());
}

TEST(AsyncRequestGate, CanceledPathCompletionCannotReleaseNewRequest)
{
  AsyncRequestGate gate;
  gate.published(5);
  gate.resetControlPath();
  EXPECT_TRUE(gate.canPublish());
  gate.published(6);
  gate.discardedByWorker(5);
  gate.consumed(5);
  EXPECT_FALSE(gate.canPublish());
  EXPECT_EQ(gate.pendingSequence(), 6U);
  gate.consumed(6);
  EXPECT_TRUE(gate.canPublish());
}

TEST(AsyncRequestGate, StoppedResetAllowsRestartedSequenceNumbers)
{
  AsyncRequestGate gate;
  gate.published(50);
  gate.discardedByWorker(50);
  gate.resetStopped();
  gate.published(1);
  EXPECT_FALSE(gate.canPublish());
  gate.consumed(1);
  EXPECT_TRUE(gate.canPublish());
}

TEST(AsyncRequestGate, VariableComputeAndHandoffDelaysKeepSourceGenerationCurrent)
{
  struct Result {
    std::uint64_t sequence{0};
    std::uint64_t source_generation{0};
    int ready_cycle{0};
  };
  AsyncRequestGate gate;
  cps_controllers::LatestValueMailbox<Result> mailbox;
  std::optional<Result> computing;
  std::uint64_t generation = 1, next_sequence = 1;
  int next_due = 0, accepted = 0, rejected_by_monitor = 0, deferred = 0;
  for (int cycle = 0; cycle < 2000; ++cycle) {
    if (computing && cycle >= computing->ready_cycle) {
      ASSERT_TRUE(mailbox.publish(*computing).published);
      computing.reset();
    }
    // Model delayed controller observation as well as variable worker cost.
    Result output;
    if (cycle % 11 >= 3 && mailbox.takeLatest(&output).taken) {
      gate.consumed(output.sequence);
      ASSERT_EQ(output.source_generation, generation) << "cycle=" << cycle;
      if (output.sequence % 7 != 0) {
        ++generation;
        ++accepted;
      } else {
        ++rejected_by_monitor;
      }
    }
    if (cycle < next_due) {
      continue;
    }
    if (!gate.canPublish()) {
      ++deferred;
      continue;
    }
    ASSERT_FALSE(computing.has_value());
    const auto sequence = next_sequence++;
    gate.published(sequence);
    computing = Result{sequence, generation, cycle + 1 + static_cast<int>(sequence % 13)};
    // Same phase as the controller: missed 5-cycle slots are coalesced.
    do {
      next_due += 5;
    } while (next_due <= cycle);
  }
  EXPECT_GT(accepted, 100);
  EXPECT_GT(rejected_by_monitor, 10);
  EXPECT_GT(deferred, 100);
}
