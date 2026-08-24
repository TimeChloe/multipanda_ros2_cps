#include <atomic>
#include <cstdint>
#include <thread>

#include <gtest/gtest.h>

#include "cps_controllers/latest_value_mailbox.hpp"

namespace cps_controllers
{
namespace
{

struct TestValue
{
  std::uint64_t sequence{0};
};

TEST(LatestValueMailbox, ConsumerTakesNewestAndDiscardsOlderReadyValues) {
  LatestValueMailbox<TestValue, 3> mailbox;

  EXPECT_TRUE(mailbox.publish(TestValue{1}).published);
  EXPECT_TRUE(mailbox.publish(TestValue{2}).published);
  EXPECT_TRUE(mailbox.publish(TestValue{3}).published);

  TestValue output;
  const auto take_result = mailbox.takeLatest(&output);
  EXPECT_TRUE(take_result.taken);
  EXPECT_EQ(output.sequence, 3U);
  EXPECT_EQ(take_result.discarded_older, 2U);
  EXPECT_FALSE(mailbox.takeLatest(&output).taken);
}

TEST(LatestValueMailbox, ProducerReplacesOldestReadyValueWhenSlotsAreFull) {
  LatestValueMailbox<TestValue, 3> mailbox;

  EXPECT_TRUE(mailbox.publish(TestValue{1}).published);
  EXPECT_TRUE(mailbox.publish(TestValue{2}).published);
  EXPECT_TRUE(mailbox.publish(TestValue{3}).published);
  const auto publish_result = mailbox.publish(TestValue{4});
  EXPECT_TRUE(publish_result.published);
  EXPECT_EQ(publish_result.overwritten_ready, 1U);

  TestValue output;
  const auto take_result = mailbox.takeLatest(&output);
  EXPECT_TRUE(take_result.taken);
  EXPECT_EQ(output.sequence, 4U);
  EXPECT_EQ(take_result.discarded_older, 2U);
}

TEST(LatestValueMailbox, ConcurrentProducerAndConsumerRemainMonotonic) {
  constexpr std::uint64_t kMessages = 20000;
  LatestValueMailbox<TestValue, 3> mailbox;
  std::atomic<bool> producer_finished{false};
  std::atomic<std::uint64_t> published{0};
  std::atomic<std::uint64_t> overwritten{0};

  std::thread producer([&]() {
      for (std::uint64_t sequence = 1; sequence <= kMessages; ++sequence) {
        while (true) {
          const auto result = mailbox.publish(TestValue{sequence});
          if (result.published) {
            published.fetch_add(1, std::memory_order_relaxed);
            overwritten.fetch_add(
              result.overwritten_ready, std::memory_order_relaxed);
            break;
          }
          std::this_thread::yield();
        }
      }
      producer_finished.store(true, std::memory_order_release);
    });

  std::uint64_t last_sequence = 0;
  std::uint64_t consumed = 0;
  while (!producer_finished.load(std::memory_order_acquire) ||
    last_sequence < kMessages)
  {
    TestValue output;
    const auto result = mailbox.takeLatest(&output);
    if (!result.taken) {
      std::this_thread::yield();
      continue;
    }
    EXPECT_GT(output.sequence, last_sequence);
    last_sequence = output.sequence;
    ++consumed;
    overwritten.fetch_add(result.discarded_older, std::memory_order_relaxed);
  }

  producer.join();
  EXPECT_EQ(last_sequence, kMessages);
  EXPECT_EQ(published.load(std::memory_order_relaxed), kMessages);
  EXPECT_EQ(consumed + overwritten.load(std::memory_order_relaxed), kMessages);
}

}  // namespace
}  // namespace cps_controllers
