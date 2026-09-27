#pragma once

#include <atomic>
#include <cstdint>

namespace cps_controllers
{

// At most one request awaits a result/acceptance in a control-path episode.
// Finishing computation is NOT completion: accepting that result can replace
// the source plan. A new request must snapshot the plan after that acceptance.
// All methods except discardedByWorker() are called by the control thread.
class AsyncRequestGate
{
public:
  bool canPublish() const
  {
    return pending_sequence_ == 0 ||
           discarded_sequence_.load(std::memory_order_acquire) >= pending_sequence_;
  }

  void published(std::uint64_t sequence) { pending_sequence_ = sequence; }

  void consumed(std::uint64_t sequence)
  {
    // A late result from a canceled path must not release its replacement.
    if (sequence == pending_sequence_) {
      pending_sequence_ = 0;
    }
  }

  // Only the single monitor worker writes this counter, in increasing input
  // order. Input sequences stay monotonic across path changes.
  void discardedByWorker(std::uint64_t sequence)
  {
    discarded_sequence_.store(sequence, std::memory_order_release);
  }

  void resetControlPath() { pending_sequence_ = 0; }

  // Both threads must be stopped before input sequence numbering is reset.
  void resetStopped()
  {
    pending_sequence_ = 0;
    discarded_sequence_.store(0, std::memory_order_relaxed);
  }

  std::uint64_t pendingSequence() const { return pending_sequence_; }

private:
  std::uint64_t pending_sequence_{0};
  std::atomic<std::uint64_t> discarded_sequence_{0};
};

}  // namespace cps_controllers
