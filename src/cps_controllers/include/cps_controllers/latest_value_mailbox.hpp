#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace cps_controllers
{

// A bounded single-producer/single-consumer mailbox for non-trivial values.
// The producer and consumer never access the same value concurrently: slot
// ownership is transferred with atomic state transitions.  When the consumer
// falls behind, the newest completed value is retained instead of building a
// FIFO backlog of stale monitor results.
template<typename T, std::size_t Capacity = 3>
class LatestValueMailbox
{
  static_assert(Capacity >= 2, "LatestValueMailbox needs at least two slots");

public:
  struct PublishResult
  {
    bool published{false};
    std::size_t overwritten_ready{0};
  };

  struct TakeResult
  {
    bool taken{false};
    std::size_t discarded_older{0};
  };

  LatestValueMailbox() = default;
  LatestValueMailbox(const LatestValueMailbox &) = delete;
  LatestValueMailbox & operator=(const LatestValueMailbox &) = delete;

  // Called only by the single producer.
  PublishResult publish(T value)
  {
    std::size_t slot_index = Capacity;
    std::size_t overwritten_ready = 0;

    // Prefer a free slot so a ready result remains available to the consumer.
    for (std::size_t i = 0; i < Capacity; ++i) {
      SlotState expected = SlotState::kFree;
      if (slots_[i].state.compare_exchange_strong(
          expected,
          SlotState::kWriting,
          std::memory_order_acq_rel,
          std::memory_order_acquire))
      {
        slot_index = i;
        break;
      }
    }

    // If every free slot is occupied, replace the oldest completed result.
    // A slot being read is never reclaimed by the producer.
    if (slot_index == Capacity) {
      std::size_t oldest_index = Capacity;
      std::uint64_t oldest_ticket = 0;
      for (std::size_t i = 0; i < Capacity; ++i) {
        if (slots_[i].state.load(std::memory_order_acquire) !=
          SlotState::kReady)
        {
          continue;
        }
        const std::uint64_t ticket =
          slots_[i].ticket.load(std::memory_order_relaxed);
        if (oldest_index == Capacity || ticket < oldest_ticket) {
          oldest_index = i;
          oldest_ticket = ticket;
        }
      }

      if (oldest_index != Capacity) {
        SlotState expected = SlotState::kReady;
        if (slots_[oldest_index].state.compare_exchange_strong(
            expected,
            SlotState::kWriting,
            std::memory_order_acq_rel,
            std::memory_order_acquire))
        {
          slot_index = oldest_index;
          overwritten_ready = 1;
        }
      }
    }

    // This is possible only during the very short interval in which the
    // consumer owns every non-free slot. Dropping the new result is preferable
    // to blocking either the worker or the real-time controller.
    if (slot_index == Capacity) {
      return PublishResult{};
    }

    Slot & slot = slots_[slot_index];
    slot.value = std::move(value);
    slot.ticket.store(next_ticket_++, std::memory_order_relaxed);
    slot.state.store(SlotState::kReady, std::memory_order_release);
    return PublishResult{true, overwritten_ready};
  }

  // Called only by the single consumer. It takes the newest completed value
  // and frees any older completed values without ever waiting for the writer.
  TakeResult takeLatest(T * output)
  {
    if (output == nullptr) {
      return TakeResult{};
    }

    for (std::size_t attempt = 0; attempt < Capacity; ++attempt) {
      std::size_t newest_index = Capacity;
      std::uint64_t newest_ticket = 0;
      for (std::size_t i = 0; i < Capacity; ++i) {
        if (slots_[i].state.load(std::memory_order_acquire) !=
          SlotState::kReady)
        {
          continue;
        }
        const std::uint64_t ticket =
          slots_[i].ticket.load(std::memory_order_relaxed);
        if (newest_index == Capacity || ticket > newest_ticket) {
          newest_index = i;
          newest_ticket = ticket;
        }
      }

      if (newest_index == Capacity) {
        return TakeResult{};
      }

      SlotState expected = SlotState::kReady;
      if (!slots_[newest_index].state.compare_exchange_strong(
          expected,
          SlotState::kReading,
          std::memory_order_acq_rel,
          std::memory_order_acquire))
      {
        continue;
      }

      // The slot may have been reclaimed and republished between the scan and
      // the state CAS (READY -> WRITING -> READY). Once READING ownership is
      // acquired, reload its actual generation before deciding which other
      // values are older.
      newest_ticket =
        slots_[newest_index].ticket.load(std::memory_order_acquire);
      *output = std::move(slots_[newest_index].value);
      slots_[newest_index].state.store(
        SlotState::kFree, std::memory_order_release);

      std::size_t discarded_older = 0;
      for (std::size_t i = 0; i < Capacity; ++i) {
        if (i == newest_index ||
          slots_[i].state.load(std::memory_order_acquire) !=
          SlotState::kReady ||
          slots_[i].ticket.load(std::memory_order_relaxed) >
          newest_ticket)
        {
          continue;
        }
        expected = SlotState::kReady;
        if (!slots_[i].state.compare_exchange_strong(
            expected,
            SlotState::kReading,
            std::memory_order_acq_rel,
            std::memory_order_acquire))
        {
          continue;
        }

        // Recheck the generation after ownership transfer to close the ABA
        // window described above. Never discard a value published after the
        // one returned in this call.
        const std::uint64_t claimed_ticket =
          slots_[i].ticket.load(std::memory_order_acquire);
        if (claimed_ticket <= newest_ticket) {
          slots_[i].state.store(SlotState::kFree, std::memory_order_release);
          ++discarded_older;
        } else {
          slots_[i].state.store(SlotState::kReady, std::memory_order_release);
        }
      }
      return TakeResult{true, discarded_older};
    }

    return TakeResult{};
  }

  // Safe while the producer is running. A result currently being written is
  // left alone and will be rejected later by the controller's plan-generation
  // check if it belongs to an old goal.
  void discardReady()
  {
    for (auto & slot : slots_) {
      SlotState expected = SlotState::kReady;
      if (slot.state.compare_exchange_strong(
          expected,
          SlotState::kReading,
          std::memory_order_acq_rel,
          std::memory_order_acquire))
      {
        slot.state.store(SlotState::kFree, std::memory_order_release);
      }
    }
  }

  // The caller must ensure that producer and consumer are stopped.
  void resetStopped()
  {
    next_ticket_ = 1;
    for (auto & slot : slots_) {
      slot.value = T{};
      slot.ticket.store(0, std::memory_order_relaxed);
      slot.state.store(SlotState::kFree, std::memory_order_relaxed);
    }
  }

private:
  enum class SlotState : std::uint8_t
  {
    kFree,
    kWriting,
    kReady,
    kReading,
  };

  struct Slot
  {
    std::atomic<SlotState> state{SlotState::kFree};
    std::atomic<std::uint64_t> ticket{0};
    T value{};
  };

  std::array<Slot, Capacity> slots_{};
  std::uint64_t next_ticket_{1};
};

}  // namespace cps_controllers
