#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <functional>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cps_controllers {

// Single-producer/single-consumer bounded file writer. The producer only
// copies into a preallocated slot and advances an atomic index; all formatting,
// filesystem access, buffering, and flushing happen on the worker thread.
// A full queue drops the newest record instead of delaying the producer.
template <typename Record>
class BoundedAsyncFileWriter {
 public:
  using RecordWriter = std::function<void(std::ostream&, const Record&)>;
  using SlotPreparer = std::function<void(Record&)>;

  BoundedAsyncFileWriter() = default;
  ~BoundedAsyncFileWriter() { stop(); }

  BoundedAsyncFileWriter(const BoundedAsyncFileWriter&) = delete;
  BoundedAsyncFileWriter& operator=(const BoundedAsyncFileWriter&) = delete;

  bool start(const std::string& file_path,
             const std::string& header,
             std::size_t queue_capacity,
             std::size_t batch_size,
             double flush_period_sec,
             RecordWriter record_writer,
             SlotPreparer slot_preparer = SlotPreparer{}) {
    stop();

    const std::size_t usable_capacity = std::max<std::size_t>(1, queue_capacity);
    slots_.clear();
    slots_.resize(usable_capacity + 1);
    if (slot_preparer) {
      for (auto& slot : slots_) {
        slot_preparer(slot);
      }
    }

    stream_buffer_.assign(1024 * 1024, '\0');
    file_.clear();
    file_.rdbuf()->pubsetbuf(stream_buffer_.data(),
                            static_cast<std::streamsize>(stream_buffer_.size()));
    file_.open(file_path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
      slots_.clear();
      stream_buffer_.clear();
      return false;
    }

    file_ << header;
    if (header.empty() || header.back() != '\n') {
      file_ << '\n';
    }

    record_writer_ = std::move(record_writer);
    batch_size_ = std::max<std::size_t>(1, batch_size);
    flush_period_ = std::chrono::duration<double>(
        std::max(0.05, flush_period_sec));
    write_index_.store(0, std::memory_order_relaxed);
    read_index_.store(0, std::memory_order_relaxed);
    enqueued_count_.store(0, std::memory_order_relaxed);
    written_count_.store(0, std::memory_order_relaxed);
    dropped_count_.store(0, std::memory_order_relaxed);
    running_.store(true, std::memory_order_release);
    worker_thread_ = std::thread(&BoundedAsyncFileWriter::workerLoop, this);
    return true;
  }

  template <typename FillSlot>
  bool tryEmplace(FillSlot&& fill_slot) {
    if (!running_.load(std::memory_order_acquire) || slots_.empty()) {
      return false;
    }

    const std::size_t write = write_index_.load(std::memory_order_relaxed);
    const std::size_t next = increment(write);
    if (next == read_index_.load(std::memory_order_acquire)) {
      dropped_count_.fetch_add(1, std::memory_order_relaxed);
      return false;
    }

    fill_slot(slots_[write]);
    write_index_.store(next, std::memory_order_release);
    enqueued_count_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  void stop() {
    running_.store(false, std::memory_order_release);
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    if (file_.is_open()) {
      file_.flush();
      file_.close();
    }
    file_.clear();
    record_writer_ = RecordWriter{};
  }

  bool running() const {
    return running_.load(std::memory_order_acquire);
  }

  std::size_t queueDepth() const {
    if (slots_.empty()) {
      return 0;
    }
    const std::size_t write = write_index_.load(std::memory_order_acquire);
    const std::size_t read = read_index_.load(std::memory_order_acquire);
    return write >= read ? write - read : slots_.size() - read + write;
  }

  std::size_t enqueuedCount() const {
    return enqueued_count_.load(std::memory_order_relaxed);
  }

  std::size_t writtenCount() const {
    return written_count_.load(std::memory_order_relaxed);
  }

  std::size_t droppedCount() const {
    return dropped_count_.load(std::memory_order_relaxed);
  }

 private:
  std::size_t increment(std::size_t index) const {
    ++index;
    return index == slots_.size() ? 0 : index;
  }

  void workerLoop() {
    auto last_flush = std::chrono::steady_clock::now();
    while (running_.load(std::memory_order_acquire) || queueDepth() > 0) {
      std::size_t batch_count = 0;
      while (batch_count < batch_size_) {
        const std::size_t read = read_index_.load(std::memory_order_relaxed);
        if (read == write_index_.load(std::memory_order_acquire)) {
          break;
        }

        record_writer_(file_, slots_[read]);
        read_index_.store(increment(read), std::memory_order_release);
        written_count_.fetch_add(1, std::memory_order_relaxed);
        ++batch_count;
      }

      const auto now = std::chrono::steady_clock::now();
      if (now - last_flush >= flush_period_) {
        file_.flush();
        last_flush = now;
      }

      if (batch_count == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }

  std::vector<Record> slots_;
  std::vector<char> stream_buffer_;
  std::ofstream file_;
  RecordWriter record_writer_;
  std::thread worker_thread_;
  std::atomic<bool> running_{false};
  std::atomic<std::size_t> write_index_{0};
  std::atomic<std::size_t> read_index_{0};
  std::atomic<std::size_t> enqueued_count_{0};
  std::atomic<std::size_t> written_count_{0};
  std::atomic<std::size_t> dropped_count_{0};
  std::size_t batch_size_{1};
  std::chrono::duration<double> flush_period_{1.0};
};

}  // namespace cps_controllers
