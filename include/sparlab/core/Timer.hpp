/// \file Timer.hpp
/// \brief Monotonic wall-clock stopwatch used for the runtime scaling study.
#pragma once

#include <chrono>
#include <map>
#include <string>

namespace sparlab {

/// Simple RAII-free stopwatch. `elapsed_seconds()` may be called repeatedly.
class Timer {
 public:
  Timer() { reset(); }

  void reset() { start_ = Clock::now(); }

  double elapsed_seconds() const {
    return std::chrono::duration<double>(Clock::now() - start_).count();
  }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

/// Accumulates named wall-clock totals so that per-phase cost (assembly,
/// factorisation, solve, sensitivity) can be reported in run summaries.
class TimingLedger {
 public:
  void add(const std::string& name, double seconds) { totals_[name] += seconds; }

  double get(const std::string& name) const {
    auto it = totals_.find(name);
    return it == totals_.end() ? 0.0 : it->second;
  }

  const std::map<std::string, double>& totals() const { return totals_; }

 private:
  std::map<std::string, double> totals_;
};

/// Adds the lifetime of the scope to a ledger entry.
class ScopedTimer {
 public:
  ScopedTimer(TimingLedger& ledger, std::string name)
      : ledger_(ledger), name_(std::move(name)) {}

  ~ScopedTimer() { ledger_.add(name_, timer_.elapsed_seconds()); }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  TimingLedger& ledger_;
  std::string name_;
  Timer timer_;
};

}  // namespace sparlab
