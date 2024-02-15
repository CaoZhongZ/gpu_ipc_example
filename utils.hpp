#pragma once

template <typename F>
class __scope_guard {
  F f;
public:
  __scope_guard(const F &f) : f(f) {}
  ~__scope_guard() { f(); }
};

template <typename T>
void fill_pattern(T *input, int rank, size_t n) {
  for (int i = 0; i < n; ++ i)
    input[i] = (T)((i % 32) * rank);
}
