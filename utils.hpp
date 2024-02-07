#pragma once

template <typename F>
class __scope_guard {
  F f;
public:
  __scope_guard(const F &f) : f(f) {}
  ~__scope_guard() { f(); }
};
