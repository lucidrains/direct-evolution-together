import nimpy

proc hamming_distance*(a, b: seq[int]): int {.exportpy.} =
  for i in 0 ..< min(a.len, b.len):
    if a[i] != b[i]:
      inc result
