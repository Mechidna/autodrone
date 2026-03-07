def clamp(x, lo, hi):
    return max(lo, min(hi, x))

if __name__ == "__main__":
    x = -1
    lo = 0
    hi = 1
    print(clamp(x, lo, hi))