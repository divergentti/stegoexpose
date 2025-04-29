class JavaRandom:
    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MASK = (1 << 48) - 1

    def __init__(self, seed=None, debug=False):
        self.debug = debug
        if seed is None:
            import time
            seed = int(time.time() * 1000)
        self.seed = (seed ^ self.MULTIPLIER) & self.MASK
        if self.debug:
            print(f"Initial seed: {self.seed}")

    def set_seed(self, seed):
        self.seed = (seed ^ self.MULTIPLIER) & self.MASK
        if self.debug:
            print(f"Seed reset to: {self.seed}")

    def next(self, bits):
        if self.debug:
            print(f"Before: seed = {self.seed} (hex: {hex(self.seed)})")
        self.seed = (self.seed * self.MULTIPLIER + self.ADDEND) & self.MASK
        if self.debug:
            print(f"After:  seed = {self.seed} (hex: {hex(self.seed)})")
        # Varmista, että palautetaan unsigned arvo
        result = (self.seed >> (48 - bits)) & ((1 << bits) - 1)
        if self.debug:
            print(f"Result (bits={bits}): {result} (hex: {hex(result)})")
        return result

    def next_int(self, bound=None):
        if bound is None:
            return self.next(32)

        if bound <= 0:
            raise ValueError("bound must be positive")

        if (bound & -bound) == bound:  # bound on 2:n potenssi
            return (bound * self.next(31)) >> 31

        bits = self.next(31)
        val = bits % bound
        if self.debug:
            print(f"Debug: n={bound}, bits={bits}, val={val}, check={bits - val + (bound - 1)}")
        while bits - val + (bound - 1) < 0:
            bits = self.next(31)
            val = bits % bound
            if self.debug:
                print(f"Debug: n={bound}, bits={bits}, val={val}, check={bits - val + (bound - 1)}")
        return val

# Testikoodi
if __name__ == "__main__":
    rand = JavaRandom(0, debug=True)
    print("Testing JavaRandom with seed 0:")
    print(f"nextInt(1600): {rand.next_int(1600)} Should be 560")
    print(f"nextInt(1200): {rand.next_int(1200)} Should be 748")
    print(f"nextInt(3): {rand.next_int(3)} Should be 1")
    print(f"nextInt(1600): {rand.next_int(1600)} Should be 1247")
    print(f"nextInt(1200): {rand.next_int(1200)} Should be 1115")
    print(f"nextInt(3): {rand.next_int(3)} Should be 2")
