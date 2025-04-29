public class JavaRandom {
    private static final long MULTIPLIER = 0x5DEECE66DL;
    private static final long ADDEND = 0xB;
    private static final long MASK = (1L << 48) - 1;

    private long seed;
    private boolean debug;

    public JavaRandom(long seed, boolean debug) {
        this.debug = debug;
        this.seed = (seed ^ MULTIPLIER) & MASK;
        if (debug) {
            System.out.println("Initial seed: " + seed);
        }
    }

    public void setSeed(long seed) {
        this.seed = (seed ^ MULTIPLIER) & MASK;
        if (debug) {
            System.out.println("Seed reset to: " + seed);
        }
    }

    public int next(int bits) {
        if (debug) {
            System.out.println("Before: seed = " + seed + " (hex: " + Long.toHexString(seed) + ")");
        }
        seed = (seed * MULTIPLIER + ADDEND) & MASK;
        if (debug) {
            System.out.println("After:  seed = " + seed + " (hex: " + Long.toHexString(seed) + ")");
        }
        long result = (seed >> (48 - bits)) & ((1L << bits) - 1);
        if (debug) {
            System.out.println("Result (bits=" + bits + "): " + result + " (hex: " + Long.toHexString(result) + ")");
        }
        return (int) result;
    }

    public int nextInt(int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("bound must be positive");
        }

        if ((bound & -bound) == bound) {  // bound on 2:n potenssi
            return (int) ((bound * (long) next(31)) >> 31);
        }

        int bits;
        int val;
        while (true) {
            bits = next(31);
            val = bits % bound;
            if (debug) {
                System.out.println("Debug: n=" + bound + ", bits=" + bits + ", val=" + val + ", check=" + (bits - val + (bound - 1)));
            }
            if (bits - val + (bound - 1) >= 0) {
                return val;
            }
        }
    }

    public static void main(String[] args) {
        JavaRandom rand = new JavaRandom(0, true);
        System.out.println("Testing JavaRandom with seed 0:");
        System.out.println("nextInt(1600): " + rand.nextInt(1600) + " Should be 560");
        System.out.println("nextInt(1200): " + rand.nextInt(1200) + " Should be 748");
        System.out.println("nextInt(3): " + rand.nextInt(3) + " Should be 1");
        System.out.println("nextInt(1600): " + rand.nextInt(1600) + " Should be 1247");
        System.out.println("nextInt(1200): " + rand.nextInt(1200) + " Should be 1115");
        System.out.println("nextInt(3): " + rand.nextInt(3) + " Should be 2");
    }
}
