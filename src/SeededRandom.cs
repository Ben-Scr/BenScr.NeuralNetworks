namespace BenScr.NeuralNetworks;

// SplitMix64 fixes initialization and shuffle streams independently of System.Random versions.
internal sealed class SeededRandom
{
    private ulong state;

    internal SeededRandom(int seed)
    {
        state = unchecked((ulong)(long)seed);
    }

    internal double NextDouble() => (NextUInt64() >> 11) * (1.0 / 9007199254740992.0);

    internal int NextIndex(int exclusiveMaximum)
    {
        ulong bound = (ulong)exclusiveMaximum;
        ulong threshold = unchecked(0UL - bound) % bound;
        ulong value;
        do
        {
            value = NextUInt64();
        } while (value < threshold);
        return (int)(value % bound);
    }

    private ulong NextUInt64()
    {
        unchecked
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong value = state;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
            return value ^ (value >> 31);
        }
    }
}
