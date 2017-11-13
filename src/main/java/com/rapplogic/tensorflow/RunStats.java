package com.rapplogic.tensorflow;

/** Accumulate and analyze stats from metadata obtained from Session.Runner.run. */
public class RunStats {

    /**
     * Options to be provided to a {@link org.tensorflow.Session.Runner} to enable stats accumulation.
     */
    public static byte[] runOptions() {
        return fullTraceRunOptions;
    }

    public RunStats() {

    }

    // Hack: This is what a serialized RunOptions protocol buffer with trace_level: FULL_TRACE ends
    // up as.
    private static byte[] fullTraceRunOptions = new byte[] {0x08, 0x03};
}

