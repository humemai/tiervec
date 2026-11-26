package ai.humem.tiervec;

import java.util.function.LongConsumer;
import java.util.function.LongSupplier;

/**
 * The push form of the statistics of an index: a host hands one to {@link TieredIndex#bindMetrics} and forwards the numbers to whatever
 * metrics system it has. The library depends on none. The names are the keys of {@link TieredIndex#metrics()} and start with
 * {@code tiervec.}.
 *
 * <p>The library calls a counter on the thread where the event happened, which can be the maintenance worker or a writer, and ignores a
 * {@link RuntimeException} that it throws. A counter must be cheap, must not block, and must not call into the index.
 */
public interface MetricSink {
    /**
     * Returns the counter of a name. The library asks once per name and then passes each increment, a number above 0.
     *
     * @param name the name of the counter
     * @return what receives the increments
     */
    LongConsumer counter(String name);

    /**
     * Registers a value that the host reads when it wants it, from any thread.
     *
     * @param name the name of the gauge
     * @param value reads the current value
     */
    void gauge(String name, LongSupplier value);
}
