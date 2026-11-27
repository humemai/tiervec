package ai.humem.tiervec;

import java.io.IOException;
import java.nio.file.Path;

/**
 * The reader under test of the segment, manifest, and tier-reopen tests: the heap reader, or the mapped one in a subclass named
 * {@code Mapped...Test} (#27). Surefire starts one JVM per test class, so a subclass that changes {@link #storage} changes no other class.
 */
final class TestReaders {
    static TieredIndex.VectorStorage storage = TieredIndex.VectorStorage.HEAP;

    private TestReaders() {}

    static SegmentReader.Loaded readFull(Path file, boolean verifyAll) throws IOException {
        return SegmentReader.readFull(file, verifyAll, storage);
    }

    static SealedGraph read(Path file, boolean verifyAll) throws IOException {
        return readFull(file, verifyAll).graph();
    }

    /** heap bytes that a loaded graph of {@code rows} rows holds less than the graph it was written from: the mapped vectors */
    static long mappedBytes(int rows, int dim) {
        return storage == TieredIndex.VectorStorage.MAPPED ? 4L * rows * dim : 0;
    }
}
