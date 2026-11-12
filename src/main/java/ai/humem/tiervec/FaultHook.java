package ai.humem.tiervec;

import java.io.IOException;

/**
 * A test seam in {@link Manifest#publish}: the publisher calls {@link #at} after each step of the publish order. A hook that throws stops
 * the publish at that point and leaves exactly the files a process kill after that step would leave (the publisher does no cleanup of its
 * own when it fails, because the next open removes whatever no manifest names).
 */
@Internal
public interface FaultHook {
    /** the points between the publish steps, in order */
    enum Step {
        /** the segment files are written and flushed (done by the caller before publish) */
        SEGMENTS_WRITTEN,
        /** the tombstone file is written and flushed, and the directory is flushed: the new names are durable */
        DIR_SYNCED_AFTER_SEGMENTS,
        /** the temporary manifest holds its bytes but they are not flushed yet */
        MANIFEST_TMP_WRITTEN,
        /** the temporary manifest is flushed */
        MANIFEST_TMP_SYNCED,
        /** the temporary manifest was renamed to its final name: the new generation is visible */
        MANIFEST_RENAMED,
        /** the directory is flushed after the rename */
        DIR_SYNCED_AFTER_RENAME
    }

    void at(Step step) throws IOException;

    /** no fault */
    FaultHook NONE = step -> { };
}
