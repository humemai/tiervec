package ai.humem.tiervec;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

/**
 * The claim of one process on an index directory: an exclusive operating system lock on the file {@code LOCK} in it. The tiers of one process
 * share the claim (the lock is released when the last handle is closed), because the operating system gives a process one lock on a file and
 * not one per tier. The file is empty, stays in the directory and is never deleted: deleting a lock file would let a third process lock a new
 * file while the second still holds the old one. The operating system releases the lock when the process ends, however it ends, so no stale
 * lock is left behind.
 *
 * <p>No lock is taken where it would mean nothing: on a file system other than the default one, and in a directory that this process cannot
 * write to (it can publish nothing there).
 */
final class DirectoryLock implements AutoCloseable {
    /** the name of the lock file; the cleanup of a checkpoint leaves it alone */
    static final String FILE = "LOCK";

    /** the locks that this process holds, by the real path of the directory; guarded by its own monitor */
    private static final Map<Path, Held> HELD = new HashMap<>();

    private static final class Held {
        final FileChannel channel;
        int handles;

        Held(FileChannel channel) {
            this.channel = channel;
        }
    }

    private Path key; // null for a handle that holds nothing, and after close; guarded by HELD

    private DirectoryLock(Path key) {
        this.key = key;
    }

    /**
     * A handle on the lock of {@code dir}, which exists.
     *
     * @throws DirectoryLockedException when another process holds the lock
     */
    static DirectoryLock acquire(Path dir) throws IOException {
        if (dir.getFileSystem() != FileSystems.getDefault()) return new DirectoryLock(null);
        Path real = dir.toRealPath();
        synchronized (HELD) {
            Held h = HELD.get(real);
            if (h == null) {
                FileChannel ch;
                try {
                    ch = FileChannel.open(real.resolve(FILE), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                } catch (IOException e) {
                    if (!Files.isWritable(real)) return new DirectoryLock(null);
                    throw e;
                }
                FileLock lock;
                try {
                    lock = ch.tryLock();
                } catch (OverlappingFileLockException e) {
                    // this process holds it through another copy of this class (another class loader). The channel stays open: where a lock
                    // belongs to the process, closing any channel of the file would release the lock of the other copy.
                    throw new DirectoryLockedException(dir);
                } catch (IOException | RuntimeException e) {
                    ch.close();
                    throw e;
                }
                if (lock == null) { // another process holds it
                    ch.close();
                    throw new DirectoryLockedException(dir);
                }
                h = new Held(ch);
                HELD.put(real, h);
            }
            h.handles++;
            return new DirectoryLock(real);
        }
    }

    /** a handle on {@code dir} when it is a directory, or null when there is nothing to lock yet */
    static DirectoryLock acquireIfPresent(Path dir) throws IOException {
        return Files.isDirectory(dir) ? acquire(dir) : null;
    }

    /** releases this handle; the lock itself goes with the last handle of the process. A second call does nothing. */
    @Override
    public void close() {
        synchronized (HELD) {
            if (key == null) return;
            Held h = HELD.get(key);
            if (--h.handles == 0) {
                HELD.remove(key);
                try {
                    h.channel.close(); // releases the lock
                } catch (IOException e) {
                    // the descriptor is gone either way, and the lock with it
                }
            }
            key = null;
        }
    }
}
