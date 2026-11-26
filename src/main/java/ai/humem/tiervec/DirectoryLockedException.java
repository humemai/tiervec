package ai.humem.tiervec;

import java.nio.file.Path;

/**
 * An index directory is in use by another process: that process holds the lock on its {@code LOCK} file. {@link TieredIndex#open} (every
 * overload) and {@link TieredIndex#checkpoint} throw it and change nothing in the directory. It is not a failure of the files, so an
 * {@code open} that has a rebuild source does not rebuild: two owners of one directory would delete each other's files.
 */
public final class DirectoryLockedException extends IllegalStateException {
    private static final long serialVersionUID = 1L;

    private final String directory;

    DirectoryLockedException(Path directory) {
        super("the index directory " + directory + " is in use by another process, which holds the lock on its " + DirectoryLock.FILE + " file");
        this.directory = directory.toString();
    }

    /** @return the directory that is in use */
    public String directory() {
        return directory;
    }
}
