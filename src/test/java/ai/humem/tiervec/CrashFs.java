package ai.humem.tiervec;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.SeekableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.AccessMode;
import java.nio.file.CopyOption;
import java.nio.file.DirectoryStream;
import java.nio.file.FileStore;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.nio.file.WatchEvent;
import java.nio.file.WatchKey;
import java.nio.file.WatchService;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.FileAttributeView;
import java.nio.file.attribute.UserPrincipalLookupService;
import java.nio.file.spi.FileSystemProvider;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * A file system that records what a caller writes, flushes, renames and deletes, so that a test can build the directories that a crash may
 * leave. It wraps a real directory: every call goes through to the default file system, and every call that changes something is also appended
 * to {@link #log}. The library needs no seam for this: a {@link Path} of this file system takes the same {@code Files} and
 * {@code FileChannel} calls to {@link Provider}. A call that the recorder does not know (a mapped buffer, a transfer, a copy) throws
 * {@link UnsupportedOperationException}, so a new way of writing in the library cannot pass unrecorded.
 *
 * <p>The crash model, for a prefix of the log ({@link #replay}, {@link Disk#image}):
 * <ul>
 * <li>The bytes of a file are durable up to its last flush ({@code FileChannel.force} on the file). What was written later may be there in
 * full, in part, as zeros, or not at all ({@link Content}).</li>
 * <li>A name (a new file or directory, a rename, a delete) is durable from the next flush of the directory that holds it. Before that flush
 * each such change may or may not be there, independently of the others and in any combination.</li>
 * <li>A process kill keeps everything: that is the image with every change and every byte.</li>
 * </ul>
 * This is a model that is harsher than common file systems, not a proof about any one of them. It does not model a drive that lies about a
 * flush.
 */
final class CrashFs {
    enum Kind { CREATE, MKDIR, WRITE, TRUNCATE, SYNC_FILE, SYNC_DIR, RENAME, DELETE, NOTHING }

    /** one recorded change. {@code dir} is the directory of a name change or of a directory flush, {@code node} the file or directory it is about */
    record Op(Kind kind, int dir, int node, String name, String to, long position, byte[] data) {
        @Override
        public String toString() {
            return kind + (name == null ? "" : " " + name) + (to == null ? "" : " -> " + to) + (kind == Kind.WRITE ? " " + data.length + " bytes at " + position : "")
                    + " [node " + node + (dir >= 0 ? " in " + dir : "") + "]";
        }
    }

    static final Op NOTHING = new Op(Kind.NOTHING, -1, -1, null, null, 0, null);
    /** the node of the directory that the file system wraps; it exists and is durable from the start */
    static final int ROOT = 0;

    final Path realRoot;
    final List<Op> log = new ArrayList<>();
    /** the node of every file and directory that exists now, by real path */
    private final Map<Path, Integer> nodes = new HashMap<>();
    private int nextNode = 1;
    private final Provider provider = new Provider();
    private final Fs fs = new Fs();

    /** records into a new, empty directory */
    CrashFs(Path emptyDir) throws IOException {
        this.realRoot = emptyDir.toRealPath();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(realRoot)) {
            if (ds.iterator().hasNext()) throw new IllegalArgumentException(emptyDir + " is not empty");
        }
        nodes.put(realRoot, ROOT);
    }

    /** the wrapped directory as a path of this file system */
    Path root() {
        return new CrashPath(realRoot);
    }

    /**
     * Goes on after a process kill: writes what the kill left (every change of {@code prefix}) into a new empty directory and records further
     * calls after the prefix, so that what the prefix had not flushed is still not flushed.
     */
    static CrashFs resume(List<Op> prefix, Path emptyDir) throws IOException {
        CrashFs c = new CrashFs(emptyDir);
        Disk d = replay(prefix, prefix.size());
        d.image(d.pending() == 0 ? new boolean[0] : filled(d.pending()), Content.WRITTEN).writeTo(c.realRoot, (path, node) -> c.nodes.put(path, node));
        c.log.addAll(prefix);
        for (Op op : prefix) c.nextNode = Math.max(c.nextNode, op.node() + 1);
        return c;
    }

    private static boolean[] filled(int n) {
        boolean[] b = new boolean[n];
        Arrays.fill(b, true);
        return b;
    }

    // ---- the model ----

    /** which bytes of a file a crash leaves when some were written after the last flush */
    enum Content {
        /** only what was flushed */
        FLUSHED,
        /** everything that was written (a process kill) */
        WRITTEN,
        /** the flushed length, or the written one, cut halfway between */
        HALF,
        /** the written length, with zeros wherever the bytes were not flushed */
        ZEROS,
        /** a torn write: the written length, every other block of 512 bytes as written and the others as flushed (zeros past the flushed end) */
        TORN
    }

    private static final class FileState {
        byte[] written = new byte[0];
        int length;
        byte[] flushed = new byte[0];

        void write(long position, byte[] data) {
            int end = Math.toIntExact(position + data.length);
            if (end > written.length) written = Arrays.copyOf(written, Math.max(end, written.length * 2));
            System.arraycopy(data, 0, written, (int) position, data.length);
            length = Math.max(length, end);
        }

        byte[] image(Content c) {
            byte[] w = Arrays.copyOf(written, length);
            if (Arrays.equals(w, flushed)) return w;
            switch (c) {
                case FLUSHED:
                    return flushed;
                case WRITTEN:
                    return w;
                case HALF:
                    return Arrays.copyOf(w, (flushed.length + w.length) / 2);
                case ZEROS:
                    for (int i = 0; i < w.length; i++) {
                        if (i >= flushed.length || flushed[i] != w[i]) w[i] = 0;
                    }
                    return w;
                default:
                    for (int i = 0; i < w.length; i++) {
                        if ((i / 512) % 2 == 1) w[i] = i < flushed.length ? flushed[i] : 0;
                    }
                    return w;
            }
        }
    }

    private static final class DirState {
        final Map<String, Integer> flushed = new TreeMap<>();
        final List<Op> pending = new ArrayList<>();
    }

    /** what a prefix of the log leaves, before it is decided which of the unflushed changes survive */
    static final class Disk {
        private final Map<Integer, FileState> files = new HashMap<>();
        private final Map<Integer, DirState> dirs = new TreeMap<>();

        /** the number of name changes that no directory flush has made durable */
        int pending() {
            return dirs.values().stream().mapToInt(d -> d.pending.size()).sum();
        }

        /** the unflushed name changes, in the order that {@link #image} numbers them */
        List<Op> pendingOps() {
            List<Op> out = new ArrayList<>();
            for (DirState d : dirs.values()) out.addAll(d.pending);
            return out;
        }

        /** the directory tree after a crash in which the unflushed name changes marked in {@code kept} survived and files hold {@code content} */
        Image image(boolean[] kept, Content content) {
            Map<Integer, Map<String, Integer>> names = new HashMap<>();
            int i = 0;
            for (Map.Entry<Integer, DirState> e : dirs.entrySet()) {
                Map<String, Integer> m = new TreeMap<>(e.getValue().flushed);
                for (Op op : e.getValue().pending) {
                    if (kept[i++]) apply(m, op);
                }
                names.put(e.getKey(), m);
            }
            Image out = new Image();
            walk(ROOT, "", names, content, out);
            return out;
        }

        private void walk(int dir, String prefix, Map<Integer, Map<String, Integer>> names, Content content, Image out) {
            for (Map.Entry<String, Integer> e : names.get(dir).entrySet()) {
                String path = prefix + e.getKey();
                if (names.containsKey(e.getValue())) {
                    out.entries.put(path, new Item(e.getValue(), null));
                    walk(e.getValue(), path + "/", names, content, out);
                } else {
                    out.entries.put(path, new Item(e.getValue(), files.get(e.getValue()).image(content)));
                }
            }
        }

        private static void apply(Map<String, Integer> names, Op op) {
            switch (op.kind()) {
                case CREATE:
                case MKDIR:
                    names.put(op.name(), op.node());
                    break;
                case RENAME:
                    names.remove(op.name(), op.node()); // nothing to remove when the old name did not survive
                    names.put(op.to(), op.node());
                    break;
                case DELETE:
                    names.remove(op.name(), op.node());
                    break;
                default:
                    throw new IllegalStateException(op.toString());
            }
        }
    }

    /** a file ({@code bytes} not null) or a directory */
    record Item(int node, byte[] bytes) {}

    /** a directory tree by relative path, parents before their children */
    static final class Image {
        final Map<String, Item> entries = new LinkedHashMap<>();

        void writeTo(Path emptyDir, java.util.function.BiConsumer<Path, Integer> each) throws IOException {
            for (Map.Entry<String, Item> e : entries.entrySet()) {
                Path p = emptyDir.resolve(e.getKey());
                if (e.getValue().bytes() == null) Files.createDirectory(p);
                else Files.write(p, e.getValue().bytes());
                each.accept(p, e.getValue().node());
            }
        }
    }

    /** replays the first {@code k} changes of the log */
    static Disk replay(List<Op> log, int k) {
        Disk d = new Disk();
        d.dirs.put(ROOT, new DirState());
        for (Op op : log.subList(0, k)) {
            switch (op.kind()) {
                case CREATE:
                    d.files.put(op.node(), new FileState());
                    d.dirs.get(op.dir()).pending.add(op);
                    break;
                case MKDIR:
                    d.dirs.put(op.node(), new DirState());
                    d.dirs.get(op.dir()).pending.add(op);
                    break;
                case RENAME:
                case DELETE:
                    d.dirs.get(op.dir()).pending.add(op);
                    break;
                case WRITE:
                    d.files.get(op.node()).write(op.position(), op.data());
                    break;
                case TRUNCATE: {
                    FileState f = d.files.get(op.node());
                    f.written = Arrays.copyOf(f.written, (int) op.position());
                    f.length = (int) op.position();
                    break;
                }
                case SYNC_FILE: {
                    FileState f = d.files.get(op.node());
                    f.flushed = Arrays.copyOf(f.written, f.length);
                    break;
                }
                case SYNC_DIR: {
                    DirState s = d.dirs.get(op.node());
                    for (Op p : s.pending) Disk.apply(s.flushed, p);
                    s.pending.clear();
                    break;
                }
                default:
                    break;
            }
        }
        return d;
    }

    // ---- recording ----

    private synchronized int node(Path real) throws IOException {
        Integer n = nodes.get(real);
        if (n == null) throw new IOException("the recorder does not know " + real);
        return n;
    }

    private synchronized void record(Kind kind, int dir, int node, String name, String to, long position, byte[] data) {
        log.add(new Op(kind, dir, node, name, to, position, data));
    }

    private static Path real(Path p) {
        if (p instanceof CrashPath c) return c.real;
        throw new java.nio.file.ProviderMismatchException(String.valueOf(p));
    }

    private final class Fs extends FileSystem {
        @Override public FileSystemProvider provider() { return provider; }
        @Override public void close() {}
        @Override public boolean isOpen() { return true; }
        @Override public boolean isReadOnly() { return false; }
        @Override public String getSeparator() { return realRoot.getFileSystem().getSeparator(); }
        @Override public Iterable<Path> getRootDirectories() { throw new UnsupportedOperationException(); }
        @Override public Iterable<FileStore> getFileStores() { throw new UnsupportedOperationException(); }
        @Override public Set<String> supportedFileAttributeViews() { return Set.of("basic"); }
        @Override public Path getPath(String first, String... more) { return new CrashPath(realRoot.getFileSystem().getPath(first, more)); }
        @Override public PathMatcher getPathMatcher(String s) { throw new UnsupportedOperationException(); }
        @Override public UserPrincipalLookupService getUserPrincipalLookupService() { throw new UnsupportedOperationException(); }
        @Override public WatchService newWatchService() { throw new UnsupportedOperationException(); }
    }

    private final class CrashPath implements Path {
        final Path real;

        CrashPath(Path real) {
            this.real = real;
        }

        private Path wrap(Path p) {
            return p == null ? null : new CrashPath(p);
        }

        @Override public FileSystem getFileSystem() { return fs; }
        @Override public boolean isAbsolute() { return real.isAbsolute(); }
        @Override public Path getRoot() { return wrap(real.getRoot()); }
        @Override public Path getFileName() { return wrap(real.getFileName()); }
        @Override public Path getParent() { return wrap(real.getParent()); }
        @Override public int getNameCount() { return real.getNameCount(); }
        @Override public Path getName(int i) { return wrap(real.getName(i)); }
        @Override public Path subpath(int a, int b) { return wrap(real.subpath(a, b)); }
        @Override public boolean startsWith(Path other) { return real.startsWith(real(other)); }
        @Override public boolean endsWith(Path other) { return real.endsWith(real(other)); }
        @Override public Path normalize() { return wrap(real.normalize()); }
        @Override public Path resolve(Path other) { return wrap(real.resolve(real(other))); }
        @Override public Path relativize(Path other) { return wrap(real.relativize(real(other))); }
        @Override public URI toUri() { throw new UnsupportedOperationException(); }
        @Override public Path toAbsolutePath() { return wrap(real.toAbsolutePath()); }
        @Override public Path toRealPath(LinkOption... o) throws IOException { return wrap(real.toRealPath(o)); }
        @Override public WatchKey register(WatchService w, WatchEvent.Kind<?>[] e, WatchEvent.Modifier... m) { throw new UnsupportedOperationException(); }
        @Override public int compareTo(Path other) { return real.compareTo(real(other)); }
        @Override public boolean equals(Object o) { return o instanceof CrashPath c && c.fs() == fs && c.real.equals(real); }
        @Override public int hashCode() { return real.hashCode(); }
        @Override public String toString() { return real.toString(); }

        private Fs fs() {
            return fs;
        }

        @Override
        public Iterator<Path> iterator() {
            List<Path> names = new ArrayList<>();
            for (Path p : real) names.add(wrap(p));
            return names.iterator();
        }
    }

    private final class Provider extends FileSystemProvider {
        @Override public String getScheme() { return "crash"; }
        @Override public FileSystem newFileSystem(URI u, Map<String, ?> env) { throw new UnsupportedOperationException(); }
        @Override public FileSystem getFileSystem(URI u) { return fs; }
        @Override public Path getPath(URI u) { throw new UnsupportedOperationException(); }

        @Override
        public FileChannel newFileChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException {
            Path p = real(path);
            synchronized (CrashFs.this) {
                final boolean existed = Files.exists(p);
                final boolean directory = existed && Files.isDirectory(p);
                final long before = existed && !directory ? Files.size(p) : 0;
                FileChannel ch = FileChannel.open(p, options, attrs);
                if (!existed) {
                    int n = nextNode++;
                    nodes.put(p, n);
                    record(Kind.CREATE, node(p.getParent()), n, p.getFileName().toString(), null, 0, null);
                } else if (options.contains(StandardOpenOption.TRUNCATE_EXISTING) && options.contains(StandardOpenOption.WRITE) && before > 0) {
                    record(Kind.TRUNCATE, -1, node(p), null, null, 0, null);
                }
                // a directory above the wrapped one (the parent of the wrapped directory) is outside the model: its flush changes nothing here
                int n = directory && !p.startsWith(realRoot) ? -1 : node(p);
                return new Channel(ch, n, directory);
            }
        }

        @Override
        public SeekableByteChannel newByteChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException {
            return newFileChannel(path, options, attrs);
        }

        @Override
        public DirectoryStream<Path> newDirectoryStream(Path dir, DirectoryStream.Filter<? super Path> filter) throws IOException {
            List<Path> out = new ArrayList<>();
            try (DirectoryStream<Path> ds = Files.newDirectoryStream(real(dir))) {
                for (Path p : ds) {
                    Path w = new CrashPath(p);
                    if (filter.accept(w)) out.add(w);
                }
            }
            return new DirectoryStream<>() {
                @Override public Iterator<Path> iterator() { return out.iterator(); }
                @Override public void close() {}
            };
        }

        @Override
        public void createDirectory(Path dir, FileAttribute<?>... attrs) throws IOException {
            Path p = real(dir).toAbsolutePath();
            synchronized (CrashFs.this) {
                Files.createDirectory(p, attrs);
                int n = nextNode++;
                nodes.put(p, n);
                record(Kind.MKDIR, node(p.getParent()), n, p.getFileName().toString(), null, 0, null);
            }
        }

        @Override
        public void delete(Path path) throws IOException {
            Path p = real(path);
            synchronized (CrashFs.this) {
                if (Files.isDirectory(p)) throw new UnsupportedOperationException("deleting a directory is not recorded");
                Files.delete(p);
                int n = node(p);
                nodes.remove(p);
                record(Kind.DELETE, node(p.getParent()), n, p.getFileName().toString(), null, 0, null);
            }
        }

        @Override
        public void move(Path source, Path target, CopyOption... options) throws IOException {
            Path from = real(source), to = real(target);
            if (!Arrays.asList(options).contains(StandardCopyOption.ATOMIC_MOVE) || !from.getParent().equals(to.getParent())) {
                throw new UnsupportedOperationException("only an atomic rename within a directory is recorded");
            }
            synchronized (CrashFs.this) {
                Files.move(from, to, options);
                int n = node(from);
                nodes.remove(from);
                nodes.put(to, n);
                record(Kind.RENAME, node(from.getParent()), n, from.getFileName().toString(), to.getFileName().toString(), 0, null);
            }
        }

        @Override public void copy(Path a, Path b, CopyOption... o) { throw new UnsupportedOperationException("copy is not recorded"); }
        @Override public boolean isSameFile(Path a, Path b) throws IOException { return Files.isSameFile(real(a), real(b)); }
        @Override public boolean isHidden(Path p) { return false; }
        @Override public FileStore getFileStore(Path p) throws IOException { return Files.getFileStore(real(p)); }
        @Override public void checkAccess(Path p, AccessMode... modes) throws IOException { real(p).getFileSystem().provider().checkAccess(real(p), modes); }

        @Override
        public <V extends FileAttributeView> V getFileAttributeView(Path p, Class<V> type, LinkOption... o) {
            return Files.getFileAttributeView(real(p), type, o);
        }

        @Override
        public <A extends BasicFileAttributes> A readAttributes(Path p, Class<A> type, LinkOption... o) throws IOException {
            return Files.readAttributes(real(p), type, o);
        }

        @Override public Map<String, Object> readAttributes(Path p, String a, LinkOption... o) throws IOException { return Files.readAttributes(real(p), a, o); }
        @Override public void setAttribute(Path p, String a, Object v, LinkOption... o) { throw new UnsupportedOperationException(); }
    }

    private final class Channel extends FileChannel {
        private final FileChannel ch;
        private final int node;
        private final boolean directory;

        Channel(FileChannel ch, int node, boolean directory) {
            this.ch = ch;
            this.node = node;
            this.directory = directory;
        }

        @Override
        public void force(boolean metaData) throws IOException {
            ch.force(metaData);
            if (node >= 0) record(directory ? Kind.SYNC_DIR : Kind.SYNC_FILE, -1, node, null, null, 0, null);
        }

        @Override
        public int write(ByteBuffer src, long position) throws IOException {
            ByteBuffer view = src.duplicate();
            int n = ch.write(src, position);
            byte[] data = new byte[n];
            view.get(data);
            record(Kind.WRITE, -1, node, null, null, position, data);
            return n;
        }

        @Override
        public int write(ByteBuffer src) throws IOException {
            int n = write(src, ch.position());
            ch.position(ch.position() + n);
            return n;
        }

        @Override
        public FileChannel truncate(long size) throws IOException {
            if (size < ch.size()) record(Kind.TRUNCATE, -1, node, null, null, size, null);
            ch.truncate(size);
            return this;
        }

        @Override public int read(ByteBuffer dst) throws IOException { return ch.read(dst); }
        @Override public int read(ByteBuffer dst, long position) throws IOException { return ch.read(dst, position); }
        @Override public long read(ByteBuffer[] dsts, int off, int len) throws IOException { return ch.read(dsts, off, len); }
        @Override public long position() throws IOException { return ch.position(); }
        @Override public FileChannel position(long p) throws IOException { ch.position(p); return this; }
        @Override public long size() throws IOException { return ch.size(); }
        @Override public long write(ByteBuffer[] s, int off, int len) { throw new UnsupportedOperationException("a gathering write is not recorded"); }
        @Override public long transferTo(long p, long n, WritableByteChannel t) { throw new UnsupportedOperationException(); }
        @Override public long transferFrom(ReadableByteChannel s, long p, long n) { throw new UnsupportedOperationException("a transfer is not recorded"); }
        @Override public MappedByteBuffer map(MapMode m, long p, long n) { throw new UnsupportedOperationException("a mapping is not recorded"); }
        @Override public FileLock lock(long p, long n, boolean shared) { throw new UnsupportedOperationException(); }
        @Override public FileLock tryLock(long p, long n, boolean shared) { throw new UnsupportedOperationException(); }
        @Override protected void implCloseChannel() throws IOException { ch.close(); }
    }
}
