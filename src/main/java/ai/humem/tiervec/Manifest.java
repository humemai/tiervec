package ai.humem.tiervec;

import ai.humem.tiervec.SegmentFormat.CorruptException;
import ai.humem.tiervec.SegmentFormat.UnsupportedException;

import java.io.IOException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.DirectoryStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One generation of the index directory: which segment files are live, the tombstone file that goes with them, and a fingerprint of the
 * data-affecting configuration. A manifest is a file {@code manifest-<generation>.tvm} that is written once under a temporary name and
 * renamed into place; it is never modified. The newest generation that validates is the live one.
 *
 * <p>{@code epoch} and {@code hwm} (the host's id epoch and high-water id) are carried in the file and are {@link #NONE} in this version.
 * {@code versionCounter} restores the tier's version counter; {@code nextSegId} is the smallest segment id not used by this generation.
 * {@code tombGeneration}, {@code tombLength} and {@code tombCrc} describe the tombstone file the manifest names; {@link #publish} fills
 * them in.
 */
@Internal
public record Manifest(long generation, long epoch, long hwm, long versionCounter, long nextSegId, Fingerprint fingerprint,
                       List<Entry> entries, long tombGeneration, long tombLength, int tombCrc) {

    /** a value that was not set (epoch and high-water id in this version) */
    public static final long NONE = Long.MIN_VALUE;
    /** a reader refuses a larger major and opens a larger minor, like the segment format */
    public static final int FORMAT_MAJOR = 1, FORMAT_MINOR = 0;

    private static final byte[] MAGIC = {'T', 'V', 'M', 'A', 'N', '0', '0', '1'};
    private static final byte[] TOMB_MAGIC = {'T', 'V', 'T', 'O', 'M', 'B', '0', '1'};
    private static final Pattern MANIFEST_NAME = Pattern.compile("manifest-(\\d{20})\\.tvm");
    private static final Pattern GEN_FILE = Pattern.compile("(manifest|tomb)-(\\d{20})\\.(tvm|tvd)(\\.tmp)?");
    private static final Pattern SEG_FILE = Pattern.compile("seg-(\\d{6,19})\\.tvs(\\.tmp)?");
    private static final int FIXED_BODY = 100;
    private static final int ENTRY_BYTES = 48;
    private static final long MAX_MANIFEST = 64L << 20;

    /** the data-affecting configuration: segments built with other values are not this index's. efSearch is a search knob and is not part of it. */
    public record Fingerprint(int dim, Metric metric, int M, int efConstruction, boolean sq8) {}

    /** one live segment file {@code seg-<segId>.tvs} and what its header and footer said when it was written */
    public record Entry(long segId, long n, long fileLength, int headerCrc, long verMin, long verMax) {}

    public Manifest {
        Objects.requireNonNull(fingerprint, "fingerprint");
        entries = List.copyOf(entries);
        if (generation < 0 || nextSegId < 0 || tombGeneration < 0) throw new IllegalArgumentException("negative generation or id");
    }

    /** a manifest with epoch and high-water id unset and {@code nextSegId} above every listed segment; the tombstone fields are filled by publish */
    public static Manifest of(long generation, Fingerprint fingerprint, List<Entry> entries) {
        long next = 1;
        for (Entry e : entries) next = Math.max(next, e.segId() + 1);
        return new Manifest(generation, NONE, NONE, 0L, next, fingerprint, entries, 0L, 0L, 0);
    }

    public static String manifestName(long generation) {
        return String.format(Locale.ROOT, "manifest-%020d.tvm", generation);
    }

    public static String segName(long segId) {
        return String.format(Locale.ROOT, "seg-%06d.tvs", segId);
    }

    public static String tombName(long generation) {
        return String.format(Locale.ROOT, "tomb-%020d.tvd", generation);
    }

    // ---- publish ----

    /**
     * Makes {@code m} the live generation. The segment files it names must already exist, written and flushed (SegmentWriter does that).
     * Order: write and flush the tombstone file, flush the directory (the new names are durable), write the manifest under a temporary
     * name, flush it, rename it to its final name, flush the directory. The hook is called after each step. Nothing is cleaned up here:
     * {@link #gc} removes what the new manifest no longer names, once the caller has switched to the new generation.
     *
     * @return the manifest as written (with the tombstone fields filled in)
     */
    public static Manifest publish(Path dir, Manifest m, FaultHook hook) throws IOException {
        Objects.requireNonNull(hook, "hook");
        final Path fin = dir.resolve(manifestName(m.generation()));
        if (Files.exists(fin)) throw new FileAlreadyExistsException(fin.toString());
        hook.at(FaultHook.Step.SEGMENTS_WRITTEN);
        for (Entry e : m.entries()) {
            Path sp = dir.resolve(segName(e.segId()));
            long len = Files.size(sp);
            if (len != e.fileLength()) throw new CorruptException("segment file " + sp.getFileName() + " has " + len + " bytes, the manifest says " + e.fileLength(), "MANIFEST");
        }
        byte[] tomb = encodeTombstones(m.generation(), m.entries());
        try (FileChannel ch = FileChannel.open(dir.resolve(tombName(m.generation())), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            writeFully(ch, tomb);
            ch.force(true);
        }
        Manifest out = new Manifest(m.generation(), m.epoch(), m.hwm(), m.versionCounter(), m.nextSegId(), m.fingerprint(), m.entries(),
                m.generation(), tomb.length, SegmentFormat.crc32c(tomb, 0, tomb.length));
        SegmentFormat.fsyncDir(dir);
        hook.at(FaultHook.Step.DIR_SYNCED_AFTER_SEGMENTS);
        byte[] body = out.encode(FORMAT_MAJOR, FORMAT_MINOR);
        final Path tmp = dir.resolve(manifestName(m.generation()) + ".tmp");
        try (FileChannel ch = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
            writeFully(ch, body);
            hook.at(FaultHook.Step.MANIFEST_TMP_WRITTEN);
            ch.force(true);
        }
        hook.at(FaultHook.Step.MANIFEST_TMP_SYNCED);
        Files.move(tmp, fin, StandardCopyOption.ATOMIC_MOVE);
        hook.at(FaultHook.Step.MANIFEST_RENAMED);
        SegmentFormat.fsyncDir(dir);
        hook.at(FaultHook.Step.DIR_SYNCED_AFTER_RENAME);
        return out;
    }

    private static void writeFully(FileChannel ch, byte[] b) throws IOException {
        ByteBuffer bb = ByteBuffer.wrap(b);
        while (bb.hasRemaining()) ch.write(bb);
    }

    // ---- read ----

    /**
     * The newest manifest in the directory that validates: magic, version, CRC32C, and every file it names present with the recorded
     * length and a valid header, footer and locator (a constant amount of work per segment, no section is read). A newer candidate that
     * fails is skipped and the next older one is tried, which is what a crash between the steps of {@link #publish} needs. Returns null when
     * the directory has no manifest file at all. Throws {@link CorruptException} when there are manifests but none validates. A manifest
     * or segment that is intact but written in a format this reader does not understand throws {@link UnsupportedException} at once and
     * is not skipped: falling back to an older generation would silently serve stale data.
     */
    public static Manifest readLatest(Path dir) throws IOException {
        List<Long> gens = generations(dir);
        if (gens.isEmpty()) return null;
        CorruptException first = null;
        for (long g : gens) {
            try {
                return load(dir, g);
            } catch (CorruptException e) {
                if (first == null) first = e;
                else first.addSuppressed(e);
            }
        }
        throw new CorruptException("no valid manifest among " + gens.size() + " candidates in " + dir + "; the newest failed with: " + first.getMessage(), first);
    }

    private static List<Long> generations(Path dir) throws IOException {
        List<Long> out = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                Matcher mt = MANIFEST_NAME.matcher(p.getFileName().toString());
                if (!mt.matches()) continue;
                try {
                    out.add(Long.parseLong(mt.group(1)));
                } catch (NumberFormatException e) {
                    // more digits than a long: not a file this code wrote
                }
            }
        }
        out.sort(Comparator.reverseOrder());
        return out;
    }

    private static Manifest load(Path dir, long gen) throws IOException {
        final Path p = dir.resolve(manifestName(gen));
        if (Files.size(p) > MAX_MANIFEST) throw new CorruptException("manifest file is implausibly large", "MANIFEST");
        Manifest m = decode(Files.readAllBytes(p));
        if (m.generation() != gen) throw new CorruptException("generation " + m.generation() + " inside " + p.getFileName() + " does not match its name", "MANIFEST");
        Fingerprint fp = m.fingerprint();
        for (Entry e : m.entries()) {
            Path sp = dir.resolve(segName(e.segId()));
            SegmentInfo si;
            try {
                si = SegmentReader.inspect(sp);
            } catch (NoSuchFileException x) {
                throw new CorruptException("segment file " + sp.getFileName() + " is missing", "MANIFEST");
            }
            if (si.fileLength() != e.fileLength() || si.headerCrc() != e.headerCrc() || si.n() != e.n() || si.verMin() != e.verMin()
                    || si.verMax() != e.verMax() || si.dim() != fp.dim() || si.metric() != fp.metric()) {
                throw new CorruptException("segment file " + sp.getFileName() + " does not match its manifest entry", "MANIFEST");
            }
        }
        final Path tp = dir.resolve(tombName(m.tombGeneration()));
        byte[] tomb;
        try {
            tomb = Files.readAllBytes(tp);
        } catch (NoSuchFileException x) {
            throw new CorruptException("tombstone file " + tp.getFileName() + " is missing", "MANIFEST");
        }
        if (tomb.length != m.tombLength() || SegmentFormat.crc32c(tomb, 0, tomb.length) != m.tombCrc()) {
            throw new CorruptException("tombstone file " + tp.getFileName() + " does not match the manifest", "TOMBSTONES");
        }
        verifyTombstones(tomb, m);
        return m;
    }

    // ---- cleanup ----

    /**
     * Deletes every file of this format in the directory that {@code live} does not name: older and newer manifests, retired segment files,
     * old tombstone files and temporary files. Files of other names are left alone. Returns how many files were deleted.
     */
    public static int gc(Path dir, Manifest live) throws IOException {
        Set<String> keep = new HashSet<>();
        keep.add(manifestName(live.generation()));
        keep.add(tombName(live.tombGeneration()));
        for (Entry e : live.entries()) keep.add(segName(e.segId()));
        int deleted = 0;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                String name = p.getFileName().toString();
                if (keep.contains(name) || !(GEN_FILE.matcher(name).matches() || SEG_FILE.matcher(name).matches())) continue;
                if (!Files.isRegularFile(p, LinkOption.NOFOLLOW_LINKS)) continue;
                if (Files.deleteIfExists(p)) deleted++;
            }
        }
        return deleted;
    }

    /** the highest generation and the highest segment id seen in any file name of this format, temporary files included (0 when none): new names must go above both */
    static long[] highestOnDisk(Path dir) throws IOException {
        long maxGen = 0, maxSeg = 0;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                String name = p.getFileName().toString();
                Matcher g = GEN_FILE.matcher(name);
                Matcher s = SEG_FILE.matcher(name);
                try {
                    if (g.matches()) maxGen = Math.max(maxGen, Long.parseLong(g.group(2)));
                    else if (s.matches()) maxSeg = Math.max(maxSeg, Long.parseLong(s.group(1)));
                } catch (NumberFormatException e) {
                    // more digits than a long: ignore
                }
            }
        }
        return new long[] {maxGen, maxSeg};
    }

    // ---- bytes of the manifest ----

    byte[] encode(int major, int minor) {
        final int body = FIXED_BODY + ENTRY_BYTES * entries.size();
        ByteBuffer b = ByteBuffer.allocate(body + 4).order(ByteOrder.LITTLE_ENDIAN);
        b.put(MAGIC).putShort((short) major).putShort((short) minor).putInt(body);
        b.putLong(generation).putLong(epoch).putLong(hwm).putLong(versionCounter).putLong(nextSegId);
        b.putInt(fingerprint.dim()).put((byte) SegmentFormat.metricCode(fingerprint.metric())).put((byte) SegmentFormat.ID_WIDTH);
        b.put((byte) SegmentFormat.LAYOUT_CSR).put((byte) (fingerprint.sq8() ? 1 : 0));
        b.putInt(fingerprint.M()).putInt(fingerprint.efConstruction());
        b.putLong(tombGeneration).putLong(tombLength).putInt(tombCrc).putInt(0);
        b.putInt(entries.size());
        for (Entry e : entries) {
            b.putLong(e.segId()).putLong(e.n()).putLong(e.fileLength()).putInt(e.headerCrc()).putInt(0).putLong(e.verMin()).putLong(e.verMax());
        }
        if (b.position() != body) throw new IllegalStateException("manifest body is " + b.position() + " bytes, expected " + body);
        b.putInt(SegmentFormat.crc32c(b.array(), 0, body));
        return b.array();
    }

    static Manifest decode(byte[] raw) throws IOException {
        if (raw.length < FIXED_BODY + 4) throw new CorruptException("manifest too short (" + raw.length + " bytes)", "MANIFEST");
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < MAGIC.length; i++) {
            if (b.get(i) != MAGIC[i]) throw new CorruptException("bad manifest magic", "MANIFEST");
        }
        final int major = b.getShort(8) & 0xFFFF;
        if (major < 1 || major > FORMAT_MAJOR) throw new UnsupportedException("manifest format major version " + major + " is not supported (this reader supports " + FORMAT_MAJOR + ")");
        final long bodyLen = Integer.toUnsignedLong(b.getInt(12));
        if (bodyLen < FIXED_BODY || bodyLen + 4 != raw.length) throw new CorruptException("bad manifest length", "MANIFEST");
        final int bl = (int) bodyLen;
        if (SegmentFormat.crc32c(raw, 0, bl) != b.getInt(bl)) throw new CorruptException("CRC32C mismatch", "MANIFEST");
        b.limit(bl);
        b.position(16);
        try {
            long gen = b.getLong();
            long epoch = b.getLong();
            long hwm = b.getLong();
            long counter = b.getLong();
            long nextSeg = b.getLong();
            int dim = b.getInt();
            int metricCode = b.get() & 0xFF;
            int idWidth = b.get() & 0xFF;
            int layout = b.get() & 0xFF;
            boolean sq8 = (b.get() & 0xFF) != 0;
            int m = b.getInt();
            int efc = b.getInt();
            long tombGen = b.getLong();
            long tombLen = b.getLong();
            int tombCrc = b.getInt();
            b.getInt(); // reserved
            long count = Integer.toUnsignedLong(b.getInt());
            if (count > (bodyLen - FIXED_BODY) / ENTRY_BYTES) throw new CorruptException("entry count " + count + " does not fit the manifest", "MANIFEST");
            if (dim < 1 || gen < 0 || nextSeg < 0 || tombGen < 0 || tombLen < 0) throw new CorruptException("bad manifest field", "MANIFEST");
            Metric metric = SegmentFormat.metricOf(metricCode);
            if (idWidth != SegmentFormat.ID_WIDTH) throw new UnsupportedException("id width " + idWidth + " is not supported");
            if (layout != SegmentFormat.LAYOUT_CSR) throw new UnsupportedException("layer 0 layout " + layout + " is not supported");
            List<Entry> entries = new ArrayList<>();
            Set<Long> seen = new HashSet<>();
            for (long i = 0; i < count; i++) {
                long segId = b.getLong();
                long n = b.getLong();
                long fileLength = b.getLong();
                int headerCrc = b.getInt();
                b.getInt(); // reserved
                long verMin = b.getLong();
                long verMax = b.getLong();
                if (segId < 0 || n < 0 || fileLength < 0 || !seen.add(segId)) throw new CorruptException("bad manifest entry " + i, "MANIFEST");
                entries.add(new Entry(segId, n, fileLength, headerCrc, verMin, verMax));
            }
            return new Manifest(gen, epoch, hwm, counter, nextSeg, new Fingerprint(dim, metric, m, efc, sq8), entries, tombGen, tombLen, tombCrc);
        } catch (BufferUnderflowException e) {
            throw new CorruptException("manifest is shorter than its fields", e);
        }
    }

    // ---- bytes of the tombstone file ----

    /**
     * The tombstone file of a generation: magic, version, generation, segment count, then per segment its id, its row count and one bit per
     * row (set = deleted), and a CRC32C of everything before it. This version writes only all-zero bitmaps.
     */
    private static byte[] encodeTombstones(long gen, List<Entry> entries) {
        long size = 24 + 4;
        for (Entry e : entries) size += 16 + 8L * ((e.n() + 63) / 64);
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(size)).order(ByteOrder.LITTLE_ENDIAN);
        b.put(TOMB_MAGIC).putShort((short) FORMAT_MAJOR).putShort((short) FORMAT_MINOR).putLong(gen).putInt(entries.size());
        for (Entry e : entries) {
            b.putLong(e.segId()).putLong(e.n());
            b.position(b.position() + 8 * (int) ((e.n() + 63) / 64)); // all-zero bitmap: the array is already zero
        }
        if (b.position() != size - 4) throw new IllegalStateException("tombstone file is " + b.position() + " bytes, expected " + (size - 4));
        b.putInt(SegmentFormat.crc32c(b.array(), 0, (int) size - 4));
        return b.array();
    }

    private static void verifyTombstones(byte[] raw, Manifest m) throws IOException {
        if (raw.length < 28) throw new CorruptException("tombstone file too short", "TOMBSTONES");
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < TOMB_MAGIC.length; i++) {
            if (b.get(i) != TOMB_MAGIC[i]) throw new CorruptException("bad tombstone magic", "TOMBSTONES");
        }
        if ((b.getShort(8) & 0xFFFF) > FORMAT_MAJOR) throw new UnsupportedException("tombstone format major version is not supported");
        final int end = raw.length - 4;
        if (SegmentFormat.crc32c(raw, 0, end) != b.getInt(end)) throw new CorruptException("CRC32C mismatch", "TOMBSTONES");
        if (b.getLong(12) != m.tombGeneration() || Integer.toUnsignedLong(b.getInt(20)) != m.entries().size()) {
            throw new CorruptException("tombstone file does not belong to this manifest", "TOMBSTONES");
        }
        long pos = 24;
        for (Entry e : m.entries()) {
            long words = (e.n() + 63) / 64;
            if (pos + 16 + 8 * words > end || b.getLong((int) pos) != e.segId() || b.getLong((int) pos + 8) != e.n()) {
                throw new CorruptException("tombstone bitmap of segment " + e.segId() + " is missing or does not match", "TOMBSTONES");
            }
            pos += 16;
            for (long w = 0; w < words; w++, pos += 8) {
                if (b.getLong((int) pos) != 0L) throw new UnsupportedException("non-empty tombstones are not supported by this version");
            }
        }
        if (pos != end) throw new CorruptException("tombstone file has trailing bytes", "TOMBSTONES");
    }
}
