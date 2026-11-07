package com.humemai.vec;

import static org.junit.jupiter.api.Assertions.assertFalse;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;

/** The segment file tests (a sealed graph written, read back and searched, with and without SQ8 codes) with the scalar twin kernels. Runs only in the surefire execution "scalar-twin", which sets -Dhumemai.vec.scalar=true. The tests are the ones of SegmentFormatTest. */
@Tag("scalar")
class ScalarTwinSegmentFormatTest extends SegmentFormatTest {
    @BeforeAll
    static void requireScalarTwin() {
        assertFalse(Kernels.VECTOR, "this class needs -Dhumemai.vec.scalar=true, which the scalar-twin surefire execution sets");
    }
}
