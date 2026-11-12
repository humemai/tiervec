package com.humemai.vec;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a public class that is not part of the supported API (see docs/API.md). It is public only because the classes of the library
 * live in one package and call each other. It can change or disappear in any release, a patch release included, with no changelog line.
 * A nested type that docs/API.md lists as supported stays supported when its outer class is marked.
 */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Internal {
}
