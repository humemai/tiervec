package com.humemai.vec;

/** One search result. score: similarity, HIGHER IS BETTER for every metric (cosine and dot as is, L2 as the negative squared distance). */
public record Hit(long id, float score) {}
