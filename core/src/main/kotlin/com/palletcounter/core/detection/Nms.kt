package com.palletcounter.core.detection

/**
 * Greedy non-maximum suppression.
 *
 * @param classAgnostic when true, boxes of different classes suppress each other too.
 * @param maxCandidates only the highest-scoring candidates are considered (bounds cost).
 */
fun nonMaxSuppression(
    detections: List<Detection>,
    iouThreshold: Float,
    maxDetections: Int = 100,
    classAgnostic: Boolean = true,
    maxCandidates: Int = 1000,
): List<Detection> {
    if (detections.isEmpty()) return emptyList()
    val sorted = detections.sortedByDescending { it.confidence }.take(maxCandidates)
    val suppressed = BooleanArray(sorted.size)
    val kept = ArrayList<Detection>(minOf(sorted.size, maxDetections))
    for (i in sorted.indices) {
        if (suppressed[i]) continue
        val a = sorted[i]
        kept += a
        if (kept.size >= maxDetections) break
        for (j in i + 1 until sorted.size) {
            if (suppressed[j]) continue
            val b = sorted[j]
            if (!classAgnostic && a.classId != b.classId) continue
            if (a.box.iou(b.box) > iouThreshold) suppressed[j] = true
        }
    }
    return kept
}
