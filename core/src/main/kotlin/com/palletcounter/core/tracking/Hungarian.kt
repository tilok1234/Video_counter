package com.palletcounter.core.tracking

/**
 * Minimum-cost bipartite assignment (Kuhn-Munkres with potentials, O(n^2 m)).
 *
 * Entries equal to [INFEASIBLE] (or above `maxCost`) are never returned as matches.
 */
object Hungarian {
    const val INFEASIBLE = 1e6f

    /**
     * @param cost rows x cols matrix (rows may differ from cols).
     * @return pairs (row, col) of the optimal assignment with cost <= [maxCost].
     */
    fun solve(cost: Array<FloatArray>, maxCost: Float = INFEASIBLE / 2): List<Pair<Int, Int>> {
        val rows = cost.size
        if (rows == 0) return emptyList()
        val cols = cost[0].size
        if (cols == 0) return emptyList()
        val transposed = rows > cols
        val n = if (transposed) cols else rows
        val m = if (transposed) rows else cols
        fun a(i: Int, j: Int): Double =
            (if (transposed) cost[j - 1][i - 1] else cost[i - 1][j - 1]).toDouble()

        val u = DoubleArray(n + 1)
        val v = DoubleArray(m + 1)
        val p = IntArray(m + 1)
        val way = IntArray(m + 1)
        for (i in 1..n) {
            p[0] = i
            var j0 = 0
            val minv = DoubleArray(m + 1) { Double.POSITIVE_INFINITY }
            val used = BooleanArray(m + 1)
            do {
                used[j0] = true
                val i0 = p[j0]
                var delta = Double.POSITIVE_INFINITY
                var j1 = 0
                for (j in 1..m) {
                    if (used[j]) continue
                    val cur = a(i0, j) - u[i0] - v[j]
                    if (cur < minv[j]) {
                        minv[j] = cur
                        way[j] = j0
                    }
                    if (minv[j] < delta) {
                        delta = minv[j]
                        j1 = j
                    }
                }
                for (j in 0..m) {
                    if (used[j]) {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }
                j0 = j1
            } while (p[j0] != 0)
            do {
                val j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while (j0 != 0)
        }
        val result = ArrayList<Pair<Int, Int>>()
        for (j in 1..m) {
            val i = p[j]
            if (i == 0) continue
            val row = if (transposed) j - 1 else i - 1
            val col = if (transposed) i - 1 else j - 1
            if (cost[row][col] <= maxCost) result += row to col
        }
        return result
    }
}
