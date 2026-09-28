// Plugins shared by several modules are put on the root classpath once.
// The Android Gradle Plugin is intentionally declared only in :app so that
// `-PcoreOnly` builds never need Google Maven.
plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
