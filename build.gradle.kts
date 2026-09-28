// The Android Gradle Plugin goes on the root buildscript classpath so it shares a
// classloader with the Kotlin Gradle Plugin (required by AGP 9's built-in Kotlin support).
// It is skipped for `-PcoreOnly` builds so the pure-Kotlin core can be built and tested on
// machines that cannot reach Google Maven.
buildscript {
    if (!gradle.startParameter.projectProperties.containsKey("coreOnly")) {
        repositories {
            google()
            mavenCentral()
        }
        dependencies {
            classpath(libs.android.gradle.plugin)
        }
    }
}

plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
