pluginManagement {
    repositories {
        // Google Maven is only consulted for Android/Google groups, so the pure-Kotlin
        // core can be built on machines that cannot reach it (see -PcoreOnly below).
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
    }
}

rootProject.name = "EurPalletCounter"

// Pure Kotlin/JVM: detection types, YOLO decoding, tracking, counting, replay CLI.
include(":core")

// Android app. Skip it with `./gradlew -PcoreOnly :core:test` to build and test the
// vision/counting core without an Android SDK.
if (!providers.gradleProperty("coreOnly").isPresent) {
    include(":app")
}
