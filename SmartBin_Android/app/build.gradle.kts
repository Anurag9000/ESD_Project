plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.hilt)
    id("kotlin-kapt")
}

val defaultApiBaseUrl = providers.gradleProperty("SMARTBIN_API_BASE_URL")
    .orElse("http://10.0.2.2:8000/")
    .get()
val defaultWsEventsUrl = providers.gradleProperty("SMARTBIN_WS_EVENTS_URL")
    .orElse("ws://10.0.2.2:8000/events/stream")
    .get()

android {
    namespace = "com.example.smartbin"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.smartbin"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug {
            buildConfigField("boolean", "DEFAULT_DEMO_MODE", "true")
            buildConfigField("String", "API_BASE_URL", "\"$defaultApiBaseUrl\"")
            buildConfigField("String", "WS_EVENTS_URL", "\"$defaultWsEventsUrl\"")
        }
        release {
            isMinifyEnabled = false
            buildConfigField("boolean", "DEFAULT_DEMO_MODE", "false")
            buildConfigField("String", "API_BASE_URL", "\"$defaultApiBaseUrl\"")
            buildConfigField("String", "WS_EVENTS_URL", "\"$defaultWsEventsUrl\"")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
        isCoreLibraryDesugaringEnabled = true
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        buildConfig = true
        compose = true
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.navigation.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    // Hilt
    implementation(libs.hilt.android)
    kapt(libs.hilt.compiler)
    implementation(libs.hilt.navigation.compose)

    // Networking
    implementation(libs.retrofit)
    implementation(libs.retrofit.serialization)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging)
    implementation(libs.kotlinx.serialization.json)

    // Mapping
    implementation(libs.maplibre.android)

    // Charts
    implementation(libs.vico.compose)
    implementation(libs.vico.compose.m3)
    implementation(libs.vico.core)

    // Utils
    implementation(libs.timber)
    coreLibraryDesugaring(libs.desugar.jdk.libs)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}

kapt {
    correctErrorTypes = true
}
