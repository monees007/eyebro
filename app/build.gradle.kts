plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.eyebro"
    compileSdk = 34 // STABLE SDK

    defaultConfig {
        applicationId = "com.example.eyebro"
        minSdk = 24 // Required for ARCore
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        mlModelBinding = true
    }
}

dependencies {
    // FORCE STABLE VERSIONS to avoid the API 36 error
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Standard UI
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // ARCore (Google Play Services for AR)
    implementation("com.google.ar:core:1.41.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // Testing (Standard default)
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    implementation("com.google.mlkit:image-labeling:17.0.7")
    implementation("com.google.android.material:material:1.9.0")
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
}