plugins { java }

dependencies {
    implementation(project(":nn"))
    implementation("org.openjdk.jmh:jmh-core:1.37")
    annotationProcessor("org.openjdk.jmh:jmh-generator-annprocess:1.37")
    testImplementation(platform("org.junit:junit-bom:5.13.4"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<JavaExec>().configureEach {
    javaLauncher.set(javaToolchains.launcherFor(java.toolchain))
    classpath = sourceSets.main.get().runtimeClasspath
    minHeapSize = "256m"
    maxHeapSize = "512m"
    workingDir(rootProject.projectDir)
}

for (kind in listOf("kernel", "training", "inference")) {
    tasks.register<JavaExec>(kind) {
        group = "benchmark"
        description = "Runs forked JMH $kind benchmarks with allocation and GC profiling"
        mainClass.set("ch.lolo.benchmark.BenchmarkRunner")
        systemProperty("benchmark.kind", kind)
    }
}

tasks.register<JavaExec>("profile") {
    group = "benchmark"
    description = "Profiles training phases over repeated complete synthetic epochs"
    mainClass.set("ch.lolo.benchmark.TrainingProfile")
    val recording = providers.gradleProperty("profileJfr")
    if (recording.isPresent) {
        jvmArgs("-XX:StartFlightRecording=filename=${recording.get()},settings=profile,dumponexit=true")
    }
}
