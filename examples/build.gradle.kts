plugins {
    application
}

dependencies {
    implementation(project(":nn"))
    testImplementation(platform("org.junit:junit-bom:5.13.4"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test { useJUnitPlatform() }

application {
    mainClass.set("ch.lolo.examples.XorExample")
    // Generated distribution launchers need the same startup configuration as run tasks.
    applicationDefaultJvmArgs = if (providers.gradleProperty("tensorBackend").orElse("vector").get() == "vector")
        listOf("--add-modules", "jdk.incubator.vector", "-Dtensor.backend=vector")
    else listOf("-Dtensor.backend=java")
}

tasks.register<JavaExec>("runXor") {
    group = "application"
    description = "Runs the XOR neural network example"

    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("ch.lolo.examples.XorExample")
}

tasks.register<JavaExec>("runMnist") {
    group = "application"
    description = "Runs the MNIST neural network example"

    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("ch.lolo.examples.MnistExample")
    workingDir(rootProject.projectDir)
}

// Custom run tasks must use the project toolchain, not the Gradle daemon JVM.
tasks.withType<JavaExec>().configureEach {
    javaLauncher.set(javaToolchains.launcherFor(java.toolchain))
}

tasks.register<JavaExec>("runKernelBenchmark") {
    group = "application"
    description = "Runs the local kernel timing and allocation benchmark"
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("ch.lolo.benchmark.KernelBenchmark")
    minHeapSize = "256m"
    maxHeapSize = "512m"
}
