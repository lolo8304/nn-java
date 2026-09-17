plugins {
    application
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
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
