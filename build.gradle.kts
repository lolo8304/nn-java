allprojects {
    group = "ch.lolo"
    version = "1.0.0"
    repositories { mavenCentral() }
}
subprojects {
    plugins.withType<JavaPlugin> {
        val selectedBackend = providers.gradleProperty("tensorBackend").orElse("vector").get()
        require(selectedBackend in listOf("java", "vector")) { "tensorBackend must be java or vector" }
        tasks.withType<JavaExec>().configureEach {
            systemProperty("tensor.backend", selectedBackend)
            if (selectedBackend == "vector") jvmArgs("--add-modules", "jdk.incubator.vector")
        }
        tasks.withType<Test>().configureEach { useJUnitPlatform() }
        tasks.named<Test>("test") {
            systemProperty("tensor.backend", selectedBackend)
            if (selectedBackend == "vector") jvmArgs("--add-modules", "jdk.incubator.vector")
        }
        val sourceSets = extensions.getByType<SourceSetContainer>()
        val vectorTest = tasks.register<Test>("vectorTest") {
            description = "Runs the test suite with the Vector API backend"
            group = "verification"
            testClassesDirs = sourceSets.named("test").get().output.classesDirs
            classpath = sourceSets.named("test").get().runtimeClasspath
            systemProperty("tensor.backend", "vector")
            jvmArgs("--add-modules", "jdk.incubator.vector")
            shouldRunAfter(tasks.named("test"))
        }
        val javaTest = tasks.register<Test>("javaTest") {
            description = "Runs the test suite with the Java backend and no Vector module"
            group = "verification"
            testClassesDirs = sourceSets.named("test").get().output.classesDirs
            classpath = sourceSets.named("test").get().runtimeClasspath
            systemProperty("tensor.backend", "java")
            shouldRunAfter(tasks.named("test"))
        }
        tasks.named("check") {
            dependsOn(if (selectedBackend == "vector") javaTest else vectorTest)
        }
        extensions.configure<JavaPluginExtension> {
            toolchain { languageVersion.set(JavaLanguageVersion.of(26)) }
        }
    }
}
