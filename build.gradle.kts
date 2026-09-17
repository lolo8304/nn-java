allprojects {
    group = "ch.lolo"
    version = "1.0.0"
    repositories { mavenCentral() }
}
subprojects {
    plugins.withType<JavaPlugin> {
        extensions.configure<JavaPluginExtension> {
            toolchain { languageVersion.set(JavaLanguageVersion.of(25)) }
        }
    }
}
