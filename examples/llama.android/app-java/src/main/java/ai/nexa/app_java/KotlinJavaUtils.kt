package ai.nexa.app_java

import java.util.function.Consumer

object KotlinJavaUtils {
    @JvmStatic
    fun toKotlinCallback(callback: Consumer<String>): (String) -> Unit = { value ->
        callback.accept(value)
        Unit
    }
}