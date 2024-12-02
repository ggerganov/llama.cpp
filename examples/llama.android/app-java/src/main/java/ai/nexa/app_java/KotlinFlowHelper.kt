package ai.nexa.app_java

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class KotlinFlowHelper {
    private val scope = CoroutineScope(Dispatchers.IO)

    fun collectFlow(
        flow: Flow<String>,  // Added missing flow parameter
        onToken: (String) -> Unit,
        onComplete: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        scope.launch {
            try {
                val fullResponse = StringBuilder()
                withContext(Dispatchers.IO) {
                    flow.collect { value ->
                        fullResponse.append(value)
                        withContext(Dispatchers.Main) {
                            onToken(value)
                        }
                    }
                }
                withContext(Dispatchers.Main) {
                    onComplete(fullResponse.toString())
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    onError(e.message ?: "Unknown error")
                }
            }
        }
    }

    fun cancel() {
        scope.coroutineContext.cancelChildren()
    }
}