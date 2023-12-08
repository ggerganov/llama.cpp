#include <stdio.h>
#include <string>
#include <string.h>
#define NAPI_EXPERIMENTAL
#define NAPI_EMBEDDING
//#include <node/node_api.h>
#include <libnode/node_api.h>
#include <libnode/js_native_api.h>
#include <libnode/js_native_api_types.h>
std::string process_output_plugin_node(const std::string start,
				  const std::string state,
				  const std::string input) {

    // !!! All napi calls for one given environment must
    // !!! be made from the same thread that created it
    // (except everything napi_threadsafe_function related)

    // This the V8 engine, there must be only one
    napi_platform platform;
    // This is a V8 isolate, there may be multiple
    napi_env env;
    // This holds local references, when it is closed
    // they become available to the GC
    napi_handle_scope scope;
    // These are JS values
    napi_value global;
    napi_value key;
    napi_value cb;
    napi_value result;

    const char *main_script = "console.log('hello world'); "
                              "function callMe() { console.log('called you'); }"
                              // or you can use vm.runInThisContext
                              "global.callMe = callMe;";

    // Do only once
    if (napi_create_platform(0, NULL, 0, NULL, NULL, 0, &platform) != napi_ok) {
        fprintf(stderr, "Failed creating the platform\n");
        return "error";
    }

    // Do for each environment (V8 isolate)
    // 'hello world' will be printed here
    if (napi_create_environment(platform, NULL, main_script, &env) != napi_ok) {
        fprintf(stderr, "Failed running JS\n");
        return "error1";
    }

    // Here you can interact with the environment through Node-API env
    // (refer to the Node-API doc)
    if (napi_get_global(env, &global) != napi_ok) {
        fprintf(stderr, "Failed accessing the global object\n");
        return "Failed accessing the global object";
    }
    napi_create_string_utf8(env, "callMe", strlen("callMe"), &key);
    if (napi_get_property(env, global, key, &cb) != napi_ok) {
        fprintf(stderr, "Failed accessing the global object\n");
        return "Failed accessing the global object";
    }

    // This cycle can be repeated
    {
        // Call a JS function
        // V8 will run in this thread
        if (napi_call_function(env, global, cb, 0, NULL, &result) != napi_ok) {
            fprintf(stderr, "Failed calling JS callback\n");
            return "Failed calling JS callback";
        }
        // (optional) Call this to flush all pending async callbacks
        // V8 will run in this thread
        if (napi_run_environment(env) != napi_ok) {
            fprintf(stderr, "Failed flushing pending JS callbacks\n");
            return "Failed flushing pending JS callbacks";
        }
    }

    // Shutdown everyhing
    napi_close_handle_scope(env, scope);

    if (napi_destroy_environment(env, NULL) != napi_ok) {
        return "destroy";
    }

    if (napi_destroy_platform(platform) != napi_ok) {
        fprintf(stderr, "Failed destroying the platform\n");
        return "Failed destroying the platform";
    }

    return "OK";
}
