#include <stdio.h>
#include <string>
#include <string.h>
#define NAPI_EXPERIMENTAL
#define NAPI_EMBEDDING
//#include <node/node_api.h>
#include <libnode/node_api.h>
#include <libnode/js_native_api.h>
#include <libnode/js_native_api_types.h>


class Context {
public:
    napi_platform platform;

};

static Context context;

void process_output_plugin_node_init()
{
    if (napi_create_platform(0, NULL, 0, NULL, NULL, 0, &context.platform) != napi_ok) {
        fprintf(stderr, "Failed creating the platform\n");
        return "error";
    }
    

}

std::string process_output_plugin_node(const std::string start,
				  const std::string state,
				  const std::string input) {

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
      "global.callMe = callMe;";

    if (napi_create_environment(context.platform, NULL, main_script, &env) != napi_ok) {
      fprintf(stderr, "Failed running JS\n");
      return "error1";
    }

    if (napi_get_global(env, &global) != napi_ok) {
        fprintf(stderr, "Failed accessing the global object\n");
        return "Failed accessing the global object";
    }
    napi_create_string_utf8(env, "callMe", strlen("callMe"), &key);
    if (napi_get_property(env, global, key, &cb) != napi_ok) {
        fprintf(stderr, "Failed accessing the global object\n");
        return "Failed accessing the global object";
    }
    {
        if (napi_call_function(env, global, cb, 0, NULL, &result) != napi_ok) {
            fprintf(stderr, "Failed calling JS callback\n");
            return "Failed calling JS callback";
        }
        if (napi_run_environment(env) != napi_ok) {
            fprintf(stderr, "Failed flushing pending JS callbacks\n");
            return "Failed flushing pending JS callbacks";
        }
    }
    napi_close_handle_scope(env, scope);
    if (napi_destroy_environment(env, NULL) != napi_ok) {
        return "destroy";
    }
    return "OK";
}


void process_output_plugin_node_destroy();
void process_output_plugin_node_destroy()
{

    if (napi_destroy_platform(context.platform) != napi_ok) {
        fprintf(stderr, "Failed destroying the platform\n");
        //return "Failed destroying the platform";
    }
}
