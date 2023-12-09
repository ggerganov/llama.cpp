#include <stdio.h>
#include <string>
#include <string.h>
#include <metacall/metacall.h>
#include <stdio.h>

int sum(double a, double b)
{
	// Parameters to be passed to the sum function
	void * args[] =
	{
		metacall_value_create_double(a), metacall_value_create_double(b)
	};

	void * ret = NULL;

	// Call to sum function
	ret = metacallv("sum", args);

	// Clean up arguments
	for (size_t it = 0; it < sizeof(args) / sizeof(args[0]); ++it)
	{
		metacall_value_destroy(args[it]);
	}

	if (ret == NULL)
	{
	  printf("Function sum returned: NULL\n");
	  return 1;
	}

	printf("Function sum returned: %f\n", metacall_value_to_double(ret));

	// Clean up return value
	metacall_value_destroy(ret);

	return 0;
}

class Context {
public:
  
  struct metacall_log_stdio_type log_stdio = { stdout };
  //void* handle = NULL; // function pointer
};

static Context context;

void process_output_plugin_metacall_init()
{

  
  printf(metacall_print_info());
  
  // Define log stream
  if (metacall_log(METACALL_LOG_STDIO, (void *)&context.log_stdio) != 0)
    {
      printf("error setting log");
      //return cleanup(1);
    }

  // Initialize MetaCall
  if (metacall_initialize() != 0)
    {
      printf("error init");
      //return cleanup(2);
    }

  // Array of scripts to be loaded by MetaCall
  const char * js_scripts[] =
    {
      "script.js"
    };
  
		
		
  // Load scripts
  if (metacall_load_from_file("node",
			      js_scripts,
			      sizeof(js_scripts) / sizeof(js_scripts[0]),
			      //&context.handle
			      NULL
			      ) != 0)
    {
      printf("error loading scripts!");
      //return cleanup(3);
      //return "error loading";
    }

}


std::string process_output_plugin_metacall(const std::string start,
				  const std::string state,
				  const std::string input) {

	// NodeJS

		// Execute sum function
		if (sum(3, 4) != 0)
		{
		  return "error executing";
		}


		

	return "OK";
       
}


void process_output_plugin_metacall_destroy()
{

  //metacall_clear(context.handle);
  //if (
  metacall_destroy();
	//!= 0)
	//{
      //return code != 0 ? -code : -255;
	//    }
}
