#include <boost/python.hpp>
#include <iostream>
#include <frameobject.h>
#include <string>

class Base {
public:
    Base() : mName("Base") {}
    Base(const std::string& name) : mName(name) {}
    virtual ~Base() {}
    std::string name() const
    { return mName; }
private:
    std::string mName;
};


using namespace boost::python;

#if PY_MAJOR_VERSION >= 3
#   define INIT_MODULE PyInit_mymodule
    extern "C" PyObject* INIT_MODULE();
#else
#   define INIT_MODULE initmymodule
    extern "C" void INIT_MODULE();
#endif


std::string process_output_plugin(const std::string input)
{
    try {
        PyImport_AppendInittab((char*)"mymodule", INIT_MODULE);
        Py_Initialize();
        object main_module = import("__main__");
        dict main_namespace = extract<dict>(main_module.attr("__dict__"));
        object mymodule = import("mymodule");

        main_namespace["precreated_object"] = Base("created on C++ side");
	main_namespace["llm_input"] = input;       
        exec_file("embedding.py", main_namespace, main_namespace);

	boost::python::object llm_output = main_namespace["llm_output"];
	std::string message = boost::python::extract<std::string>(llm_output);

	return message;
	
    } catch (error_already_set& e) {
        PyErr_PrintEx(0);
        return "";
    }
}


using namespace boost::python;

BOOST_PYTHON_MODULE(mymodule)
{
    class_<Base>("Base")
        .def("__str__", &Base::name)
    ;
}

