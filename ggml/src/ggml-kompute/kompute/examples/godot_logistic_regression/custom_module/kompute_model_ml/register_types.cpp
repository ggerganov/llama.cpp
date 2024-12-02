/* register_types.cpp */

#include "register_types.h"

#include "KomputeModelMLNode.h"
#include "core/class_db.h"

void
register_kompute_model_ml_types()
{
    ClassDB::register_class<KomputeModelMLNode>();
}

void
unregister_kompute_model_ml_types()
{
    // Nothing to do here in this example.
}
