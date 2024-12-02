/* register_types.cpp */

#include "register_types.h"

#include "KomputeSummatorNode.h"
#include "core/class_db.h"

void
register_kompute_summator_types()
{
    ClassDB::register_class<KomputeSummatorNode>();
}

void
unregister_kompute_summator_types()
{
    // Nothing to do here in this example.
}
