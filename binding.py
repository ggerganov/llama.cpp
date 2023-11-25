import os
import json
import re
import clang.cindex

# configurable part

CLANG_VERSION='13.0.1'
#   homebrew installs for llvm (brew info llvm gives details):
#       x64: /usr/local/opt/llvm/lib
#       arm64: /opt/homebrew/opt/llvm/lib
llvmLibPath = "/usr/lib/llvm-15/lib/"

cxxClientRoot = "/home/mdupont/experiments/llama.cpp/"

fileList = [
#    "ggml.cpp",
#    "llama.cpp",
    "examples/server/server.cpp",
]

typeList = [
]

# end of configurable part

clang.cindex.Config.set_library_path(llvmLibPath)


def list_headers_in_dir(path):
    # enumerates a folder but keeps the full pathing for the files returned
    # and removes certain files we don't want (like non-hxx, _json.hxx or _fmt.hxx)

    # list all the files in the folder
    files = os.listdir(path)
    # only include .hxx files
    files = list(filter(lambda x: x.endswith('.hxx'), files))
    # add the folder path back on
    files = list(map(lambda x: path + x, files))
    return files


# parse through the list of files specified and expand wildcards
fullFileList = []
for filePath in fileList:
    if "*" in filePath:
        # wildcard path
        basePath = filePath[:-1]
        if "*" in basePath:
            # if there is still a wildcard, we have an issue...
            raise NotImplementedError(
                "wildcard only supported at end of file path")
        files = list_headers_in_dir(os.path.join(cxxClientRoot, basePath))
        fullFileList = fullFileList + files
    else:
        # normal path
        ff = os.path.join(cxxClientRoot, filePath)
        fullFileList.append(ff)
        print("DBUG",ff)
# exclude _json.hxx files
fullFileList = list(
    filter(lambda x: not x.endswith('_json.hxx'), fullFileList))
# exclude _fmt.hxx files
fullFileList = list(
    filter(lambda x: not x.endswith('_fmt.hxx'), fullFileList))


# generate a list of regexps from the type list (for handling wildcards)
typeListRe = list(map(lambda x: x.replace("*", "(.*)") + "(.*)", typeList))


def is_included_type(name, with_durability=False):

    # TODO(brett19): This should be generalized somehow...
    if "is_compound_operation" in name:
        return False

    if "replica_context" in name:
        return False

    if with_durability is True and '_with_legacy_durability' not in name:
        return False

    for x in typeListRe:
        if re.fullmatch(x, name):
            return True
    return False


opTypes = []
opEnums = []


def parse_type(type):
    typeStr = type.get_canonical().spelling
    return parse_type_str(typeStr)

std_comparators = ["std::less<>", "std::greater<>", "std::less_equal<>", "std::greater_equal<>"]

def parse_type_str(typeStr):
    if typeStr == "std::mutex":
        return {"name": "std::mutex"}
    if typeStr == "std::string":
        return {"name": "std::string"}
    if typeStr == "std::chrono::duration<long long>":
        return {"name": "std::chrono::seconds"}
    if typeStr == "std::chrono::duration<long long, std::ratio<1, 1000>>":
        return {"name": "std::chrono::milliseconds"}
    if typeStr == "std::chrono::duration<long long, std::ratio<1, 1000000>>":
        return {"name": "std::chrono::microseconds"}
    if typeStr == "std::chrono::duration<long long, std::ratio<1, 1000000000>>":
        return {"name": "std::chrono::nanoseconds"}
    if typeStr == "std::error_code":
        return {"name": "std::error_code"}
    if typeStr == "std::monostate":
        return {"name": "std::monostate"}
    if typeStr == "std::byte":
        return {"name": "std::byte"}
    if typeStr == "unsigned long":
        return {"name": "std::size_t"}
    if typeStr == "char":
        return {"name": "std::int8_t"}
    if typeStr == "unsigned char":
        return {"name": "std::uint8_t"}
    if typeStr == "short":
        return {"name": "std::int16_t"}
    if typeStr == "unsigned short":
        return {"name": "std::uint16_t"}
    if typeStr == "int":
        return {"name": "std::int32_t"}
    if typeStr == "unsigned int":
        return {"name": "std::uint32_t"}
    if typeStr == "long long":
        return {"name": "std::int64_t"}
    if typeStr == "unsigned long long":
        return {"name": "std::uint64_t"}
    if typeStr == "bool":
        return {"name": "std::bool"}
    if typeStr == "float":
        return {"name": "std::float"}
    if typeStr == "double":
        return {"name": "std::double"}
    if typeStr == "std::nullptr_t":
        return {"name": "std::nullptr_t"}
    if typeStr in std_comparators:
        return {"name": typeStr}

    tplParts = typeStr.split("<", 1)
    if len(tplParts) > 1:
        tplClassName = tplParts[0]
        tplParams = tplParts[1][:-1]
        if tplClassName == "std::function":
            return {
                "name": "std::function"
            }
        if tplClassName == "std::optional":
            return {
                "name": "std::optional",
                "of": parse_type_str(tplParams)
            }
        if tplClassName == "std::vector":
            return {
                "name": "std::vector",
                "of": parse_type_str(tplParams)
            }
        if tplClassName == "std::set":
            return {
                "name": "std::set",
                "of": parse_type_str(tplParams)
            }
        if tplClassName == "std::variant":
            variantParts = tplParams.split(", ")
            variantTypes = []
            for variantPart in variantParts:
                variantTypes.append(parse_type_str(variantPart))
            return {
                "name": "std::variant",
                "of": variantTypes
            }
        if tplClassName == "std::array":
            variantParts = tplParams.split(", ")
            if len(variantParts) != 2:
                print("FAILED TO PARSE ARRAY TYPES: " + typeStr)
                return {"name": "unknown", "str": typeStr}
            return {
                "name": "std::array",
                "of": parse_type_str(variantParts[0]),
                "size": int(variantParts[1])
            }
        if tplClassName == "std::map":
            variantParts = tplParams.split(", ")
            if len(variantParts) < 2 or len(variantParts) > 3:
                print("FAILED TO PARSE MAP TYPES: " + typeStr)
                return {"name": "unknown", "str": typeStr}

            if len(variantParts) == 2:
                return {
                    "name": "std::map",
                    "of": parse_type_str(variantParts[0]),
                    "to": parse_type_str(variantParts[1])
                }
            else:
                return {
                    "name": "std::map",
                    "of": parse_type_str(variantParts[0]),
                    "to": parse_type_str(variantParts[1]),
                    "comparator": parse_type_str(variantParts[2])
                }

        if tplClassName == "std::shared_ptr":
            return {
                "name": "std::shared_ptr",
                "of": parse_type_str(tplParams)
            }

        #return {"name": "unknown", "str": typeStr}

    if 'unnamed struct' in typeStr:
        print("WARNING:  Found unnamed struct: " + typeStr)

    return {"name": typeStr}

internal_structs = []
UNNAMED_STRUCT_DELIM = '::(unnamed struct'

def traverse(node, namespace, main_file):
    # only scan the elements of the file we parsed


    if node.kind == clang.cindex.CursorKind.STRUCT_DECL or node.kind == clang.cindex.CursorKind.CLASS_DECL:
        fullStructName = "::".join([*namespace, node.displayname])
        print("#FILE", node.location.file )
        print("REFL_TYPE(" + fullStructName + ")")

        structFields = []
        for child in node.get_children():
            if child.kind == clang.cindex.CursorKind.FIELD_DECL:
                struct_type = parse_type(child.type)
                type_str = child.type.get_canonical().spelling
                print("  REFL_FIELD(" + child.displayname + ")")
                if 'unnamed' in type_str:
                    name_tokens = type_str.split('::')
                    name_override = '::'.join(name_tokens[:-1] + [child.displayname])
                    struct_type['name'] = name_override
                    internal_structs.append(name_override)

                    structFields.append({
                        "name": child.displayname,
                        "type": struct_type,
                    })
            # replica read changes introduced duplicate get requests
            #if any(map(lambda op: op['name'] == fullStructName, opTypes)):
            #    return

            #opTypes.append({
            #    "name": fullStructName,
            #    "fields": structFields,
            #})
        print("REFL_END")

        
    if node.kind == clang.cindex.CursorKind.TYPE_ALIAS_DECL:
        fullStructName = "::".join([*namespace, node.displayname])
        if is_included_type(fullStructName, with_durability=True):
            type_ref = next((c for c in node.get_children() if c.kind == clang.cindex.CursorKind.TYPE_REF), None)
            if type_ref:
                base_request_name = type_ref.displayname.replace('struct', '').strip()
                base_request = next((op for op in opTypes if op['name'] == base_request_name), None)
                if base_request:
                    new_fields = [f for f in base_request['fields'] if f['name'] != 'durability_level']
                    new_fields.extend([
                            {"name":"persist_to", "type":{"name":"couchbase::persist_to"}},
                            {"name":"replicate_to", "type":{"name":"couchbase::replicate_to"}}
                        ])

                    opTypes.append({
                        "name": fullStructName,
                        "fields": new_fields
                    })
    if node.kind == clang.cindex.CursorKind.ENUM_DECL:
        fullEnumName = "::".join([*namespace, node.displayname])
        if is_included_type(fullEnumName):
            enumValues = []

            for child in node.get_children():
                if child.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL:
                    enumValues.append({
                        "name": child.displayname,
                        "value": child.enum_value,
                    })
            opEnums.append({
                "name": fullEnumName,
                "type": parse_type(node.enum_type),
                "values": enumValues,
            })

    if node.kind == clang.cindex.CursorKind.NAMESPACE:
        namespace = [*namespace, node.displayname]
    if node.kind == clang.cindex.CursorKind.CLASS_DECL:
        namespace = [*namespace, node.displayname]
    if node.kind == clang.cindex.CursorKind.STRUCT_DECL:
        namespace = [*namespace, node.displayname]

    for child in node.get_children():
        traverse(child, namespace, main_file)

for headerPath in fullFileList:
    print("processing " + headerPath)
    index = clang.cindex.Index.create()
    args = [
        '-std=c++17',
    ]
    
    try:
        translation_unit = index.parse(headerPath, args=args)
    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
        raise e

    # output clang compiler diagnostics information (for debugging)

    for diagnostic in translation_unit.diagnostics:
        diagnosticMsg = diagnostic.format()
        print(diagnostic)

    traverse(translation_unit.cursor, [], headerPath)

jsonData = json.dumps({
    'op_structs': opTypes,
    'op_enums': opEnums
})

f = open("bindings.json", "w")
f.write(jsonData)
f.close()
