file(REMOVE_RECURSE
  "libllava_static.a"
  "libllava_static.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/llava_static.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
