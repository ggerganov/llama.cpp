INCLUDE(CheckCSourceRuns)

SET(AVX_CODE "
  #include <immintrin.h>
  int main()
  {
    __m256 a;
    a = _mm256_set1_ps(0);
    return 0;
  }
")

SET(AVX512_CODE "
  #include <immintrin.h>
  int main()
  {
    __m512i a = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0);
    __m512i b = a;
    __mmask64 equality_mask = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_EQ);
    return 0;
  }
")

SET(AVX2_CODE "
  #include <immintrin.h>
  int main()
  {
    __m256i a = {0};
    a = _mm256_abs_epi16(a);
    __m256i x;
    _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
    return 0;
  }
")

SET(FMA_CODE "
  #include <immintrin.h>
  int main()
  {
    __m256 acc = _mm256_setzero_ps();
    const __m256 d = _mm256_setzero_ps();
    const __m256 p = _mm256_setzero_ps();
    acc = _mm256_fmadd_ps( d, p, acc );
    return 0;
  }
")

MACRO(CHECK_SSE type flags)
  SET(__FLAG_I 1)
  SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
  FOREACH(__FLAG ${flags})
    IF(NOT ${type}_FOUND)
      SET(CMAKE_REQUIRED_FLAGS ${__FLAG})
      CHECK_C_SOURCE_RUNS("${${type}_CODE}" HAS_${type}_${__FLAG_I})
      IF(HAS_${type}_${__FLAG_I})
        SET(${type}_FOUND TRUE CACHE BOOL "${type} support")
        SET(${type}_FLAGS "${__FLAG}" CACHE STRING "${type} flags")
      ENDIF()
      MATH(EXPR __FLAG_I "${__FLAG_I}+1")
    ENDIF()
  ENDFOREACH()
  SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

  IF(NOT ${type}_FOUND)
    SET(${type}_FOUND FALSE CACHE BOOL "${type} support")
    SET(${type}_FLAGS "" CACHE STRING "${type} flags")
  ENDIF()

  MARK_AS_ADVANCED(${type}_FOUND ${type}_FLAGS)

ENDMACRO()

CHECK_SSE("AVX" " ;/arch:AVX")
IF(NOT ${AVX_FOUND})
    set(LLAMA_AVX OFF)
ELSE()
    set(LLAMA_AVX ON)
ENDIF()

CHECK_SSE("AVX2" " ;/arch:AVX2")
IF(NOT ${AVX2_FOUND})
    set(LLAMA_AVX2 OFF)
ELSE()
    set(LLAMA_AVX2 ON)
ENDIF()

CHECK_SSE("AVX512" " ;/arch:AVX512")
IF(NOT ${AVX512_FOUND})
    set(LLAMA_AVX512 OFF)
ELSE()
    set(LLAMA_AVX512 ON)
ENDIF()
