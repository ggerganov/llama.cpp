#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void nvapi_init();
void nvapi_free();

void nvapi_set_pstate(int pstate);

#ifdef __cplusplus
}
#endif
