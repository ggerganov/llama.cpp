#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void nvapi_init();
void nvapi_free();

void nvapi_set_pstate(int ids[], int ids_size, int pstate);
void nvapi_set_pstate_high();
void nvapi_set_pstate_low();

#ifdef __cplusplus
}
#endif
