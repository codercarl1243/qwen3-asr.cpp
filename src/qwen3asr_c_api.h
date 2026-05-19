// qwen3asr_c_api.h — C ABI surface for KaiwaFlow Rust FFI.
// Exposes opaque handle + per-token streaming callback + last-error accessor.
//
// Stability: signatures locked by the kaiwaflow audio/08 spec. Do not change
// without coordinating with the Rust AsrEngine wrapper.

#ifndef QWEN3ASR_C_API_H
#define QWEN3ASR_C_API_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Qwen3AsrHandle Qwen3AsrHandle;

// Per-token callback. `text` is a null-terminated UTF-8 fragment valid only
// for the duration of the call. `user_data` is the opaque pointer passed to
// qwen3_asr_transcribe_stream and round-tripped back unchanged.
typedef void (*qwen3_asr_token_cb_t)(const char* text, void* user_data);

// Construct a handle from a local GGUF model path. Returns NULL on failure;
// caller may then read qwen3_asr_get_error(NULL) for the last global error.
Qwen3AsrHandle* qwen3_asr_create(const char* model_path);

// Blocking streaming transcription. Fires `cb` once per decoded token.
// `samples` is 16kHz mono f32 in [-1.0, 1.0]; `sample_rate` must be 16000.
// Returns true on success; false on failure (caller MUST then read
// qwen3_asr_get_error(h) — errors are not always surfaced via the return
// value alone per the audio/08 edge-case contract).
bool qwen3_asr_transcribe_stream(Qwen3AsrHandle* h,
                                 const float* samples,
                                 int32_t n_samples,
                                 int32_t sample_rate,
                                 qwen3_asr_token_cb_t cb,
                                 void* user_data);

// Release a handle. Safe on NULL.
void qwen3_asr_destroy(Qwen3AsrHandle* h);

// Last error string for the handle (or the last create-time error if h is
// NULL). Returns NULL when no error has been recorded. The pointer is owned
// by the handle and is valid until the next qwen3_asr_* call on it.
const char* qwen3_asr_get_error(Qwen3AsrHandle* h);

#ifdef __cplusplus
}
#endif

#endif // QWEN3ASR_C_API_H
