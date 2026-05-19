// test_c_api_stub.cpp — CI-runnable smoke test for the C API surface.
// Does not require model weights: confirms the callback typedef compiles,
// qwen3_asr_destroy(NULL) is safe, qwen3_asr_get_error reports a non-NULL
// message after a bogus create, and qwen3_asr_create with a missing model
// returns NULL rather than crashing.

#include "qwen3asr_c_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void noop_token_cb(const char * /*text*/, void * /*user_data*/) {
    // The callback typedef must instantiate. No-op body is intentional.
}

static int fail(const char * msg) {
    std::fprintf(stderr, "test_c_api_stub FAIL: %s\n", msg);
    return 1;
}

int main(void) {
    // qwen3_asr_destroy on NULL must be a no-op, not a crash.
    qwen3_asr_destroy(nullptr);

    // qwen3_asr_create with NULL path returns NULL and records an error.
    Qwen3AsrHandle * null_h = qwen3_asr_create(nullptr);
    if (null_h != nullptr) {
        qwen3_asr_destroy(null_h);
        return fail("qwen3_asr_create(NULL) returned non-NULL handle");
    }
    const char * err_after_null = qwen3_asr_get_error(nullptr);
    if (err_after_null == nullptr) {
        return fail("qwen3_asr_get_error(NULL) returned NULL after a failed create");
    }
    std::fprintf(stdout, "after NULL create: %s\n", err_after_null);

    // qwen3_asr_create with a path that cannot exist must return NULL
    // (not crash, not assert). The handle is never produced, so the
    // global last-error must reflect the failure.
    const char * bogus_path = "/this/path/definitely/does/not/exist/qwen3-asr.gguf";
    Qwen3AsrHandle * bogus_h = qwen3_asr_create(bogus_path);
    if (bogus_h != nullptr) {
        qwen3_asr_destroy(bogus_h);
        return fail("qwen3_asr_create(bogus) returned non-NULL handle");
    }
    const char * err_after_bogus = qwen3_asr_get_error(nullptr);
    if (err_after_bogus == nullptr) {
        return fail("qwen3_asr_get_error(NULL) returned NULL after a bogus create");
    }
    std::fprintf(stdout, "after bogus create: %s\n", err_after_bogus);

    // Confirm the callback typedef can be assigned a function pointer with
    // the exact signature without warnings.
    qwen3_asr_token_cb_t cb = &noop_token_cb;
    if (cb == nullptr) {
        return fail("callback typedef rejected a valid function pointer");
    }

    std::fprintf(stdout, "test_c_api_stub PASS\n");
    return 0;
}
