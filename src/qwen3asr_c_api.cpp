// qwen3asr_c_api.cpp — C ABI wrapper over qwen3_asr::Qwen3ASR.
// Bridges the C function-pointer callback to a std::function on the C++
// class, stashes per-handle last-error strings, and translates exceptions
// across the FFI boundary.

#include "qwen3asr_c_api.h"
#include "qwen3_asr.h"

#include <exception>
#include <mutex>
#include <new>
#include <string>

struct Qwen3AsrHandle {
    qwen3_asr::Qwen3ASR impl;
    std::string last_error;
};

namespace {
// Last error from create() — the caller has no handle to query, so we stash
// it globally. Cleared on the next qwen3_asr_create call.
std::mutex g_create_error_mu;
std::string g_create_error;

void set_create_error(const std::string & msg) {
    std::lock_guard<std::mutex> lock(g_create_error_mu);
    g_create_error = msg;
}

std::string take_create_error() {
    std::lock_guard<std::mutex> lock(g_create_error_mu);
    return g_create_error;
}
} // namespace

extern "C" Qwen3AsrHandle * qwen3_asr_create(const char * model_path) {
    set_create_error("");
    if (model_path == nullptr) {
        set_create_error("qwen3_asr_create: model_path is null");
        return nullptr;
    }
    Qwen3AsrHandle * h = nullptr;
    try {
        h = new Qwen3AsrHandle();
    } catch (const std::bad_alloc &) {
        set_create_error("qwen3_asr_create: out of memory");
        return nullptr;
    }
    try {
        if (!h->impl.load_model(model_path)) {
            set_create_error(h->impl.get_error());
            delete h;
            return nullptr;
        }
    } catch (const std::exception & e) {
        set_create_error(std::string("qwen3_asr_create: exception: ") + e.what());
        delete h;
        return nullptr;
    } catch (...) {
        set_create_error("qwen3_asr_create: unknown exception");
        delete h;
        return nullptr;
    }
    return h;
}

extern "C" bool qwen3_asr_transcribe_stream(Qwen3AsrHandle * h,
                                            const float * samples,
                                            int32_t n_samples,
                                            int32_t sample_rate,
                                            qwen3_asr_token_cb_t cb,
                                            void * user_data) {
    if (h == nullptr) {
        return false;
    }
    h->last_error.clear();
    if (samples == nullptr || n_samples <= 0) {
        h->last_error = "qwen3_asr_transcribe_stream: samples is null or empty";
        return false;
    }

    h->impl.set_token_callback([cb, user_data](const std::string & tok) {
        if (cb != nullptr) {
            cb(tok.c_str(), user_data);
        }
    });

    bool ok = false;
    try {
        ok = h->impl.transcribe_stream(samples, n_samples, sample_rate);
    } catch (const std::exception & e) {
        h->last_error = std::string("qwen3_asr_transcribe_stream: exception: ") + e.what();
        ok = false;
    } catch (...) {
        h->last_error = "qwen3_asr_transcribe_stream: unknown exception";
        ok = false;
    }

    h->impl.set_token_callback({});

    if (!ok && h->last_error.empty()) {
        // Per audio/08 edge-case contract, callers are required to inspect
        // qwen3_asr_get_error after a false return — make sure we always
        // have something meaningful to hand them.
        const std::string & impl_err = h->impl.get_error();
        h->last_error = impl_err.empty() ? std::string("qwen3_asr_transcribe_stream: failed (no detail)")
                                          : impl_err;
    }
    return ok;
}

extern "C" void qwen3_asr_destroy(Qwen3AsrHandle * h) {
    delete h;
}

extern "C" const char * qwen3_asr_get_error(Qwen3AsrHandle * h) {
    if (h == nullptr) {
        const std::string & e = []() -> const std::string & {
            static thread_local std::string snapshot;
            snapshot = take_create_error();
            return snapshot;
        }();
        return e.empty() ? nullptr : e.c_str();
    }
    return h->last_error.empty() ? nullptr : h->last_error.c_str();
}
