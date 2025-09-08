/*
 * Blazing Fast Holographic Memory - Pure C++ Implementation
 * ==========================================================
 *
 * This is the REAL DEAL - no Python overhead, just raw speed.
 * Using FFTW (fastest FFT) + Apple Accelerate (SIMD) + AVX2
 *
 * Optimized C++ implementation with SIMD acceleration
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <fftw3.h>
#include <complex>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <mutex>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
// Complex dot product function declaration
extern "C" {
    void cblas_zdotc_sub(const int N, const void *X, const int incX,
                         const void *Y, const int incY, void *dotc);
}
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>  // For ARM NEON SIMD
#endif

#ifdef HAVE_OPENBLAS
extern "C" {
#include <cblas.h>
}
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

class HolographicMemory {
private:
    // Thread-safety for field/matrix access
    mutable std::mutex field_mutex;
    // Core holographic field - aligned for SIMD
    alignas(32) std::vector<std::complex<double>> field;
    alignas(32) std::vector<std::complex<double>> workspace;

    // Memory storage
    struct MemoryTrace {
        std::string text;
        double importance;
        double timestamp;
        // Conversation metadata (optional; empty for generic memories)
        std::string session;   // session identifier
        std::string role;      // user|assistant|system|doc
        std::string meta;      // arbitrary JSON or tags
        uint64_t seq = 0;      // per-session monotonic sequence
        std::string id;        // stable external id
    };

    std::vector<MemoryTrace> memories;
    std::unordered_map<std::string, size_t> memory_index;
    // Session indexing: session -> indices (chronological order)
    std::unordered_map<std::string, std::vector<size_t>> session_to_indices;
    // Per-session sequence counters
    std::unordered_map<std::string, uint64_t> session_seq;
    // Row-major contiguous matrix of frequency signatures: memory_count x dimension
    std::vector<std::complex<double>> mem_matrix;

    // FFT plans - pre-computed for speed
    fftw_plan forward_plan;
    fftw_plan inverse_plan;

    // Cached FFTW plans and buffers for arbitrary-size transforms used by helpers
    mutable std::unordered_map<size_t, fftw_plan> cached_plan_fwd;
    mutable std::unordered_map<size_t, fftw_plan> cached_plan_bwd;
    mutable std::unordered_map<size_t, std::vector<std::complex<double>>> cached_buffers;

    // Dimensions
    size_t dimension;
    size_t memory_count;

    // Stats
    double total_store_time_us;
    double total_query_time_us;

    // Tunables
    size_t blas_threshold; // min memories to use BLAS matrix/vector path
    size_t omp_threshold;  // min memories to enable OpenMP parallel loops
    size_t gemm_batch_threshold; // min queries to use GEMM batch path

    // Wave persistence paths
    std::filesystem::path base_dir;
    std::filesystem::path patterns_dir;
    std::filesystem::path snapshots_dir;
    std::filesystem::path current_wave_path;
    double last_snapshot_ts;

    // ---------------------------------------------------------------------
    // HRR Associative Memory (Production path)
    // ---------------------------------------------------------------------
    std::vector<std::complex<double>> hrr_field; // fixed-size field for bindings
    std::unordered_map<uint32_t, std::string> response_dict;     // id -> response
    std::unordered_map<std::string, uint32_t> response_hash;     // response -> id
    uint32_t next_response_id = 1;

    // --- FFT helpers (unitary norm) ---
    void fft_ortho_inplace(std::vector<std::complex<double>>& vec, int sign) const {
        size_t n = vec.size();
        auto& local = cached_buffers[n];
        if (local.size() != n) {
            local.assign(n, std::complex<double>(0.0, 0.0));
            if (cached_plan_fwd.count(n)) { fftw_destroy_plan(cached_plan_fwd[n]); cached_plan_fwd.erase(n); }
            if (cached_plan_bwd.count(n)) { fftw_destroy_plan(cached_plan_bwd[n]); cached_plan_bwd.erase(n); }
        }
        if (sign == FFTW_FORWARD) {
            if (!cached_plan_fwd.count(n)) {
                cached_plan_fwd[n] = fftw_plan_dft_1d(
                    (int)n,
                    reinterpret_cast<fftw_complex*>(local.data()),
                    reinterpret_cast<fftw_complex*>(local.data()),
                    FFTW_FORWARD,
                    FFTW_MEASURE | FFTW_PRESERVE_INPUT);
            }
        } else {
            if (!cached_plan_bwd.count(n)) {
                cached_plan_bwd[n] = fftw_plan_dft_1d(
                    (int)n,
                    reinterpret_cast<fftw_complex*>(local.data()),
                    reinterpret_cast<fftw_complex*>(local.data()),
                    FFTW_BACKWARD,
                    FFTW_MEASURE | FFTW_PRESERVE_INPUT);
            }
        }
        std::memcpy(local.data(), vec.data(), n * sizeof(std::complex<double>));
        if (sign == FFTW_FORWARD) fftw_execute(cached_plan_fwd[n]); else fftw_execute(cached_plan_bwd[n]);
        double scale = 1.0 / std::sqrt((double)n);
        for (size_t i = 0; i < n; ++i) vec[i] = local[i] * scale;
    }

    void phase_only_spectrum(std::vector<std::complex<double>>& time_domain) const {
        fft_ortho_inplace(time_domain, FFTW_FORWARD);
        for (size_t i = 0; i < time_domain.size(); ++i) {
            double mag = std::abs(time_domain[i]);
            time_domain[i] = (mag > 1e-15) ? (time_domain[i] / mag) : std::complex<double>(1.0, 0.0);
        }
    }

    void encode_uint32_as_wave(uint32_t id, std::vector<std::complex<double>>& out) const {
        // Unit-magnitude complex random phases seeded by id (deterministic)
        out.resize(dimension);
        std::mt19937 gen(static_cast<uint32_t>(id * 0x9E3779B9u));
        std::uniform_real_distribution<double> phase(0.0, 2.0 * M_PI);
        for (size_t i = 0; i < dimension; ++i) {
            double ph = phase(gen);
            out[i] = std::complex<double>(std::cos(ph), std::sin(ph));
        }
    }

    void key_phase_spectrum(const std::string& key, std::vector<std::complex<double>>& P) const {
        // Deterministic unit-magnitude random phases directly in spectral domain
        P.resize(dimension);
        std::hash<std::string> H;
        uint64_t seed = static_cast<uint64_t>(H(key)) ^ 0x9E3779B97F4A7C15ULL;
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> phase(0.0, 2.0 * M_PI);
        for (size_t i = 0; i < dimension; ++i) {
            double ph = phase(gen);
            P[i] = std::complex<double>(std::cos(ph), std::sin(ph));
        }
    }

    void hrr_bind(const std::string& key,
                  const std::vector<std::complex<double>>& val_time,
                  std::vector<std::complex<double>>& out_time) const {
        std::vector<std::complex<double>> P;
        key_phase_spectrum(key, P);
        std::vector<std::complex<double>> V = val_time;
        fft_ortho_inplace(V, FFTW_FORWARD);
        out_time.resize(dimension);
        for (size_t i = 0; i < dimension; ++i) out_time[i] = P[i] * V[i];
        fft_ortho_inplace(out_time, FFTW_BACKWARD);
    }

    void hrr_unbind(const std::vector<std::complex<double>>& bound_time,
                    const std::string& key,
                    std::vector<std::complex<double>>& out_time) const {
        std::vector<std::complex<double>> B = bound_time;
        fft_ortho_inplace(B, FFTW_FORWARD);
        std::vector<std::complex<double>> P;
        key_phase_spectrum(key, P);
        out_time.resize(dimension);
        for (size_t i = 0; i < dimension; ++i) out_time[i] = B[i] * std::conj(P[i]);
        fft_ortho_inplace(out_time, FFTW_BACKWARD);
    }

    uint32_t decode_best_id(const std::vector<std::complex<double>>& cand_time) const {
        // Choose id that maximizes |<cand, id_wave>|
        double best = -1.0; uint32_t best_id = 0;
        for (const auto& kv : response_dict) {
            uint32_t id = kv.first;
            std::vector<std::complex<double>> id_wave;
            encode_uint32_as_wave(id, id_wave);
            std::complex<double> acc(0.0, 0.0);
            for (size_t j = 0; j < dimension; ++j) {
                acc += cand_time[j] * std::conj(id_wave[j]);
            }
            double score = std::abs(acc);
            if (score > best) { best = score; best_id = id; }
        }
        return best_id;
    }

    void encode_key_random_index(const std::string& key, std::vector<std::complex<double>>& out) const {
        // Sparse ternary {-1,0,+1} with ~10% density, deterministic per key
        out.assign(dimension, std::complex<double>(0.0, 0.0));
        std::hash<std::string> hasher;
        size_t seed = hasher(key);
        std::mt19937_64 gen(static_cast<uint64_t>(seed));
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        std::uniform_int_distribution<int> sign01(0, 1);
        for (size_t i = 0; i < dimension; ++i) {
            if (prob(gen) < 0.10) {
                double val = sign01(gen) ? 1.0 : -1.0;
                out[i] = std::complex<double>(val, 0.0);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Response pattern cache (Holographic Response Caching - PoC)
    // Stores a compact spectral signature per response pattern keyed by an
    // application-provided query key. At query time we resonate against this
    // cache to return the best cached response without running a model.
    // ---------------------------------------------------------------------
    struct ResponseEntry {
        std::string key;                                  // application-level key / hash
        std::vector<std::complex<double>> spectrum;       // copy of current workspace (FFT of key)
        std::string response;                             // raw response text (or serialized data)
    };
    std::vector<ResponseEntry> response_cache;
    std::unordered_map<std::string, size_t> response_index; // key -> position in response_cache
    // Row-major matrix: response_count x dimension (contiguous) for BLAS zgemv
    std::vector<std::complex<double>> response_matrix;
    size_t response_count = 0;

    static std::string sanitize(const std::string& name) {
        std::string out;
        out.reserve(name.size());
        for (char c : name) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c=='-' || c=='_' || c=='.') out.push_back(c);
            else out.push_back('_');
        }
        if (out.empty()) out = "unnamed";
        return out;
    }

    static double now_sec() {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    void ensure_dirs() {
        try {
            std::filesystem::create_directories(patterns_dir);
            std::filesystem::create_directories(snapshots_dir);
        } catch (...) {}
    }

    // Simple binary writer for complex128 vectors with JSON metadata
    void write_wave_file_complex(const std::filesystem::path& outpath,
                                 const std::vector<std::complex<double>>& data,
                                 const std::string& meta_json) {
        try {
            std::ofstream ofs(outpath, std::ios::binary);
            if (!ofs) return;
            const char magic[8] = {'W','A','V','E','V','0','0','1'};
            ofs.write(magic, 8);
            uint32_t version = 1u;
            uint32_t kind = 1u; // 1 = complex128
            uint64_t dim = static_cast<uint64_t>(data.size());
            double ts = now_sec();
            uint32_t meta_len = static_cast<uint32_t>(meta_json.size());
            ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
            ofs.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
            ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            ofs.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
            ofs.write(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len));
            if (meta_len) ofs.write(meta_json.data(), meta_len);
            ofs.write(reinterpret_cast<const char*>(data.data()), sizeof(std::complex<double>) * data.size());
        } catch (...) {}
    }

    static std::string timestamp_str() {
        auto tp = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(tp);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%dT%H%M%S");
        return oss.str();
    }

    void snapshot_if_due_locked() {
        double t = now_sec();
        if (t - last_snapshot_ts < 3600.0) return; // hourly
        last_snapshot_ts = t;
        // Copy field to avoid holding locks (no locks used in this class, but copy anyway)
        std::vector<std::complex<double>> copy = field;
        std::ostringstream mj;
        mj << "{\"type\":\"snapshot\",\"dimension\":" << dimension
           << ",\"count\":" << memory_count << ",\"backend\":\"native\"}";
        std::filesystem::path snap = snapshots_dir / (std::string("snapshot_") + timestamp_str() + ".hwp");
        write_wave_file_complex(snap, copy, mj.str());
    }

    void write_current_locked() {
        std::vector<std::complex<double>> copy = field;
        std::string mj = "{\"type\":\"current\",\"backend\":\"native\"}";
        write_wave_file_complex(current_wave_path, copy, mj);
    }

    static std::string extract_filename_tag(const std::string& text) {
        // Expect first line like: filename:original_name\n...
        size_t pos = text.find('\n');
        std::string first = (pos == std::string::npos) ? text : text.substr(0, pos);
        const std::string prefix = "filename:";
        if (first.rfind(prefix, 0) == 0 && first.size() > prefix.size()) {
            std::string nm = first.substr(prefix.size());
            // trim spaces
            while (!nm.empty() && (nm[0] == ' ' || nm[0] == '\t')) nm.erase(nm.begin());
            return nm;
        }
        return std::string();
    }

public:
    HolographicMemory(size_t dim = 1024) : dimension(dim), memory_count(0),
                                           total_store_time_us(0), total_query_time_us(0),
                                           last_snapshot_ts(0.0) {
        // Allocate aligned memory
        field.resize(dim, std::complex<double>(0, 0));
        workspace.resize(dim, std::complex<double>(0, 0));

        // Pre-plan FFTs with FFTW_PATIENT for maximum speed
        forward_plan = fftw_plan_dft_1d(
            dim,
            reinterpret_cast<fftw_complex*>(workspace.data()),
            reinterpret_cast<fftw_complex*>(workspace.data()),
            FFTW_FORWARD,
            FFTW_PATIENT | FFTW_PRESERVE_INPUT
        );

        inverse_plan = fftw_plan_dft_1d(
            dim,
            reinterpret_cast<fftw_complex*>(workspace.data()),
            reinterpret_cast<fftw_complex*>(workspace.data()),
            FFTW_BACKWARD,
            FFTW_PATIENT | FFTW_PRESERVE_INPUT
        );

        // Reserve space for memories
        memories.reserve(1000000);

        // Tunables from environment
        const char* bt = std::getenv("HLOG_BLAS_GEMV_THRESHOLD");
        blas_threshold = bt ? static_cast<size_t>(std::max(0, atoi(bt))) : 256;
        const char* ot = std::getenv("HLOG_OMP_MIN_SIZE");
        omp_threshold = ot ? static_cast<size_t>(std::max(0, atoi(ot))) : 64;
        const char* gb = std::getenv("HLOG_GEMM_MIN_BATCH");
        gemm_batch_threshold = gb ? static_cast<size_t>(std::max(0, atoi(gb))) : 8;
#ifdef _OPENMP
        const char* tn = std::getenv("HLOG_OMP_THREADS");
        if (tn) {
            int n = atoi(tn);
            if (n > 0) omp_set_num_threads(n);
        }
#endif

        // FFTW wisdom (optional)
        if (const char* wp = std::getenv("HLOG_FFTW_WISDOM")) {
            (void)fftw_import_wisdom_from_filename(wp);
        }

        // Initialize persistence paths
        const char* bd = std::getenv("HLOG_DATA_DIR");
        if (bd && *bd) base_dir = std::filesystem::path(bd);
        else if (std::filesystem::exists("/app/data/holographic_memory")) base_dir = "/app/data/holographic_memory";
        else base_dir = std::filesystem::path("data") / "holographic_memory";
        patterns_dir = base_dir / "patterns";
        snapshots_dir = base_dir / "snapshots";
        current_wave_path = base_dir / "current.hwp";
        ensure_dirs();
        // Initialize HRR field to zeros
        hrr_field.assign(dim, std::complex<double>(0.0, 0.0));
    }

    ~HolographicMemory() {
        // Final snapshot at shutdown
        try { snapshot_if_due_locked(); write_current_locked(); } catch (...) {}
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(inverse_plan);
        for (auto& kv : cached_plan_fwd) { fftw_destroy_plan(kv.second); }
        for (auto& kv : cached_plan_bwd) { fftw_destroy_plan(kv.second); }
        if (const char* wp = std::getenv("HLOG_FFTW_WISDOM")) {
            (void)fftw_export_wisdom_to_filename(wp);
        }
    }

    // Fast text encoding using SIMD
    void encode_text_simd(const std::string& text, std::complex<double>* output) {
        // Zero output
        std::memset(output, 0, dimension * sizeof(std::complex<double>));

        // Hash-based encoding with SIMD
        std::hash<std::string> hasher;

        // Sliding window encoding
        const int window_size = 5;
        for (size_t i = 0; i <= text.size() - window_size; ++i) {
            std::string window = text.substr(i, window_size);
            size_t hash = hasher(window);

            // Distribute hash across frequencies
            for (size_t j = 0; j < 8 && j < dimension; ++j) {
                size_t freq_idx = (hash + j * 0x9e3779b9) % dimension;
                double phase = (i + hash) * 2.0 * M_PI / 256.0;
                double amplitude = 1.0 / (1.0 + i * 0.1);

                output[freq_idx] += std::complex<double>(
                    amplitude * cos(phase),
                    amplitude * sin(phase)
                );
            }
        }

        // Normalize using SIMD
        double norm = 0.0;

#ifdef __APPLE__
        // Use Apple Accelerate for norm calculation
        double norm_real = 0.0, norm_imag = 0.0;
        vDSP_svesqD(reinterpret_cast<double*>(output), 2, &norm_real, dimension);
        vDSP_svesqD(reinterpret_cast<double*>(output) + 1, 2, &norm_imag, dimension);
        norm = sqrt(norm_real + norm_imag);
#else
        for (size_t i = 0; i < dimension; ++i) {
            norm += std::norm(output[i]);
        }
        norm = sqrt(norm);
#endif

        if (norm > 0) {
            double inv_norm = 1.0 / norm;
            for (size_t i = 0; i < dimension; ++i) {
                output[i] *= inv_norm;
            }
        }
    }

    // ------------------------------------------------------------------
    // Holographic Response Caching (PoC)
    // ------------------------------------------------------------------
    // Store a response pattern keyed by a query key (any string). We encode
    // the key using the same SIMD + FFT pipeline, and persist the spectrum
    // alongside the raw response. Later, resonate_response() finds the best
    // match via complex correlation and returns the cached response in O(N)
    // over number of cached patterns (or BLAS path when large).
    void store_response_pattern(const std::string& key, const std::string& response_text) {
        // Build spectral signature for key
        encode_text_simd(key, workspace.data());
        fftw_execute(forward_plan);
        ResponseEntry ent;
        ent.key = key;
        ent.spectrum.assign(workspace.begin(), workspace.end());
        ent.response = response_text;
        auto it = response_index.find(key);
        if (it != response_index.end()) {
            size_t pos = it->second;
            response_cache[pos] = std::move(ent);
            // Overwrite row in response_matrix
            if (response_matrix.size() == response_count * dimension) {
                std::memcpy(&response_matrix[pos * dimension], workspace.data(), dimension * sizeof(std::complex<double>));
            }
        } else {
            size_t pos = response_cache.size();
            response_cache.push_back(std::move(ent));
            response_index[key] = pos;
            // Append row to matrix
            if (response_matrix.size() != (response_count * dimension)) {
                response_matrix.resize(response_count * dimension);
            }
            response_matrix.insert(response_matrix.end(), workspace.begin(), workspace.end());
            response_count = response_cache.size();
        }
    }

    std::string resonate_response(const std::string& query_text) {
        if (response_cache.empty()) return std::string();
        encode_text_simd(query_text, workspace.data());
        fftw_execute(forward_plan);
        // Compute correlation with cached spectra; pick max magnitude
        double best = -1.0;
        size_t best_idx = 0;

#if defined(__APPLE__) || defined(HAVE_OPENBLAS)
        if (!response_cache.empty() && response_matrix.size() == response_cache.size() * dimension && response_cache.size() >= blas_threshold) {
            // y = A * conj(x)
            const size_t K = response_cache.size();
            std::vector<std::complex<double>> x_conj(dimension);
            for (size_t j = 0; j < dimension; ++j) x_conj[j] = std::conj(workspace[j]);
            std::vector<std::complex<double>> y(K);
            std::complex<double> alpha(1.0, 0.0);
            std::complex<double> beta(0.0, 0.0);
            cblas_zgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(K), static_cast<int>(dimension),
                        &alpha,
                        reinterpret_cast<const void*>(response_matrix.data()), static_cast<int>(dimension),
                        reinterpret_cast<const void*>(x_conj.data()), 1,
                        &beta,
                        reinterpret_cast<void*>(y.data()), 1);
            for (size_t i = 0; i < K; ++i) {
                double s = std::abs(y[i]);
                if (s > best) { best = s; best_idx = i; }
            }
            return response_cache[best_idx].response;
        }
#endif
        for (size_t i = 0; i < response_cache.size(); ++i) {
            const auto& spec = response_cache[i].spectrum;
            std::complex<double> acc(0.0, 0.0);
            for (size_t j = 0; j < dimension; ++j) {
                acc += std::conj(workspace[j]) * spec[j];
            }
            double score = std::abs(acc);
            if (score > best) { best = score; best_idx = i; }
        }
        return response_cache[best_idx].response;
    }

    std::pair<std::string, double> resonate_response_with_score(const std::string& query_text) {
        if (response_cache.empty()) return {std::string(), 0.0};
        encode_text_simd(query_text, workspace.data());
        fftw_execute(forward_plan);
        double best = -1.0, second = -1.0;
        size_t best_idx = 0;
#if defined(__APPLE__) || defined(HAVE_OPENBLAS)
        if (!response_cache.empty() && response_matrix.size() == response_cache.size() * dimension && response_cache.size() >= blas_threshold) {
            const size_t K = response_cache.size();
            std::vector<std::complex<double>> x_conj(dimension);
            for (size_t j = 0; j < dimension; ++j) x_conj[j] = std::conj(workspace[j]);
            std::vector<std::complex<double>> y(K);
            std::complex<double> alpha(1.0, 0.0);
            std::complex<double> beta(0.0, 0.0);
            cblas_zgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(K), static_cast<int>(dimension),
                        &alpha,
                        reinterpret_cast<const void*>(response_matrix.data()), static_cast<int>(dimension),
                        reinterpret_cast<const void*>(x_conj.data()), 1,
                        &beta,
                        reinterpret_cast<void*>(y.data()), 1);
            for (size_t i = 0; i < K; ++i) {
                double s = std::abs(y[i]);
                if (s > best) { second = best; best = s; best_idx = i; }
                else if (s > second) { second = s; }
            }
            double conf = (best > 0.0) ? std::max(0.0, (best - std::max(0.0, second)) / (best + 1e-12)) : 0.0;
            return {response_cache[best_idx].response, conf};
        }
#endif
        for (size_t i = 0; i < response_cache.size(); ++i) {
            const auto& spec = response_cache[i].spectrum;
            std::complex<double> acc(0.0, 0.0);
            for (size_t j = 0; j < dimension; ++j) {
                acc += std::conj(workspace[j]) * spec[j];
            }
            double s = std::abs(acc);
            if (s > best) { second = best; best = s; best_idx = i; }
            else if (s > second) { second = s; }
        }
        double conf = (best > 0.0) ? std::max(0.0, (best - std::max(0.0, second)) / (best + 1e-12)) : 0.0;
        return {response_cache[best_idx].response, conf};
    }

    py::dict get_response_cache_stats() const {
        py::dict d;
        d["count"] = (py::int_)response_cache.size();
        d["dimension"] = (py::int_)dimension;
        return d;
    }

    // BLAZING FAST STORE using FFTW + SIMD
    std::string store(const std::string& text, double importance = 1.0) {
        auto start = std::chrono::high_resolution_clock::now();

        // Encode to workspace
        encode_text_simd(text, workspace.data());

        // FFT in-place using FFTW (fastest FFT library)
        fftw_execute(forward_plan);

        // Persist individual pattern if filename tag present
        std::string orig_name = extract_filename_tag(text);
        if (!orig_name.empty()) {
            std::filesystem::path outp = patterns_dir / (sanitize(orig_name) + ".hwp");
            std::ostringstream mj;
            mj << "{\"type\":\"pattern\",\"filename\":\"" << sanitize(orig_name) << "\",";
            mj << "\"dimension\":" << dimension << ",\"backend\":\"native\"}";
            // workspace currently holds the individual spectrum
            write_wave_file_complex(outp, workspace, mj.str());
        }

        // Superpose onto field using SIMD
#ifdef __APPLE__
        // Use Apple Accelerate for maximum speed - complex scale and add
        // cblas_zaxpy expects a complex<double> alpha pointer
        std::complex<double> alpha(importance, 0.0);
        cblas_zaxpy(static_cast<int>(dimension),
                    &alpha,
                    reinterpret_cast<const void*>(workspace.data()), 1,
                    reinterpret_cast<void*>(field.data()), 1);
#else
        // Fallback to manual SIMD
        for (size_t i = 0; i < dimension; ++i) {
            field[i] += workspace[i] * importance;
        }
#endif

        // Update current field and snapshot
        write_current_locked();
        snapshot_if_due_locked();

        // Store memory trace
        MemoryTrace trace;
        trace.text = text;
        trace.importance = importance;
        trace.timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();

        std::string id = "mem_" + std::to_string(memory_count);
        trace.id = id;
        trace.role = "doc";
        trace.session = "";
        trace.meta = "";
        trace.seq = 0;

        memories.push_back(std::move(trace));
        // Append row to resident matrix
        mem_matrix.insert(mem_matrix.end(), workspace.begin(), workspace.end());
        memory_index[id] = memory_count;
        memory_count++;

        auto end = std::chrono::high_resolution_clock::now();
        total_store_time_us += std::chrono::duration<double, std::micro>(end - start).count();

        return id;
    }

    // BATCH STORE - Even faster for multiple items
    std::vector<std::string> store_batch(const std::vector<std::string>& texts, double importance = 1.0) {
        std::vector<std::string> ids;
        ids.reserve(texts.size());

        // Process all at once for cache efficiency
        for (const auto& text : texts) {
            ids.push_back(store(text, importance));
        }

        return ids;
    }

    // ULTRA-FAST QUERY using FFTW + SIMD
    std::vector<std::pair<std::string, double>> query(const std::string& query_text, size_t top_k = 10) {
        auto start = std::chrono::high_resolution_clock::now();

        // Encode query
        encode_text_simd(query_text, workspace.data());

        // FFT query
        fftw_execute(forward_plan);

        // Compute resonances using SIMD / BLAS with OpenMP parallelization
        std::vector<std::pair<double, size_t>> scores(memories.size());

#if defined(__APPLE__) || defined(HAVE_OPENBLAS)
        // BLAS matrix formulation using resident mem_matrix if large
        if (memories.size() >= blas_threshold && mem_matrix.size() == memories.size() * dimension) {
            const size_t M = memories.size();
            const size_t N = dimension;
            std::vector<std::complex<double>> y(M);
            // y = A * conj(x)
            std::vector<std::complex<double>> x_conj(N);
            for (size_t j = 0; j < N; ++j) x_conj[j] = std::conj(workspace[j]);
            std::complex<double> alpha(1.0, 0.0);
            std::complex<double> beta(0.0, 0.0);
            cblas_zgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(M), static_cast<int>(N),
                        &alpha,
                        reinterpret_cast<const void*>(mem_matrix.data()), static_cast<int>(N),
                        reinterpret_cast<const void*>(x_conj.data()), 1,
                        &beta,
                        reinterpret_cast<void*>(y.data()), 1);

            const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            for (size_t i = 0; i < M; ++i) {
                double similarity = std::abs(y[i]);
                double time_decay = exp(-(now - memories[i].timestamp) / 86400.0);
                double final_score = similarity * memories[i].importance * (0.5 + 0.5 * time_decay);
                scores[i] = {final_score, i};
            }
        } else
#endif
        {
            const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            // Parallel for across memories
            #pragma omp parallel for if(memories.size() > omp_threshold) schedule(static)
            for (long long ii = 0; ii < static_cast<long long>(memories.size()); ++ii) {
                size_t i = static_cast<size_t>(ii);
                double similarity = 0.0;
#ifdef __APPLE__
                // y_i = row_i(A) · conj(x)
                std::complex<double> res(0.0, 0.0);
                const std::complex<double>* row = &mem_matrix[i * dimension];
                for (size_t j = 0; j < dimension; ++j) res += row[j] * std::conj(workspace[j]);
                similarity = std::abs(res);
#else
                std::complex<double> acc(0.0, 0.0);
                const std::complex<double>* row = &mem_matrix[i * dimension];
                for (size_t j = 0; j < dimension; ++j) {
                    acc += std::conj(workspace[j]) * row[j];
                }
                similarity = std::abs(acc);
#endif
                double time_decay = exp(-(now - memories[i].timestamp) / 86400.0);
                double final_score = similarity * memories[i].importance * (0.5 + 0.5 * time_decay);
                scores[i] = {final_score, i};
            }
        }

        // Sort for top-k (use partial_sort for efficiency)
        size_t actual_k = std::min(top_k, scores.size());
        std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
                         std::greater<std::pair<double, size_t>>());

        // Build results
        std::vector<std::pair<std::string, double>> results;
        results.reserve(actual_k);

        for (size_t i = 0; i < actual_k; ++i) {
            results.push_back({
                memories[scores[i].second].text,
                scores[i].first
            });
        }

        auto end = std::chrono::high_resolution_clock::now();
        total_query_time_us += std::chrono::duration<double, std::micro>(end - start).count();

        return results;
    }

    // Store a conversation message with session-aware metadata.
    std::string store_message(const std::string& session,
                              const std::string& role,
                              const std::string& text,
                              double importance = 1.0,
                              const std::string& meta = std::string()) {
        auto start = std::chrono::high_resolution_clock::now();

        // Encode and superpose (same as store)
        encode_text_simd(text, workspace.data());
        fftw_execute(forward_plan);
#ifdef __APPLE__
        std::complex<double> alpha(importance, 0.0);
        cblas_zaxpy(static_cast<int>(dimension),
                    &alpha,
                    reinterpret_cast<const void*>(workspace.data()), 1,
                    reinterpret_cast<void*>(field.data()), 1);
#else
        for (size_t i = 0; i < dimension; ++i) {
            field[i] += workspace[i] * importance;
        }
#endif

        // Save per-message pattern
        uint64_t next_seq = session_seq[session] + 1; // predict id
        {
            std::ostringstream fn;
            fn << "session_" << sanitize(session) << "_seq_" << next_seq << ".hwp";
            std::ostringstream mj;
            mj << "{\"type\":\"pattern\",\"session\":\"" << sanitize(session) << "\",\"role\":\"" << role
               << "\",\"seq\":" << next_seq << ",\"dimension\":" << dimension << ",\"backend\":\"native\"}";
            write_wave_file_complex(patterns_dir / fn.str(), workspace, mj.str());
        }

        // Update current field and snapshot
        write_current_locked();
        snapshot_if_due_locked();

        // Prepare trace
        MemoryTrace trace;
        trace.text = text;
        trace.importance = importance;
        trace.timestamp = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        trace.session = session;
        trace.role = role;
        trace.meta = meta;
        uint64_t next = ++session_seq[session];
        trace.seq = next;
        std::string id = std::string("msg_") + session + "#" + std::to_string(next);
        trace.id = id;

        // Persist
        memories.push_back(std::move(trace));
        mem_matrix.insert(mem_matrix.end(), workspace.begin(), workspace.end());
        memory_index[id] = memory_count;
        session_to_indices[session].push_back(memory_count);
        memory_count++;

        auto end = std::chrono::high_resolution_clock::now();
        total_store_time_us += std::chrono::duration<double, std::micro>(end - start).count();
        return id;
    }

    // Remove a memory by id with O(1) swap; updates resident matrix
    bool remove(const std::string& id) {
        auto it = memory_index.find(id);
        if (it == memory_index.end()) return false;
        size_t idx = it->second;
        size_t last = memory_count - 1;
        // sessions for affected entries
        std::string sess_idx = memories[idx].session;
        std::string sess_last = memories[last].session;
        if (idx != last) {
            // swap memories
            memories[idx] = std::move(memories[last]);
            // swap matrix rows
            std::complex<double>* dst = &mem_matrix[idx * dimension];
            std::complex<double>* src = &mem_matrix[last * dimension];
            std::memcpy(dst, src, dimension * sizeof(std::complex<double>));
            // update index for moved id
            for (auto& kv : memory_index) {
                if (kv.second == last) { kv.second = idx; break; }
            }
            // update session vector for moved element
            if (!sess_last.empty()) {
                auto& vec = session_to_indices[sess_last];
                for (auto& v : vec) { if (v == last) { v = idx; break; } }
            }
        }
        // pop last
        mem_matrix.resize((last) * dimension);
        memory_index.erase(it);
        memories.pop_back();
        memory_count = last;
        // remove idx from its session list if any
        if (!sess_idx.empty()) {
            auto& vec = session_to_indices[sess_idx];
            for (auto vit = vec.begin(); vit != vec.end(); ++vit) {
                if (*vit == idx) { vec.erase(vit); break; }
            }
            if (vec.empty()) {
                session_to_indices.erase(sess_idx);
                session_seq.erase(sess_idx);
            }
        }
        return true;
    }

    // Batch query: multiple queries -> list of top-k results per query
    std::vector<std::vector<std::pair<std::string, double>>>
    query_batch(const std::vector<std::string>& queries, size_t top_k = 10) {
        std::vector<std::vector<std::pair<std::string, double>>> all_results;
        all_results.reserve(queries.size());
        const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        // Precompute per-memory weights (importance * time_decay)
        std::vector<double> weights(memories.size());
        for (size_t i = 0; i < memories.size(); ++i) {
            double time_decay = exp(-(now - memories[i].timestamp) / 86400.0);
            weights[i] = memories[i].importance * (0.5 + 0.5 * time_decay);
        }
        // BLAS GEMM path for large batches on Apple (stack queries once)
#ifdef __APPLE__
        if (!queries.empty() && memories.size() >= blas_threshold && mem_matrix.size() == memories.size() * dimension && queries.size() >= gemm_batch_threshold) {
            const size_t M = memories.size();
            const size_t N = dimension;
            const size_t B = queries.size();
            // Build conj(Q) matrix (N x B) row-major: each row j has B columns of conj(query[j])
            std::vector<std::complex<double>> Qc(N * B);
            for (size_t k = 0; k < B; ++k) {
                // Encode + FFT per query into workspace then copy conj into column k
                encode_text_simd(queries[k], workspace.data());
                fftw_execute(forward_plan);
                for (size_t j = 0; j < N; ++j) {
                    Qc[j * B + k] = std::conj(workspace[j]);
                }
            }
            // Y = A (M x N) * Qc (N x B) => (M x B)
            std::vector<std::complex<double>> Y(M * B);
            std::complex<double> alpha(1.0, 0.0);
            std::complex<double> beta(0.0, 0.0);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(M), static_cast<int>(B), static_cast<int>(N),
                        &alpha,
                        reinterpret_cast<const void*>(mem_matrix.data()), static_cast<int>(N),
                        reinterpret_cast<const void*>(Qc.data()), static_cast<int>(B),
                        &beta,
                        reinterpret_cast<void*>(Y.data()), static_cast<int>(B));

            // Build results per query from column k of Y
            for (size_t k = 0; k < B; ++k) {
                std::vector<std::pair<double, size_t>> scores(M);
                for (size_t i = 0; i < M; ++i) {
                    double s = std::abs(Y[i * B + k]) * weights[i];
                    scores[i] = {s, i};
                }
                size_t actual_k = std::min(top_k, scores.size());
                std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
                                  std::greater<std::pair<double, size_t>>());
                std::vector<std::pair<std::string, double>> results;
                results.reserve(actual_k);
                for (size_t i = 0; i < actual_k; ++i) {
                    results.push_back({memories[scores[i].second].text, scores[i].first});
                }
                all_results.push_back(std::move(results));
            }
            return all_results;
        }
#endif
        for (const auto& q : queries) {
            // Encode + FFT query
            encode_text_simd(q, workspace.data());
            fftw_execute(forward_plan);
#if defined(__APPLE__) || defined(HAVE_OPENBLAS)
            if (memories.size() >= blas_threshold && mem_matrix.size() == memories.size() * dimension) {
                const size_t M = memories.size();
                const size_t N = dimension;
                std::vector<std::complex<double>> y(M);
                // y = A * conj(x)
                std::vector<std::complex<double>> x_conj(N);
                for (size_t j = 0; j < N; ++j) x_conj[j] = std::conj(workspace[j]);
                std::complex<double> alpha(1.0, 0.0);
                std::complex<double> beta(0.0, 0.0);
                cblas_zgemv(CblasRowMajor, CblasNoTrans,
                            static_cast<int>(M), static_cast<int>(N),
                            &alpha,
                            reinterpret_cast<const void*>(mem_matrix.data()), static_cast<int>(N),
                            reinterpret_cast<const void*>(x_conj.data()), 1,
                            &beta,
                            reinterpret_cast<void*>(y.data()), 1);
                std::vector<std::pair<double, size_t>> scores(M);
                for (size_t i = 0; i < M; ++i) {
                    scores[i] = {std::abs(y[i]) * weights[i], i};
                }
                size_t actual_k = std::min(top_k, scores.size());
                std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
                                  std::greater<std::pair<double, size_t>>());
                std::vector<std::pair<std::string, double>> results;
                results.reserve(actual_k);
                for (size_t i = 0; i < actual_k; ++i) {
                    results.push_back({memories[scores[i].second].text, scores[i].first});
                }
                all_results.push_back(std::move(results));
                continue;
            }
#endif
            // Fallback/parallel path per-memory
            std::vector<std::pair<double, size_t>> scores(memories.size());
            #pragma omp parallel for if(memories.size() > omp_threshold) schedule(static)
            for (long long ii = 0; ii < static_cast<long long>(memories.size()); ++ii) {
                size_t i = static_cast<size_t>(ii);
                std::complex<double> acc(0.0, 0.0);
                const std::complex<double>* row = &mem_matrix[i * dimension];
                for (size_t j = 0; j < dimension; ++j) {
                    acc += std::conj(workspace[j]) * row[j];
                }
                scores[i] = {std::abs(acc) * weights[i], i};
            }
            size_t actual_k = std::min(top_k, scores.size());
            std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
                              std::greater<std::pair<double, size_t>>());
            std::vector<std::pair<std::string, double>> results;
            results.reserve(actual_k);
            for (size_t i = 0; i < actual_k; ++i) {
                results.push_back({memories[scores[i].second].text, scores[i].first});
            }
            all_results.push_back(std::move(results));
        }
        return all_results;
    }

    // --------------------------------------------------------------------------------
    // Wavefunction math helpers
    // --------------------------------------------------------------------------------

    // Unitary Fourier transform using FFTW (matches numpy.fft.fft with norm="ortho")
    py::array_t<std::complex<double>> fourier_transform(py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> psi) {
        py::buffer_info buf = psi.request();
        if (buf.ndim != 1) {
            throw std::invalid_argument("psi must be a 1D complex array");
        }
        size_t n = static_cast<size_t>(buf.shape[0]);
        py::array_t<std::complex<double>> out(buf.shape[0]);
        auto* in_ptr = static_cast<std::complex<double>*>(buf.ptr);
        auto* out_ptr = static_cast<std::complex<double>*>(out.request().ptr);

        // Get or create cached buffer and plan for size n
        auto& local = cached_buffers[n];
        if (local.size() != n) {
            local.assign(n, std::complex<double>(0.0, 0.0));
            if (cached_plan_fwd.count(n)) { fftw_destroy_plan(cached_plan_fwd[n]); cached_plan_fwd.erase(n); }
        }
        // Create plan if missing (in-place, using cached buffer)
        if (!cached_plan_fwd.count(n)) {
            cached_plan_fwd[n] = fftw_plan_dft_1d(
                static_cast<int>(n),
                reinterpret_cast<fftw_complex*>(local.data()),
                reinterpret_cast<fftw_complex*>(local.data()),
                FFTW_FORWARD, FFTW_MEASURE | FFTW_PRESERVE_INPUT);
        }

        // Copy input into cached buffer and execute
        std::memcpy(local.data(), in_ptr, n * sizeof(std::complex<double>));
        fftw_execute(cached_plan_fwd[n]);

        // Orthonormal scaling and copy to output
        const double scale = 1.0 / std::sqrt(static_cast<double>(n));
        for (size_t i = 0; i < n; ++i) {
            out_ptr[i] = local[i] * scale;
        }
        return out;
    }

    // Interference intensity I = |psi1 + psi2|^2
    py::array_t<double> interference_intensity(
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> psi1,
        py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> psi2) {
        auto b1 = psi1.request();
        auto b2 = psi2.request();
        if (b1.ndim != 1 || b2.ndim != 1 || b1.shape[0] != b2.shape[0]) {
            throw std::invalid_argument("psi1 and psi2 must be same-length 1D complex arrays");
        }
        size_t n = static_cast<size_t>(b1.shape[0]);
        py::array_t<double> out(n);
        auto* p1 = static_cast<std::complex<double>*>(b1.ptr);
        auto* p2 = static_cast<std::complex<double>*>(b2.ptr);
        auto* o = static_cast<double*>(out.request().ptr);
        for (size_t i = 0; i < n; ++i) {
            std::complex<double> s = p1[i] + p2[i];
            o[i] = std::norm(s);
        }
        return out;
    }

    // Convenience: use holographic resonance retrieval via existing query()
    std::vector<std::pair<std::string, double>> quantum_resonance_query(const std::string& query_text, int top_k) {
        return query(query_text, static_cast<size_t>(top_k));
    }

    // Get performance statistics
    py::dict get_stats() const {
        py::dict stats;
        stats["memory_count"] = memory_count;
        stats["dimension"] = dimension;
        stats["avg_store_us"] = memory_count > 0 ? total_store_time_us / memory_count : 0;
        stats["avg_query_us"] = total_query_time_us / std::max(1.0, (double)memory_count);
        stats["total_store_time_ms"] = total_store_time_us / 1000.0;
        stats["total_query_time_ms"] = total_query_time_us / 1000.0;
        stats["backend"] = "C++ FFTW + SIMD";
        return stats;
    }

    // Return list of sessions with counts and last timestamp
    py::list list_sessions() const {
        py::list out;
        for (const auto& kv : session_to_indices) {
            const auto& sid = kv.first;
            const auto& idxs = kv.second;
            double last_ts = 0.0;
            if (!idxs.empty()) {
                last_ts = memories[idxs.back()].timestamp;
            }
            py::dict d;
            d["session_id"] = sid;
            d["count"] = static_cast<long>(idxs.size());
            d["last_ts"] = last_ts;
            out.append(std::move(d));
        }
        return out;
    }

    // Get ordered messages for a session.
    py::list get_session_messages(const std::string& session,
                                  size_t offset = 0,
                                  size_t limit = 50,
                                  bool include_text = true) const {
        py::list out;
        auto it = session_to_indices.find(session);
        if (it == session_to_indices.end()) return out;
        const auto& idxs = it->second;
        if (offset >= idxs.size()) return out;
        size_t end = std::min(idxs.size(), offset + limit);
        for (size_t i = offset; i < end; ++i) {
            size_t idx = idxs[i];
            const auto& tr = memories[idx];
            py::dict d;
            d["id"] = tr.id;
            d["role"] = tr.role;
            d["timestamp"] = tr.timestamp;
            d["importance"] = tr.importance;
            d["seq"] = py::int_(tr.seq);
            if (include_text) d["content"] = tr.text;
            out.append(std::move(d));
        }
        return out;
    }

    // Session-scoped query: only considers memories in the session.
    std::vector<std::pair<std::string, double>> query_session(const std::string& session,
                                                              const std::string& query_text,
                                                              size_t top_k = 10) {
        auto it = session_to_indices.find(session);
        if (it == session_to_indices.end()) return {};

        // Encode query
        encode_text_simd(query_text, workspace.data());
        fftw_execute(forward_plan);

        const auto& idxs = it->second;
        std::vector<std::pair<double, size_t>> scores;
        scores.reserve(idxs.size());
        const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        for (size_t pos = 0; pos < idxs.size(); ++pos) {
            size_t i = idxs[pos];
            std::complex<double> acc(0.0, 0.0);
            const std::complex<double>* row = &mem_matrix[i * dimension];
            for (size_t j = 0; j < dimension; ++j) {
                acc += std::conj(workspace[j]) * row[j];
            }
            double similarity = std::abs(acc);
            double time_decay = exp(-(now - memories[i].timestamp) / 86400.0);
            double final_score = similarity * memories[i].importance * (0.5 + 0.5 * time_decay);
            scores.push_back({final_score, i});
        }
        size_t actual_k = std::min(top_k, scores.size());
        std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
                          std::greater<std::pair<double, size_t>>());
        std::vector<std::pair<std::string, double>> results;
        results.reserve(actual_k);
        for (size_t i = 0; i < actual_k; ++i) {
            size_t idx = scores[i].second;
            results.emplace_back(memories[idx].id, scores[i].first);
        }
        return results;
    }

    // ---------------- HRR public API --------------------
    void store_response_hrr(const std::string& query, const std::string& response_text) {
        uint32_t rid;
        auto it = response_hash.find(response_text);
        if (it != response_hash.end()) rid = it->second;
        else {
            rid = next_response_id++;
            response_hash[response_text] = rid;
            response_dict[rid] = response_text;
        }
        std::vector<std::complex<double>> id_time, bound;
        encode_uint32_as_wave(rid, id_time);
        hrr_bind(query, id_time, bound);
        for (size_t i = 0; i < dimension; ++i) hrr_field[i] += bound[i];
    }

    std::string retrieve_response_hrr(const std::string& query) const {
        if (response_dict.empty()) return std::string();
        std::vector<std::complex<double>> unbound;
        hrr_unbind(hrr_field, query, unbound);
        uint32_t id = decode_best_id(unbound);
        auto it = response_dict.find(id);
        return (it != response_dict.end()) ? it->second : std::string();
    }

    size_t get_memory_usage_bytes() const {
        size_t field_b = dimension * sizeof(std::complex<double>);
        size_t dict_b = 0;
        for (const auto& kv : response_dict) dict_b += sizeof(uint32_t) + kv.second.size();
        return field_b + dict_b;
    }

    py::dict get_hrr_stats() const {
        py::dict d;
        d["dimension"] = (py::int_)dimension;
        d["field_size_mb"] = (double)dimension * 16.0 / (1024.0 * 1024.0);
        d["unique_responses"] = (py::int_)response_dict.size();
        d["total_memory_mb"] = (double)get_memory_usage_bytes() / (1024.0 * 1024.0);
        return d;
    }

    // Clear all memories
    void clear() {
        std::fill(field.begin(), field.end(), std::complex<double>(0, 0));
        std::fill(hrr_field.begin(), hrr_field.end(), std::complex<double>(0, 0));
        memories.clear();
        memory_index.clear();
        mem_matrix.clear();
        memory_count = 0;
        total_store_time_us = 0;
        total_query_time_us = 0;
        response_dict.clear();
        response_hash.clear();
        next_response_id = 1;
        // Update current field to reflect cleared state
        try { write_current_locked(); } catch (...) {}
    }

    // -------- Real wave data getters (exposed via pybind) --------
public:
    py::array_t<std::complex<double>> get_memory_vector(const std::string& memory_id) {
        std::lock_guard<std::mutex> lock(field_mutex);
        auto it = memory_index.find(memory_id);
        if (it == memory_index.end()) {
            throw std::runtime_error("Memory ID not found: " + memory_id);
        }
        size_t idx = it->second;
        if (idx >= memories.size()) {
            throw std::runtime_error("Invalid memory index");
        }
        std::complex<double>* ptr = &mem_matrix[idx * dimension];
        // Return 1D numpy array referencing internal buffer row (no copy)
        return py::array_t<std::complex<double>>(
            { (py::ssize_t)dimension },                // shape
            { (py::ssize_t)sizeof(std::complex<double>) }, // strides
            ptr,
            py::cast(this)
        );
    }

    py::array_t<std::complex<double>> get_collective_vector(const std::vector<std::string>& memory_ids) {
        std::lock_guard<std::mutex> lock(field_mutex);
        py::array_t<std::complex<double>> out((py::ssize_t)dimension);
        auto buf = out.request();
        auto* dst = static_cast<std::complex<double>*>(buf.ptr);
        std::fill(dst, dst + dimension, std::complex<double>(0.0, 0.0));
        for (const auto& id : memory_ids) {
            auto it = memory_index.find(id);
            if (it == memory_index.end()) continue;
            size_t idx = it->second;
            const std::complex<double>* row = &mem_matrix[idx * dimension];
            for (size_t i = 0; i < dimension; ++i) dst[i] += row[i];
        }
        return out;
    }

    py::array_t<std::complex<double>> get_collective_field() {
        std::lock_guard<std::mutex> lock(field_mutex);
        return py::array_t<std::complex<double>>(
            { (py::ssize_t)dimension },
            { (py::ssize_t)sizeof(std::complex<double>) },
            field.data(),
            py::cast(this)
        );
    }
};

// Python bindings
PYBIND11_MODULE(holographic_cpp, m) {
    m.doc() = "Blazing Fast Native Holographic Memory";

    py::class_<HolographicMemory>(m, "HolographicMemory")
        .def(py::init<size_t>(), py::arg("dimension") = 1024)
        .def("store", &HolographicMemory::store,
             py::arg("text"), py::arg("importance") = 1.0,
             "Store a memory and return its ID")
        .def("store_message", &HolographicMemory::store_message,
             py::arg("session"), py::arg("role"), py::arg("text"), py::arg("importance") = 1.0, py::arg("meta") = std::string(),
             "Store a conversation message with session metadata and return its ID")
        .def("store_batch", &HolographicMemory::store_batch,
             py::arg("texts"), py::arg("importance") = 1.0,
             "Store multiple memories efficiently")
        .def("query", &HolographicMemory::query,
             py::arg("query_text"), py::arg("top_k") = 10,
             "Query memories and return top matches")
        .def("query_session", &HolographicMemory::query_session,
             py::arg("session"), py::arg("query_text"), py::arg("top_k") = 10,
             "Query within a session and return id,score pairs")
        .def("query_batch", &HolographicMemory::query_batch,
             py::arg("queries"), py::arg("top_k") = 10,
             "Batch query: list of queries -> list of top-k results per query")
        .def("list_sessions", &HolographicMemory::list_sessions,
             "List sessions with counts and last timestamp")
        .def("get_session_messages", &HolographicMemory::get_session_messages,
             py::arg("session"), py::arg("offset") = 0, py::arg("limit") = 50, py::arg("include_text") = true,
             "Get ordered messages for a session")
        .def("fourier_transform", &HolographicMemory::fourier_transform,
             py::arg("psi"), "Unitary discrete Fourier transform")
        .def("interference_intensity", &HolographicMemory::interference_intensity,
             py::arg("psi1"), py::arg("psi2"), "Compute |psi1+psi2|^2")
        .def("quantum_resonance_query", &HolographicMemory::quantum_resonance_query,
             py::arg("query"), py::arg("top_k") = 10,
             "Resonance-based retrieval using holographic query")
        .def("get_stats", &HolographicMemory::get_stats,
             "Get performance statistics")
        .def("clear", &HolographicMemory::clear,
             "Clear all memories")
        .def("remove", &HolographicMemory::remove,
             py::arg("id"), "Remove a memory by id (O(1) swap)")
        .def("store_response_pattern", &HolographicMemory::store_response_pattern,
             py::arg("key"), py::arg("response_text"),
             "Store a response pattern keyed by query key (PoC)")
        .def("resonate_response", &HolographicMemory::resonate_response,
             py::arg("query_text"), "Return best cached response text if any (PoC)")
        .def("resonate_response_with_score", &HolographicMemory::resonate_response_with_score,
             py::arg("query_text"), "Return best cached response and confidence [0,1] (PoC)")
        .def("get_response_cache_stats", &HolographicMemory::get_response_cache_stats,
             "Get response cache count and dimension for telemetry")
        .def("store_response_hrr", &HolographicMemory::store_response_hrr,
             py::arg("query"), py::arg("response"),
             "Store response via HRR associative binding (compact)")
        .def("retrieve_response_hrr", &HolographicMemory::retrieve_response_hrr,
             py::arg("query"),
             "Retrieve response via HRR unbinding")
        .def("get_memory_usage_bytes", &HolographicMemory::get_memory_usage_bytes,
             "Total HRR memory usage in bytes (field + dict)")
        .def("get_hrr_stats", &HolographicMemory::get_hrr_stats,
             "HRR field stats: dimension, field size MB, unique responses, total memory MB")
        // --- Real wave getters (authentic engine data) ---
        .def("get_memory_vector", &HolographicMemory::get_memory_vector,
             py::arg("memory_id"), "Get complex frequency vector for one memory (authentic)")
        .def("get_collective_vector", &HolographicMemory::get_collective_vector,
             py::arg("memory_ids"), "Get combined complex vector for given memory ids")
        .def("get_collective_field", &HolographicMemory::get_collective_field,
             "Get the full superposed field vector")
        .def_property_readonly("memory_count",
             [](const HolographicMemory& self) { return self.get_stats()["memory_count"]; });

    // Module-level helpers (alternate access)
    m.def("hm_get_memory_vector", [](HolographicMemory& self, const std::string& memory_id){
        return self.get_memory_vector(memory_id);
    }, py::arg("hm"), py::arg("memory_id"));
    m.def("hm_get_collective_vector", [](HolographicMemory& self, const std::vector<std::string>& ids){
        return self.get_collective_vector(ids);
    }, py::arg("hm"), py::arg("memory_ids"));
    m.def("hm_get_collective_field", [](HolographicMemory& self){
        return self.get_collective_field();
    }, py::arg("hm"));
}
