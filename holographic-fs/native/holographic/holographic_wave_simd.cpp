/*
 * WAVE-BASED Holographic Memory with FULL SIMD Vectorization
 * ===========================================================
 *
 * Back to the original vision - pure wave interference!
 * No FFT needed - just superposition and resonance.
 *
 * ARM NEON vectorization everywhere for maximum speed.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>

// Custom fast sine for NEON
static inline float32x4_t vsinq_f32(float32x4_t x) {
    // Taylor series approximation: sin(x) ≈ x - x³/6 + x⁵/120
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x5 = vmulq_f32(x3, x2);

    float32x4_t term1 = x;
    float32x4_t term2 = vmulq_f32(x3, vdupq_n_f32(-1.0f/6.0f));
    float32x4_t term3 = vmulq_f32(x5, vdupq_n_f32(1.0f/120.0f));

    float32x4_t result = vaddq_f32(term1, term2);
    result = vaddq_f32(result, term3);

    return result;
}
#endif

namespace py = pybind11;

class WaveMemorySIMD {
private:
    // Wave field - aligned for SIMD
    alignas(64) std::vector<float> wave_field;  // Using float for SIMD efficiency
    alignas(64) std::vector<float> phase_field; // Phase information

    // Memory traces
    struct WavePattern {
        std::vector<float> amplitude;
        std::vector<float> phase;
        std::string text;
        float importance;
    };

    std::vector<WavePattern> memories;
    size_t dimension;
    size_t memory_count;

    // Performance stats
    double total_store_time_us;
    double total_query_time_us;

    // Persistence
    std::filesystem::path base_dir;
    std::filesystem::path patterns_dir;
    std::filesystem::path snapshots_dir;
    std::filesystem::path current_wave_path;
    double last_snapshot_ts{0.0};

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

    static std::string extract_filename_tag(const std::string& text) {
        size_t pos = text.find('\n');
        std::string first = (pos == std::string::npos) ? text : text.substr(0, pos);
        const std::string prefix = "filename:";
        if (first.rfind(prefix, 0) == 0 && first.size() > prefix.size()) {
            std::string nm = first.substr(prefix.size());
            while (!nm.empty() && (nm[0] == ' ' || nm[0] == '\t')) nm.erase(nm.begin());
            return nm;
        }
        return std::string();
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

    void write_wave_file_amp_phase(const std::filesystem::path& outp,
                                   const std::vector<float>& amp,
                                   const std::vector<float>& ph,
                                   const std::string& meta_json) {
        try {
            std::ofstream ofs(outp, std::ios::binary);
            if (!ofs) return;
            const char magic[8] = {'W','A','V','E','V','0','0','1'};
            ofs.write(magic, 8);
            uint32_t version = 1u;
            uint32_t kind = 2u; // 2 = float32 amp+phase
            uint64_t dim = static_cast<uint64_t>(amp.size());
            double ts = now_sec();
            uint32_t meta_len = static_cast<uint32_t>(meta_json.size());
            ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
            ofs.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
            ofs.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            ofs.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
            ofs.write(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len));
            if (meta_len) ofs.write(meta_json.data(), meta_len);
            ofs.write(reinterpret_cast<const char*>(amp.data()), sizeof(float) * amp.size());
            ofs.write(reinterpret_cast<const char*>(ph.data()), sizeof(float) * ph.size());
        } catch (...) {}
    }

    void snapshot_if_due_locked() {
        double t = now_sec();
        if (t - last_snapshot_ts < 3600.0) return;
        last_snapshot_ts = t;
        std::vector<float> a = wave_field;
        std::vector<float> p = phase_field;
        std::ostringstream mj;
        mj << "{\"type\":\"snapshot\",\"dimension\":" << dimension << ",\"backend\":\"wave_simd\"}";
        write_wave_file_amp_phase(snapshots_dir / (std::string("snapshot_") + timestamp_str() + ".wave"), a, p, mj.str());
    }

    void write_current_locked() {
        std::vector<float> a = wave_field;
        std::vector<float> p = phase_field;
        std::string mj = "{\"type\":\"current\",\"backend\":\"wave_simd\"}";
        write_wave_file_amp_phase(current_wave_path, a, p, mj);
    }

public:
    WaveMemorySIMD(size_t dim = 1024)
        : dimension(dim), memory_count(0),
          total_store_time_us(0), total_query_time_us(0) {

        // Ensure dimension is multiple of 16 for SIMD
        dimension = (dim + 15) & ~15;

        wave_field.resize(dimension, 0.0f);
        phase_field.resize(dimension, 0.0f);
        memories.reserve(100000);

        // Initialize persistence paths
        const char* bd = std::getenv("HLOG_DATA_DIR");
        if (bd && *bd) base_dir = std::filesystem::path(bd);
        else if (std::filesystem::exists("/app/data/holographic_memory")) base_dir = "/app/data/holographic_memory";
        else base_dir = std::filesystem::path("data") / "holographic_memory";
        patterns_dir = base_dir / "patterns";
        snapshots_dir = base_dir / "snapshots";
        current_wave_path = base_dir / "current.wave";
        ensure_dirs();
    }

    // VECTORIZED text encoding using pure waves
    void encode_text_wave_simd(const std::string& text, float* amplitude, float* phase) {
        // Zero outputs using SIMD
        std::memset(amplitude, 0, dimension * sizeof(float));
        std::memset(phase, 0, dimension * sizeof(float));

        // Generate wave patterns from text
        std::hash<std::string> hasher;
        const int window_size = 3;  // Smaller for more patterns

        for (size_t i = 0; i <= text.size() - window_size; ++i) {
            std::string_view window(&text[i], window_size);
            size_t hash = hasher(std::string(window));

            // Create wave with specific frequency and phase
            float base_freq = (hash % 100) / 100.0f * M_PI;
            float base_phase = (hash % 256) / 256.0f * 2.0f * M_PI;
            float amp_scale = 1.0f / (1.0f + i * 0.1f);

            #ifdef __ARM_NEON
            // NEON vectorized wave generation
            float32x4_t freq_vec = vdupq_n_f32(base_freq);
            float32x4_t phase_vec = vdupq_n_f32(base_phase);
            float32x4_t amp_vec = vdupq_n_f32(amp_scale);
            float32x4_t indices = {0, 1, 2, 3};
            float32x4_t four = vdupq_n_f32(4.0f);

            for (size_t j = 0; j < dimension; j += 4) {
                // Calculate wave: A * sin(freq * index + phase)
                float32x4_t freq_mult = vmulq_f32(freq_vec, indices);
                float32x4_t phase_total = vaddq_f32(freq_mult, phase_vec);

                // Fast sine approximation for SIMD
                float32x4_t sin_val = vsinq_f32(phase_total);  // Would need custom impl
                float32x4_t wave_amp = vmulq_f32(amp_vec, sin_val);

                // Add to existing amplitude
                float32x4_t existing = vld1q_f32(&amplitude[j]);
                float32x4_t result = vaddq_f32(existing, wave_amp);
                vst1q_f32(&amplitude[j], result);

                // Store phase
                float32x4_t existing_phase = vld1q_f32(&phase[j]);
                float32x4_t new_phase = vaddq_f32(existing_phase, phase_total);
                vst1q_f32(&phase[j], new_phase);

                indices = vaddq_f32(indices, four);
            }
            #elif defined(__APPLE__)
            // Use Accelerate vDSP for vectorized operations
            float freq_array[dimension];
            float phase_array[dimension];

            // Generate frequency indices
            vDSP_vramp(&base_freq, &base_freq, freq_array, 1, dimension);

            // Add phase offset
            vDSP_vsadd(freq_array, 1, &base_phase, phase_array, 1, dimension);

            // Generate sine wave
            // Use simple scalar sine for now
            for (size_t j = 0; j < dimension; ++j) {
                freq_array[j] = sinf(phase_array[j]);
            }

            // Scale and add to amplitude
            vDSP_vsma(freq_array, 1, &amp_scale, amplitude, 1, amplitude, 1, dimension);

            // Update phase field
            vDSP_vadd(phase, 1, phase_array, 1, phase, 1, dimension);
            #else
            // Scalar fallback with unrolling
            for (size_t j = 0; j < dimension; j += 4) {
                for (size_t k = 0; k < 4 && j + k < dimension; ++k) {
                    float freq = base_freq * (j + k);
                    float ph = base_phase + freq;
                    amplitude[j + k] += amp_scale * sinf(ph);
                    phase[j + k] += ph;
                }
            }
            #endif
        }

        // Normalize using SIMD
        #ifdef __APPLE__
        float norm = 0;
        vDSP_svesq(amplitude, 1, &norm, dimension);
        if (norm > 0) {
            norm = 1.0f / sqrtf(norm);
            vDSP_vsmul(amplitude, 1, &norm, amplitude, 1, dimension);
            vDSP_vsmul(phase, 1, &norm, phase, 1, dimension);
        }
        #else
        float norm = 0;
        for (size_t i = 0; i < dimension; ++i) {
            norm += amplitude[i] * amplitude[i];
        }
        if (norm > 0) {
            norm = 1.0f / sqrtf(norm);
            for (size_t i = 0; i < dimension; ++i) {
                amplitude[i] *= norm;
                phase[i] *= norm;
            }
        }
        #endif
    }

    // VECTORIZED wave superposition
    void superpose_waves(const float* new_amp, const float* new_phase, float importance) {
        #ifdef __ARM_NEON
        float32x4_t imp_vec = vdupq_n_f32(importance);

        for (size_t i = 0; i < dimension; i += 4) {
            // Load existing waves
            float32x4_t existing_amp = vld1q_f32(&wave_field[i]);
            float32x4_t existing_phase = vld1q_f32(&phase_field[i]);

            // Load new waves
            float32x4_t new_amp_vec = vld1q_f32(&new_amp[i]);
            float32x4_t new_phase_vec = vld1q_f32(&new_phase[i]);

            // Scale new waves by importance
            new_amp_vec = vmulq_f32(new_amp_vec, imp_vec);

            // Superpose amplitudes
            float32x4_t result_amp = vaddq_f32(existing_amp, new_amp_vec);

            // Average phases (simplified interference)
            float32x4_t result_phase = vaddq_f32(existing_phase, new_phase_vec);
            result_phase = vmulq_f32(result_phase, vdupq_n_f32(0.5f));

            // Store results
            vst1q_f32(&wave_field[i], result_amp);
            vst1q_f32(&phase_field[i], result_phase);
        }
        #elif defined(__APPLE__)
        // Use vDSP for vectorized superposition
        float scaled_amp[dimension];
        vDSP_vsmul(new_amp, 1, &importance, scaled_amp, 1, dimension);
        vDSP_vadd(wave_field.data(), 1, scaled_amp, 1, wave_field.data(), 1, dimension);

        // Average phases
        vDSP_vadd(phase_field.data(), 1, new_phase, 1, phase_field.data(), 1, dimension);
        float half = 0.5f;
        vDSP_vsmul(phase_field.data(), 1, &half, phase_field.data(), 1, dimension);
        #else
        // Scalar with unrolling
        for (size_t i = 0; i < dimension; i += 4) {
            for (size_t j = 0; j < 4 && i + j < dimension; ++j) {
                wave_field[i + j] += new_amp[i + j] * importance;
                phase_field[i + j] = (phase_field[i + j] + new_phase[i + j]) * 0.5f;
            }
        }
        #endif
    }

    // VECTORIZED resonance calculation
    float calculate_resonance_simd(const float* query_amp, const float* query_phase,
                                   const float* mem_amp, const float* mem_phase) {
        float resonance = 0;

        #ifdef __ARM_NEON
        float32x4_t sum = vdupq_n_f32(0);

        for (size_t i = 0; i < dimension; i += 4) {
            // Load amplitudes
            float32x4_t q_amp = vld1q_f32(&query_amp[i]);
            float32x4_t m_amp = vld1q_f32(&mem_amp[i]);

            // Load phases
            float32x4_t q_phase = vld1q_f32(&query_phase[i]);
            float32x4_t m_phase = vld1q_f32(&mem_phase[i]);

            // Calculate phase difference
            float32x4_t phase_diff = vsubq_f32(q_phase, m_phase);

            // cos(phase_diff) for interference
            // Approximate: 1 - phase_diff^2/2 for small angles
            float32x4_t phase_sq = vmulq_f32(phase_diff, phase_diff);
            float32x4_t cos_approx = vsubq_f32(vdupq_n_f32(1.0f),
                                               vmulq_f32(phase_sq, vdupq_n_f32(0.5f)));

            // Amplitude product * cos(phase_diff)
            float32x4_t amp_product = vmulq_f32(q_amp, m_amp);
            float32x4_t contribution = vmulq_f32(amp_product, cos_approx);

            sum = vaddq_f32(sum, contribution);
        }

        // Sum all lanes
        float32x2_t sum_pairs = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        resonance = vget_lane_f32(vpadd_f32(sum_pairs, sum_pairs), 0);

        #elif defined(__APPLE__)
        // Use vDSP for dot product with phase consideration
        float amp_product[dimension];
        float phase_diff[dimension];
        float cos_vals[dimension];

        // Calculate amplitude products
        vDSP_vmul(query_amp, 1, mem_amp, 1, amp_product, 1, dimension);

        // Calculate phase differences
        vDSP_vsub(mem_phase, 1, query_phase, 1, phase_diff, 1, dimension);

        // Calculate cos(phase_diff) - use scalar for simplicity
        for (size_t i = 0; i < dimension; ++i) {
            cos_vals[i] = cosf(phase_diff[i]);
        }

        // Multiply and sum
        vDSP_vmul(amp_product, 1, cos_vals, 1, amp_product, 1, dimension);
        vDSP_sve(amp_product, 1, &resonance, dimension);

        #else
        // Scalar fallback
        for (size_t i = 0; i < dimension; ++i) {
            float phase_diff = query_phase[i] - mem_phase[i];
            resonance += query_amp[i] * mem_amp[i] * cosf(phase_diff);
        }
        #endif

        return resonance;
    }

    // Batch store with full SIMD
    std::vector<std::string> store_batch(const std::vector<std::string>& texts, float importance = 1.0f) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::string> ids;
        ids.reserve(texts.size());

        // Process all texts
        for (const auto& text : texts) {
            WavePattern pattern;
            pattern.amplitude.resize(dimension);
            pattern.phase.resize(dimension);
            pattern.text = text;
            pattern.importance = importance;

            // Encode with SIMD
            encode_text_wave_simd(text, pattern.amplitude.data(), pattern.phase.data());

            // Persist individual pattern if filename tag present
            std::string orig = extract_filename_tag(text);
            if (!orig.empty()) {
                std::ostringstream mj;
                mj << "{\"type\":\"pattern\",\"filename\":\"" << sanitize(orig) << "\",\"dimension\":" << dimension << ",\"backend\":\"wave_simd\"}";
                write_wave_file_amp_phase(patterns_dir / (sanitize(orig) + ".wave"), pattern.amplitude, pattern.phase, mj.str());
            }

            // Superpose onto field with SIMD
            superpose_waves(pattern.amplitude.data(), pattern.phase.data(), importance);

            // Store pattern
            memories.push_back(std::move(pattern));
            ids.push_back("wave_" + std::to_string(memory_count++));
        }

        // Update current and snapshot after batch
        write_current_locked();
        snapshot_if_due_locked();

        auto end = std::chrono::high_resolution_clock::now();
        total_store_time_us += std::chrono::duration<double, std::micro>(end - start).count();

        return ids;
    }

    // Query with full SIMD resonance
    std::vector<std::pair<std::string, float>> query(const std::string& query_text, size_t top_k = 10) {
        auto start = std::chrono::high_resolution_clock::now();

        // Encode query
        std::vector<float> query_amp(dimension);
        std::vector<float> query_phase(dimension);
        encode_text_wave_simd(query_text, query_amp.data(), query_phase.data());

        // Calculate resonance with all memories (parallelized)
        std::vector<std::pair<float, size_t>> scores(memories.size());

        #pragma omp parallel for if(memories.size() > 64) schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(memories.size()); ++ii) {
            size_t i = static_cast<size_t>(ii);
            float resonance = calculate_resonance_simd(
                query_amp.data(), query_phase.data(),
                memories[i].amplitude.data(), memories[i].phase.data()
            );
            resonance *= memories[i].importance;
            scores[i] = {resonance, i};
        }

        // Get top-k
        size_t actual_k = std::min(top_k, scores.size());
        std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
                         std::greater<std::pair<float, size_t>>());

        // Build results
        std::vector<std::pair<std::string, float>> results;
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

    py::dict get_stats() const {
        py::dict stats;
        stats["memory_count"] = memory_count;
        stats["dimension"] = dimension;
        stats["avg_store_us"] = memory_count > 0 ? total_store_time_us / memory_count : 0;
        stats["avg_query_us"] = memory_count > 0 ? total_query_time_us / memory_count : 0;
        stats["backend"] = "Wave Interference + FULL SIMD";
        return stats;
    }

    void clear() {
        std::fill(wave_field.begin(), wave_field.end(), 0.0f);
        std::fill(phase_field.begin(), phase_field.end(), 0.0f);
        memories.clear();
        memory_count = 0;
        total_store_time_us = 0;
        total_query_time_us = 0;
        try { write_current_locked(); } catch (...) {}
    }

    ~WaveMemorySIMD() {
        try { snapshot_if_due_locked(); write_current_locked(); } catch (...) {}
    }
};

// Python bindings
PYBIND11_MODULE(holographic_wave_simd, m) {
    m.doc() = "Wave-based Holographic Memory with FULL SIMD Vectorization";

    py::class_<WaveMemorySIMD>(m, "WaveMemory")
        .def(py::init<size_t>(), py::arg("dimension") = 1024)
        .def("store_batch", &WaveMemorySIMD::store_batch,
             py::arg("texts"), py::arg("importance") = 1.0f)
        .def("query", &WaveMemorySIMD::query,
             py::arg("query_text"), py::arg("top_k") = 10)
        .def("get_stats", &WaveMemorySIMD::get_stats)
        .def("clear", &WaveMemorySIMD::clear)
        .def_property_readonly("memory_count",
             [](const WaveMemorySIMD& self) {
                 return self.get_stats()["memory_count"];
             });
}
