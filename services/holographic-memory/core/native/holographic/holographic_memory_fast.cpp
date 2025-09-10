/*
 * FAST Holographic Memory - Key Optimizations Only
 * =================================================
 * 
 * Focused optimizations that actually work:
 * - Batch FFT operations
 * - Better memory layout
 * - Reduced allocations
 * - SIMD everywhere
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fftw3.h>
#include <complex>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstring>
#include <chrono>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
extern "C" {
    void cblas_zdotc_sub(const int N, const void *X, const int incX,
                         const void *Y, const int incY, void *dotc);
}
#endif

namespace py = pybind11;

class HolographicMemoryFast {
private:
    // Main field - aligned for SIMD
    alignas(64) std::vector<std::complex<double>> field;
    
    // Pre-allocated workspace for batch operations
    alignas(64) std::vector<std::complex<double>> batch_workspace;
    
    // Memory storage
    struct MemoryTrace {
        std::vector<std::complex<double>> frequency_signature;
        std::string text;
        float importance;
        float timestamp;
    };
    
    std::vector<MemoryTrace> memories;
    
    // FFTW plans - reuse for all operations
    fftw_plan forward_plan;
    fftw_plan batch_forward_plan;
    
    size_t dimension;
    size_t memory_count;
    
    // Performance tracking
    double total_store_time_us;
    double total_query_time_us;
    
public:
    HolographicMemoryFast(size_t dim = 1024) 
        : dimension(dim), memory_count(0),
          total_store_time_us(0), total_query_time_us(0) {
        
        field.resize(dim, std::complex<double>(0, 0));
        batch_workspace.resize(dim * 16);  // Space for 16 parallel operations
        
        // Create reusable FFT plans
        forward_plan = fftw_plan_dft_1d(
            dim,
            reinterpret_cast<fftw_complex*>(batch_workspace.data()),
            reinterpret_cast<fftw_complex*>(batch_workspace.data()),
            FFTW_FORWARD,
            FFTW_MEASURE  // Spend time to find optimal algorithm
        );
        
        // Batch plan for multiple FFTs
        int n[] = {static_cast<int>(dim)};
        batch_forward_plan = fftw_plan_many_dft(
            1, n, 16,  // 16 transforms
            reinterpret_cast<fftw_complex*>(batch_workspace.data()),
            nullptr, 1, dim,
            reinterpret_cast<fftw_complex*>(batch_workspace.data()),
            nullptr, 1, dim,
            FFTW_FORWARD, FFTW_MEASURE
        );
        
        memories.reserve(100000);
    }
    
    ~HolographicMemoryFast() {
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(batch_forward_plan);
    }
    
    // Fast encoding with minimal allocations
    inline void encode_text_fast(const std::string& text, std::complex<double>* output) {
        // Use memset for fast zeroing
        std::memset(output, 0, dimension * sizeof(std::complex<double>));
        
        std::hash<std::string> hasher;
        const int window_size = 5;
        const size_t max_windows = text.size() >= window_size ? text.size() - window_size + 1 : 0;
        
        // Process all windows
        for (size_t i = 0; i < max_windows; ++i) {
            // Use string_view to avoid allocation
            std::string_view window(&text[i], window_size);
            size_t hash = hasher(std::string(window));
            
            // Unroll inner loop
            #pragma unroll 8
            for (size_t j = 0; j < 8 && j < dimension; ++j) {
                size_t freq_idx = (hash + j * 0x9e3779b9) % dimension;
                double phase = (i + hash) * 2.0 * M_PI / 256.0;
                double amplitude = 1.0 / (1.0 + i * 0.1);
                
                // Direct assignment is faster than +=
                double real = amplitude * cos(phase);
                double imag = amplitude * sin(phase);
                output[freq_idx].real(output[freq_idx].real() + real);
                output[freq_idx].imag(output[freq_idx].imag() + imag);
            }
        }
        
        // Fast normalization
        #ifdef __APPLE__
        // Use vDSP for fast norm calculation
        double norm_real = 0.0, norm_imag = 0.0;
        vDSP_svesqD(reinterpret_cast<double*>(output), 2, &norm_real, dimension);
        vDSP_svesqD(reinterpret_cast<double*>(output) + 1, 2, &norm_imag, dimension);
        double norm_sq = norm_real + norm_imag;
        
        if (norm_sq > 0) {
            double scale = 1.0 / sqrt(norm_sq);
            vDSP_vsmulD((double*)output, 1, &scale, (double*)output, 1, dimension * 2);
        }
        #else
        double norm_sq = 0.0;
        for (size_t i = 0; i < dimension; ++i) {
            norm_sq += std::norm(output[i]);
        }
        
        if (norm_sq > 0) {
            double inv_norm = 1.0 / sqrt(norm_sq);
            for (size_t i = 0; i < dimension; ++i) {
                output[i] *= inv_norm;
            }
        }
        #endif
    }
    
    // Optimized batch store - KEY TO PERFORMANCE!
    std::vector<std::string> store_batch(const std::vector<std::string>& texts, double importance = 1.0) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::string> ids;
        ids.reserve(texts.size());
        
        // Process in chunks of 16 for optimal batch FFT
        for (size_t chunk_start = 0; chunk_start < texts.size(); chunk_start += 16) {
            size_t chunk_size = std::min(size_t(16), texts.size() - chunk_start);
            
            // Encode all in chunk
            for (size_t i = 0; i < chunk_size; ++i) {
                encode_text_fast(texts[chunk_start + i], 
                               &batch_workspace[i * dimension]);
            }
            
            // Batch FFT - MUCH faster than individual FFTs
            if (chunk_size == 16) {
                fftw_execute(batch_forward_plan);
            } else {
                // Do individual FFTs for partial chunk
                for (size_t i = 0; i < chunk_size; ++i) {
                    fftw_execute_dft(forward_plan,
                        reinterpret_cast<fftw_complex*>(&batch_workspace[i * dimension]),
                        reinterpret_cast<fftw_complex*>(&batch_workspace[i * dimension]));
                }
            }
            
            // Add to field and store traces
            for (size_t i = 0; i < chunk_size; ++i) {
                // Add to field using SIMD
                #ifdef __APPLE__
                double scale = importance;
                cblas_zaxpy(dimension, &scale,
                           (double*)&batch_workspace[i * dimension], 1,
                           (double*)field.data(), 1);
                #else
                for (size_t j = 0; j < dimension; ++j) {
                    field[j] += batch_workspace[i * dimension + j] * importance;
                }
                #endif
                
                // Store trace
                MemoryTrace trace;
                trace.text = texts[chunk_start + i];
                trace.frequency_signature.assign(
                    &batch_workspace[i * dimension],
                    &batch_workspace[(i + 1) * dimension]
                );
                trace.importance = importance;
                trace.timestamp = std::chrono::duration<float>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                
                memories.push_back(std::move(trace));
                ids.push_back("mem_" + std::to_string(memory_count++));
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        total_store_time_us += std::chrono::duration<double, std::micro>(end - start).count();
        
        return ids;
    }
    
    // Single store (uses batch internally for efficiency)
    std::string store(const std::string& text, double importance = 1.0) {
        std::vector<std::string> texts = {text};
        return store_batch(texts, importance)[0];
    }
    
    // Fast query using SIMD dot products
    std::vector<std::pair<std::string, double>> query(const std::string& query_text, size_t top_k = 10) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Encode and FFT query
        encode_text_fast(query_text, batch_workspace.data());
        fftw_execute_dft(forward_plan,
            reinterpret_cast<fftw_complex*>(batch_workspace.data()),
            reinterpret_cast<fftw_complex*>(batch_workspace.data()));
        
        // Score all memories
        std::vector<std::pair<double, size_t>> scores;
        scores.reserve(memories.size());
        
        for (size_t i = 0; i < memories.size(); ++i) {
            double similarity = 0.0;
            
            #ifdef __APPLE__
            // Fast complex dot product
            std::complex<double> result;
            cblas_zdotc_sub(dimension,
                (double*)batch_workspace.data(), 1,
                (double*)memories[i].frequency_signature.data(), 1,
                &result);
            similarity = std::abs(result);
            #else
            // Manual dot product
            for (size_t j = 0; j < dimension; ++j) {
                auto dot = batch_workspace[j] * std::conj(memories[i].frequency_signature[j]);
                similarity += std::abs(dot);
            }
            #endif
            
            // Apply importance and time decay
            float current_time = std::chrono::duration<float>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            double time_decay = exp(-(current_time - memories[i].timestamp) / 86400.0);
            double final_score = similarity * memories[i].importance * (0.5 + 0.5 * time_decay);
            
            scores.push_back({final_score, i});
        }
        
        // Get top-k efficiently
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
    
    py::dict get_stats() const {
        py::dict stats;
        stats["memory_count"] = memory_count;
        stats["dimension"] = dimension;
        stats["avg_store_us"] = memory_count > 0 ? total_store_time_us / memory_count : 0;
        stats["avg_query_us"] = memory_count > 0 ? total_query_time_us / memory_count : 0;
        stats["backend"] = "C++ FAST (Batch FFT + SIMD)";
        return stats;
    }
    
    void clear() {
        std::fill(field.begin(), field.end(), std::complex<double>(0, 0));
        memories.clear();
        memory_count = 0;
        total_store_time_us = 0;
        total_query_time_us = 0;
    }
};

// Python bindings
PYBIND11_MODULE(holographic_fast, m) {
    m.doc() = "Fast Holographic Memory with Batch Operations";
    
    py::class_<HolographicMemoryFast>(m, "HolographicMemory")
        .def(py::init<size_t>(), py::arg("dimension") = 1024)
        .def("store", &HolographicMemoryFast::store,
             py::arg("text"), py::arg("importance") = 1.0)
        .def("store_batch", &HolographicMemoryFast::store_batch,
             py::arg("texts"), py::arg("importance") = 1.0)
        .def("query", &HolographicMemoryFast::query,
             py::arg("query_text"), py::arg("top_k") = 10)
        .def("get_stats", &HolographicMemoryFast::get_stats)
        .def("clear", &HolographicMemoryFast::clear)
        .def_property_readonly("memory_count",
             [](const HolographicMemoryFast& self) {
                 return self.get_stats()["memory_count"];
             });
}
