/*
 * HolographicMemory3D - True 3D spatial holographic memory (exact recall)
 * ------------------------------------------------------------
 * - 3D complex field ψ(x,y,z) stored as contiguous row‑major array (N×N×N)
 * - Disjoint spatial placement for exact round‑trip of input bytes
 * - 3D FFT (FFTW) used only for interference visualization (|FFTN(ψ)|^2)
 * - Persistence compatible with existing .hwp scheme; adds shape/dtype meta
 *
 * Build: via existing setup.py/CMake in this directory
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <fftw3.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <mutex>

namespace py = pybind11;

class HolographicMemory3D {
private:
    size_t N;                // grid size
    size_t total;            // N^3
    size_t alloc_ptr;        // next free flat index
    std::vector<std::complex<double>> psi; // spatial field ψ(x)
    mutable std::mutex mtx;  // thread-safety

    struct DocMeta {
        std::string id;
        size_t start;
        size_t length;
        double created_at;
        std::string filename;
    };
    std::unordered_map<std::string, DocMeta> docs;

    // Persistence paths
    std::filesystem::path base_dir;
    std::filesystem::path patterns_dir;
    std::filesystem::path snapshots_dir;
    std::filesystem::path current_wave_path;
    std::filesystem::path meta_path;
    double last_snapshot_ts;

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

    // Binary writer for complex128 arrays with metadata
    void write_wave_file(const std::filesystem::path& outpath,
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

    void write_current_locked() {
        // meta includes shape, dtype, backend
        std::ostringstream mj;
        mj << "{\"type\":\"current\",\"backend\":\"native_3d\",\"shape\":[" << N << "," << N << "," << N
           << "],\"dtype\":\"complex128\",\"endian\":\"little\"}";
        write_wave_file(current_wave_path, psi, mj.str());
    }

    void snapshot_if_due_locked() {
        double t = now_sec();
        if (t - last_snapshot_ts < 3600.0) return; // hourly
        last_snapshot_ts = t;
        std::vector<std::complex<double>> copy = psi;
        std::ostringstream mj;
        mj << "{\"type\":\"snapshot\",\"backend\":\"native_3d\",\"shape\":[" << N << "," << N << "," << N
           << "],\"dtype\":\"complex128\",\"endian\":\"little\"}";
        auto snap = snapshots_dir / (std::string("snapshot_") + timestamp_str() + ".hwp");
        write_wave_file(snap, copy, mj.str());
    }

    void write_pattern_locked(const std::string& name,
                              const std::vector<std::complex<double>>& amps) {
        std::ostringstream mj;
        mj << "{\"type\":\"pattern\",\"backend\":\"native_3d\",\"shape\":[" << amps.size() << "]"
              ",\"dtype\":\"complex128\",\"endian\":\"little\"}"; // 1D slice pattern
        auto outp = patterns_dir / (name + ".hwp");
        write_wave_file(outp, amps, mj.str());
    }

    void refresh_field_from_disk_locked() {
        try {
            if (!std::filesystem::exists(current_wave_path)) return;
            std::ifstream ifs(current_wave_path, std::ios::binary);
            if (!ifs) return;
            char magic[8];
            ifs.read(magic, 8);
            if (!(ifs && std::memcmp(magic, "WAVEV001", 8) == 0)) return;
            uint32_t version=0, kind=0; uint64_t dim=0; double ts=0; uint32_t mlen=0;
            ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
            ifs.read(reinterpret_cast<char*>(&kind), sizeof(kind));
            ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            ifs.read(reinterpret_cast<char*>(&ts), sizeof(ts));
            ifs.read(reinterpret_cast<char*>(&mlen), sizeof(mlen));
            if (mlen > 0) {
                std::string meta; meta.resize(mlen);
                ifs.read(meta.data(), mlen);
            }
            if (kind == 1 && dim == total) {
                ifs.read(reinterpret_cast<char*>(psi.data()), sizeof(std::complex<double>) * psi.size());
            }
        } catch (...) {}
    }

    void reload_index_from_disk_locked() {
        try {
            if (!std::filesystem::exists(meta_path)) return;
            std::ifstream ifs(meta_path);
            std::stringstream buf; buf << ifs.rdbuf();
            std::string s = buf.str();
            auto ap = s.find("\"alloc_ptr\"");
            if (ap != std::string::npos) {
                auto col = s.find(':', ap);
                if (col != std::string::npos) {
                    try { alloc_ptr = static_cast<size_t>(std::stoull(s.substr(col+1))); } catch (...) {}
                }
            }
            size_t pos = 0;
            while ((pos = s.find("\"id\"", pos)) != std::string::npos) {
                auto q1 = s.find('"', pos + 4); if (q1 == std::string::npos) break;
                auto q2 = s.find('"', q1 + 1); if (q2 == std::string::npos) break;
                std::string id = s.substr(q1 + 1, q2 - q1 - 1);
                size_t stp = s.find("\"start\"", q2);
                size_t lnp = s.find("\"length\"", q2);
                size_t ctp = s.find("\"created_at\"", q2);
                if (stp==std::string::npos || lnp==std::string::npos || ctp==std::string::npos) { pos = q2 + 1; continue; }
                auto sc = s.find(':', stp); auto lc = s.find(':', lnp); auto cc = s.find(':', ctp);
                size_t start=0, length=0; double created=0.0;
                try { start = static_cast<size_t>(std::stoull(s.substr(sc+1))); } catch (...) {}
                try { length = static_cast<size_t>(std::stoull(s.substr(lc+1))); } catch (...) {}
                try { created = std::stod(s.substr(cc+1)); } catch (...) {}
                docs[id] = DocMeta{id, start, length, created, std::string()};
                pos = q2 + 1;
            }
        } catch (...) {}
    }

public:
    explicit HolographicMemory3D(size_t grid_size = 64)
        : N(grid_size), total(grid_size * grid_size * grid_size), alloc_ptr(0), psi(total), last_snapshot_ts(0.0) {
        // Initialize persistence paths similar to native 1D module
        const char* bd = std::getenv("HLOG_DATA_DIR");
        if (bd && *bd) base_dir = std::filesystem::path(bd);
        else if (std::filesystem::exists("/app/data/holographic_memory")) base_dir = "/app/data/holographic_memory";
        else base_dir = std::filesystem::path("data") / "holographic_memory";
        patterns_dir = base_dir / "patterns";
        snapshots_dir = base_dir / "snapshots";
        current_wave_path = base_dir / "current.hwp";
        meta_path = base_dir / "metadata_3d.json";
        ensure_dirs();
        // Attempt to load field from current.hwp
        try {
            if (std::filesystem::exists(current_wave_path)) {
                std::ifstream ifs(current_wave_path, std::ios::binary);
                if (ifs) {
                    char magic[8];
                    ifs.read(magic, 8);
                    if (ifs && std::memcmp(magic, "WAVEV001", 8) == 0) {
                        uint32_t version=0, kind=0; uint64_t dim=0; double ts=0; uint32_t mlen=0;
                        ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
                        ifs.read(reinterpret_cast<char*>(&kind), sizeof(kind));
                        ifs.read(reinterpret_cast<char*>(&dim), sizeof(dim));
                        ifs.read(reinterpret_cast<char*>(&ts), sizeof(ts));
                        ifs.read(reinterpret_cast<char*>(&mlen), sizeof(mlen));
                        if (mlen > 0) {
                            std::string meta; meta.resize(mlen);
                            ifs.read(meta.data(), mlen);
                        }
                        if (kind == 1 && dim == total) {
                            ifs.read(reinterpret_cast<char*>(psi.data()), sizeof(std::complex<double>) * psi.size());
                        }
                    }
                }
            }
        } catch (...) {}
        // Attempt to load index metadata
        try {
            if (std::filesystem::exists(meta_path)) {
                std::ifstream ifs(meta_path);
                std::stringstream buf; buf << ifs.rdbuf();
                std::string s = buf.str();
                auto ap = s.find("\"alloc_ptr\"");
                if (ap != std::string::npos) {
                    auto col = s.find(':', ap);
                    if (col != std::string::npos) {
                        try { alloc_ptr = static_cast<size_t>(std::stoull(s.substr(col+1))); } catch (...) {}
                    }
                }
                size_t pos = 0;
                while ((pos = s.find("\"id\"", pos)) != std::string::npos) {
                    auto q1 = s.find('"', pos + 4); if (q1 == std::string::npos) break;
                    auto q2 = s.find('"', q1 + 1); if (q2 == std::string::npos) break;
                    std::string id = s.substr(q1 + 1, q2 - q1 - 1);
                    size_t stp = s.find("\"start\"", q2);
                    size_t lnp = s.find("\"length\"", q2);
                    size_t ctp = s.find("\"created_at\"", q2);
                    if (stp==std::string::npos || lnp==std::string::npos || ctp==std::string::npos) { pos = q2 + 1; continue; }
                    auto sc = s.find(':', stp); auto lc = s.find(':', lnp); auto cc = s.find(':', ctp);
                    size_t start=0, length=0; double created=0.0;
                    try { start = static_cast<size_t>(std::stoull(s.substr(sc+1))); } catch (...) {}
                    try { length = static_cast<size_t>(std::stoull(s.substr(lc+1))); } catch (...) {}
                    try { created = std::stod(s.substr(cc+1)); } catch (...) {}
                    docs[id] = DocMeta{id, start, length, created, std::string()};
                    pos = q2 + 1;
                }
            }
        } catch (...) {}
    }

    // Store raw data (string bytes) with provided doc_id; returns doc_id
    std::string store(const std::string& data, const std::string& doc_id) {
        std::lock_guard<std::mutex> lock(mtx);
        const size_t L = data.size();
        if (L == 0) return doc_id;
        if (alloc_ptr + L > total) {
            throw std::runtime_error("Insufficient holographic capacity for data");
        }

        // Convert bytes to amplitudes [0,1] in real part; imag=0
        std::vector<std::complex<double>> amps(L);
        for (size_t i = 0; i < L; ++i) {
            unsigned char b = static_cast<unsigned char>(data[i]);
            double a = static_cast<double>(b) / 255.0;
            amps[i] = std::complex<double>(a, 0.0);
        }

        // Place into disjoint region (superposition additive, but region is disjoint)
        size_t s = alloc_ptr;
        size_t e = s + L;
        for (size_t k = 0; k < L; ++k) {
            psi[s + k] += amps[k];
        }
        alloc_ptr = e;

        // Record metadata
        DocMeta dm{doc_id, s, L, now_sec(), std::string()};
        docs[doc_id] = dm;

        // Persist pattern (1D slice) and updated field
        write_pattern_locked(doc_id, amps);
        write_current_locked();
        snapshot_if_due_locked();
        // Update metadata
        try {
            std::ostringstream oss;
            oss << "{\n  \"grid_size\": " << N << ",\n  \"alloc_ptr\": " << alloc_ptr << ",\n  \"docs\": [\n";
            bool first = true;
            for (const auto& kv : docs) {
                const auto& d = kv.second;
                if (!first) oss << ",\n"; first = false;
                oss << "    {\"id\":\"" << d.id << "\",\"start\":" << d.start
                    << ",\"length\":" << d.length << ",\"created_at\":" << std::fixed << d.created_at << "}";
            }
            oss << "\n  ]\n}\n";
            auto tmp = meta_path; tmp += ".tmp";
            std::ofstream ofs(tmp); ofs << oss.str(); ofs.flush(); ofs.close();
            std::error_code ec; std::filesystem::rename(tmp, meta_path, ec);
        } catch (...) {}
        return doc_id;
    }

    // Retrieve raw data by id; returns original bytes string
    std::string retrieve(const std::string& doc_id) const {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = docs.find(doc_id);
        if (it == docs.end()) {
            const_cast<HolographicMemory3D*>(this)->reload_index_from_disk_locked();
            it = docs.find(doc_id);
            if (it == docs.end()) throw std::runtime_error("Document not found");
        }
        const DocMeta& dm = it->second;
        const_cast<HolographicMemory3D*>(this)->refresh_field_from_disk_locked();
        std::string out;
        out.resize(dm.length);
        for (size_t i = 0; i < dm.length; ++i) {
            double a = psi[dm.start + i].real();
            if (a < 0.0) a = 0.0; if (a > 1.0) a = 1.0;
            int v = static_cast<int>(std::llround(a * 255.0));
            out[i] = static_cast<char>(static_cast<unsigned char>(v));
        }
        return out;
    }

    // Delete document; returns True if removed
    bool delete_document(const std::string& doc_id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = docs.find(doc_id);
        if (it == docs.end()) return false;
        const DocMeta& dm = it->second;
        // Clear region to zero
        for (size_t i = 0; i < dm.length; ++i) psi[dm.start + i] = std::complex<double>(0.0, 0.0);
        // Reclaim tail if last block
        if (dm.start + dm.length == alloc_ptr) alloc_ptr = dm.start;
        docs.erase(it);
        write_current_locked();
        // persist updated metadata
        try {
            std::ostringstream oss;
            oss << "{\n  \"grid_size\": " << N << ",\n  \"alloc_ptr\": " << alloc_ptr << ",\n  \"docs\": [\n";
            bool first = true;
            for (const auto& kv : docs) {
                const auto& d = kv.second;
                if (!first) oss << ",\n"; first = false;
                oss << "    {\"id\":\"" << d.id << "\",\"start\":" << d.start
                    << ",\"length\":" << d.length << ",\"created_at\":" << std::fixed << d.created_at << "}";
            }
            oss << "\n  ]\n}\n";
            auto tmp = meta_path; tmp += ".tmp";
            std::ofstream ofs(tmp); ofs << oss.str(); ofs.flush(); ofs.close();
            std::error_code ec; std::filesystem::rename(tmp, meta_path, ec);
        } catch (...) {}
        return true;
    }

    // Return |FFTN(ψ)|^2 as double array (downsampled)
    py::array_t<double> interference_pattern(int downsample = 32) {
        std::lock_guard<std::mutex> lock(mtx);
        size_t ds = (downsample > 0) ? static_cast<size_t>(downsample) : N;
        if (ds > N) ds = N;
        if (ds < 4) ds = 4;
        size_t step = (N + ds - 1) / ds; // ceil division
        size_t eff = (N + step - 1) / step; // effective sampled size
        if (eff < 4) eff = 4;

        // Build sampled cube
        std::vector<std::complex<double>> cube(eff * eff * eff);
        auto at_flat = [&](size_t x, size_t y, size_t z) -> const std::complex<double>& {
            size_t idx = (z * N + y) * N + x; // row-major: x fast, then y, then z
            return psi[idx];
        };
        size_t xi = 0;
        for (size_t x = 0; x < N && xi < eff; x += step, ++xi) {
            size_t yi = 0;
            for (size_t y = 0; y < N && yi < eff; y += step, ++yi) {
                size_t zi = 0;
                for (size_t z = 0; z < N && zi < eff; z += step, ++zi) {
                    size_t dst = (zi * eff + yi) * eff + xi;
                    cube[dst] = at_flat(x, y, z);
                }
            }
        }

        // FFTW 3D forward
        std::vector<std::complex<double>> out(cube.size());
        fftw_plan plan = fftw_plan_dft_3d((int)eff, (int)eff, (int)eff,
                                          reinterpret_cast<fftw_complex*>(cube.data()),
                                          reinterpret_cast<fftw_complex*>(out.data()),
                                          FFTW_FORWARD, FFTW_ESTIMATE);
        if (!plan) throw std::runtime_error("FFTW plan creation failed");
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // Orthonormal scaling and intensity
        const double scale = 1.0 / std::sqrt(static_cast<double>(out.size()));
        py::array_t<double> res({(py::ssize_t)eff, (py::ssize_t)eff, (py::ssize_t)eff});
        auto r = res.mutable_unchecked<3>();
        for (size_t z = 0; z < eff; ++z) {
            for (size_t y = 0; y < eff; ++y) {
                for (size_t x = 0; x < eff; ++x) {
                    size_t idx = (z * eff + y) * eff + x;
                    std::complex<double> v = out[idx] * scale;
                    r(z, y, x) = std::norm(v);
                }
            }
        }
        return res;
    }

    // Export sampled spatial field for parity testing
    py::array_t<std::complex<double>> export_field(int downsample = -1) const {
        std::lock_guard<std::mutex> lock(mtx);
        size_t ds = (downsample > 0) ? static_cast<size_t>(downsample) : N;
        if (ds > N) ds = N;
        if (ds < 1) ds = N;
        size_t step = (N + ds - 1) / ds;
        size_t eff = (N + step - 1) / step;
        py::array_t<std::complex<double>> res({(py::ssize_t)eff,(py::ssize_t)eff,(py::ssize_t)eff});
        auto r = res.mutable_unchecked<3>();
        size_t xi=0; for (size_t x=0; x<N && xi<eff; x+=step,++xi){
            size_t yi=0; for (size_t y=0; y<N && yi<eff; y+=step,++yi){
                size_t zi=0; for (size_t z=0; z<N && zi<eff; z+=step,++zi){
                    size_t idx=(z*N + y)*N + x; r(zi,yi,xi)=psi[idx];
                }
            }
        }
        return res;
    }

    // Bytes API wrappers
    std::string store_bytes(py::bytes data, const std::string& doc_id) {
        std::string s = data;
        return store(s, doc_id);
    }
    py::bytes retrieve_bytes(const std::string& doc_id) const {
        std::string s = retrieve(doc_id);
        return py::bytes(s);
    }

    py::dict get_stats() const {
        py::dict d;
        d["backend"] = "C++ FFTW 3D";
        d["grid_size"] = (py::int_)N;
        d["total_voxels"] = (py::int_)total;
        d["alloc_ptr"] = (py::int_)alloc_ptr;
        d["documents"] = (py::int_)docs.size();
        return d;
    }
};

PYBIND11_MODULE(holographic_cpp_3d, m) {
    m.doc() = "True 3D spatial holographic memory (exact recall)";
    py::class_<HolographicMemory3D>(m, "HolographicMemory3D")
        .def(py::init<size_t>(), py::arg("grid_size") = 64)
        .def("store", &HolographicMemory3D::store, py::arg("data"), py::arg("doc_id"))
        .def("retrieve", &HolographicMemory3D::retrieve, py::arg("doc_id"))
        .def("delete_document", &HolographicMemory3D::delete_document, py::arg("doc_id"))
        .def("interference_pattern", &HolographicMemory3D::interference_pattern, py::arg("downsample") = 32)
        .def("export_field", &HolographicMemory3D::export_field, py::arg("downsample") = -1)
        .def("store_bytes", &HolographicMemory3D::store_bytes, py::arg("data"), py::arg("doc_id"))
        .def("retrieve_bytes", &HolographicMemory3D::retrieve_bytes, py::arg("doc_id"))
        .def("get_stats", &HolographicMemory3D::get_stats);
}
