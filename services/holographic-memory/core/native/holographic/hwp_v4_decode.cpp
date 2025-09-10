#include "hwp_v4_decode.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>

namespace holo {

namespace {

struct Reader {
    const std::uint8_t* p;
    const std::uint8_t* e;
    Reader(const std::uint8_t* d, std::size_t n): p(d), e(d+n) {}
    bool more(std::size_t n=1) const { return (std::size_t)(e-p) >= n; }
    std::uint8_t u8(){ if(!more()) throw std::runtime_error("EOF"); return *p++; }
    std::uint16_t u16(){ if(!more(2)) throw std::runtime_error("EOF"); std::uint16_t v = (std::uint16_t)(p[0] | (p[1]<<8)); p+=2; return v; }
    std::uint32_t u32(){ if(!more(4)) throw std::runtime_error("EOF"); std::uint32_t v = (std::uint32_t)(p[0] | (p[1]<<8) | (p[2]<<16) | (p[3]<<24)); p+=4; return v; }
    float f32(){ if(!more(4)) throw std::runtime_error("EOF"); float f; std::memcpy(&f, p, 4); p += 4; return f; }
    std::size_t varu(){
        std::size_t v=0; int shift=0; while(true){ if(!more()) throw std::runtime_error("EOF"); std::uint8_t b=*p++; v |= (std::size_t)(b & 0x7Fu) << shift; if(!(b & 0x80u)) break; shift += 7; if(shift>56) throw std::runtime_error("varu overflow"); } return v;
    }
    std::string str(std::size_t n){ if(!more(n)) throw std::runtime_error("EOF"); std::string s((const char*)p, (const char*)p+n); p += n; return s; }
};

struct SparseAccum {
    // accumulate complex spectrum as (re, im) per index
    std::vector<double> re;
    std::vector<double> im;
    void init(std::size_t n){ re.assign(n,0.0); im.assign(n,0.0); }
    void add(std::size_t idx, double amp, double phase){ if(idx>=re.size()) return; re[idx] += amp * std::cos(phase); im[idx] += amp * std::sin(phase); }
};

static std::vector<std::uint8_t> synth_from_sparse(const SparseAccum& S, std::size_t N, std::size_t orig_size){
    // Inverse DFT using only non-zero bins: s[n] = (1/N) * sum_k (Re + i Im) * e^{i 2π k n / N}
    const double two_pi = 2.0 * M_PI;
    std::vector<double> s(N, 0.0);
    for(std::size_t n=0;n<N;++n){
        double sr=0.0, si=0.0;
        for(std::size_t k=0;k<N;++k){
            double ang = two_pi * (double)k * (double)n / (double)N;
            double rk = S.re[k], ik = S.im[k];
            // (rk + i ik) * e^{i ang}
            sr += rk*std::cos(ang) - ik*std::sin(ang);
            si += rk*std::sin(ang) + ik*std::cos(ang);
        }
        s[n] = sr / (double)N; // real signal
    }
    // Normalize to [0,1] by min/max clipping
    double mn = 1e9, mx = -1e9; for(double v: s){ if(v<mn) mn=v; if(v>mx) mx=v; }
    double scale = (mx>mn)? 1.0/(mx-mn) : 1.0;
    std::vector<std::uint8_t> out;
    out.resize(orig_size);
    // Resample or crop/zero-pad to original_size via linear interpolation
    if(orig_size == N){
        for(std::size_t i=0;i<N;++i){ double x=(s[i]-mn)*scale; if(x<0)x=0; if(x>1)x=1; out[i] = (std::uint8_t)std::lround(x*255.0); }
        return out;
    }
    for(std::size_t i=0;i<orig_size;++i){
        double pos = (double)i * (double)(N-1) / (double)(orig_size-1 ? orig_size-1 : 1);
        std::size_t i0 = (std::size_t)std::floor(pos);
        std::size_t i1 = (i0+1 < N)? i0+1 : i0;
        double t = pos - (double)i0;
        double v = (1.0-t)*s[i0] + t*s[i1];
        double x = (v - mn)*scale; if(x<0)x=0; if(x>1)x=1; out[i] = (std::uint8_t)std::lround(x*255.0);
    }
    return out;
}

} // namespace

std::vector<std::uint8_t> decode_hwp_v4_to_bytes(const std::uint8_t* data, std::size_t size){
    if(size < 4) throw std::runtime_error(".hwp too small");
    Reader R(data, size);
    // magic
    std::string magic = R.str(4);
    if(magic == "H4M1"){
        // micro header-only: no coefficients -> cannot reconstruct
        throw std::runtime_error("H4M1 (micro) has no recoverable payload");
    }
    if(magic == "H4K8"){
        // flags
        (void)R.u8();
        // doc_id8
        (void)R.str(8);
        std::size_t orig = R.varu();
        std::size_t dim = R.varu();
        std::size_t k = R.varu();
        float amp_scale = R.f32();
        SparseAccum S; S.init(dim);
        const double two_pi = 2.0 * M_PI;
        for(std::size_t i=0;i<k;++i){
            std::size_t idx = R.varu();
            double amp = (double)R.u8() * (amp_scale / 255.0);
            double phs = (double)R.u16() * (two_pi / 1023.0) - M_PI;
            S.add(idx, amp, phs);
        }
        return synth_from_sparse(S, dim, orig);
    }
    // If header is 8 bytes, we already consumed 4; read next 4 to check HWP4V001
    std::string magic2 = magic + R.str(4);
    if(magic2 == "HWP4V001"){
        std::uint8_t version = R.u8(); (void)version;
        std::uint8_t flags = R.u8(); (void)flags;
        // doc_id
        std::size_t did_len = R.varu(); (void)R.str(did_len);
        // filename, orig size, content type
        std::size_t fn_len = R.varu(); (void)R.str(fn_len);
        std::size_t orig = R.varu();
        std::size_t ct_len = R.varu(); (void)R.str(ct_len);
        std::uint32_t dim = R.u32();
        std::size_t layer_count = R.varu();
        SparseAccum S; S.init(dim);
        const double two_pi = 2.0 * M_PI;
        for(std::size_t L=0; L<layer_count; ++L){
            std::size_t name_len = R.varu(); (void)R.str(name_len);
            std::size_t k = R.varu();
            float amp_scale = R.f32();
            std::vector<std::size_t> idx(k);
            for(std::size_t i=0;i<k;++i) idx[i] = R.varu();
            std::vector<std::uint8_t> amps(k);
            for(std::size_t i=0;i<k;++i) amps[i] = R.u8();
            for(std::size_t i=0;i<k;++i){
                std::uint16_t ph = R.u16();
                double amp = (double)amps[i] * (amp_scale / 255.0);
                double phs = (double)ph * (two_pi / 1023.0) - M_PI;
                S.add(idx[i], amp, phs);
            }
        }
        return synth_from_sparse(S, dim, orig);
    }
    throw std::runtime_error("Unsupported .hwp magic");
}

std::vector<std::uint8_t> decode_hwp_v4_file_to_bytes(const std::string& path){
    std::ifstream ifs(path, std::ios::binary);
    if(!ifs) throw std::runtime_error("Failed to open .hwp file");
    std::vector<std::uint8_t> buf((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    if(buf.empty()) throw std::runtime_error("Empty .hwp file");
    return decode_hwp_v4_to_bytes(buf.data(), buf.size());
}

} // namespace holo

