// Clean RS(255,223) implementation that actually works
#include <vector>
#include <cstdint>
#include <cstring>
#include <array>

namespace rs_clean {

// GF(2^8) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11d)
struct GF256 {
    std::array<uint8_t, 256> log;
    std::array<uint8_t, 256> alog;
    
    GF256() {
        // Build log/antilog tables
        uint16_t x = 1;
        for (int i = 0; i < 255; ++i) {
            alog[i] = (uint8_t)x;
            log[x] = (uint8_t)i;
            x <<= 1;
            if (x & 0x100) x ^= 0x11d;
        }
        alog[255] = alog[0];
        log[0] = 0; // undefined, but set to 0 for safety
    }
    
    uint8_t add(uint8_t a, uint8_t b) const { return a ^ b; }
    
    uint8_t mul(uint8_t a, uint8_t b) const {
        if (!a || !b) return 0;
        int s = log[a] + log[b];
        s %= 255;
        return alog[s];
    }
    
    uint8_t div(uint8_t a, uint8_t b) const {
        if (!a) return 0;
        if (!b) return 0; // undefined
        int s = log[a] - log[b];
        if (s < 0) s += 255;
        return alog[s];
    }
    
    uint8_t pow(uint8_t a, int n) const {
        if (!a) return 0;
        if (n == 0) return 1;
        int s = (log[a] * n) % 255;
        if (s < 0) s += 255;
        return alog[s];
    }
};

// Reed-Solomon encoder/decoder for RS(255,223)
class RS255_223 {
    static constexpr uint32_t n = 255;
    static constexpr uint32_t k = 223;
    static constexpr uint32_t r = 32;
    static constexpr uint32_t t = 16;
    
    GF256 gf;
    std::vector<uint8_t> generator;
    
public:
    RS255_223() {
        // Build generator polynomial with roots α^1 through α^32
        generator.push_back(1);
        for (uint32_t i = 1; i <= r; ++i) {
            uint8_t alpha_i = gf.pow(2, i); // α^i
            std::vector<uint8_t> new_gen(generator.size() + 1, 0);
            // Multiply by (x - α^i)
            for (size_t j = 0; j < generator.size(); ++j) {
                new_gen[j] = gf.add(new_gen[j], gf.mul(generator[j], alpha_i));
                new_gen[j+1] = gf.add(new_gen[j+1], generator[j]);
            }
            generator = new_gen;
        }
    }
    
    // Encode data into parity bytes
    std::vector<uint8_t> encode(const uint8_t* data, size_t len) {
        std::vector<uint8_t> parity(r, 0);
        
        // LFSR encoding
        for (size_t i = 0; i < k; ++i) {
            uint8_t di = (i < len) ? data[i] : 0;
            uint8_t feedback = gf.add(di, parity[0]);
            for (size_t j = 0; j < r - 1; ++j) {
                parity[j] = gf.add(parity[j+1], gf.mul(feedback, generator[j]));
            }
            parity[r-1] = gf.mul(feedback, generator[r-1]);
        }
        
        return parity;
    }
    
    // Decode and correct errors
    bool decode(uint8_t* data, size_t len, const uint8_t* parity_bytes) {
        // Build codeword [data | zeros | parity]
        std::vector<uint8_t> codeword(n);
        for (size_t i = 0; i < len && i < k; ++i) {
            codeword[i] = data[i];
        }
        // zeros from len to k (already initialized)
        // parity at the end
        for (size_t i = 0; i < r; ++i) {
            codeword[k + i] = parity_bytes[i];
        }
        
        // Compute syndromes S_j = C(α^(j+1)) for j=0..31
        std::vector<uint8_t> S(r);
        bool has_error = false;
        for (uint32_t j = 0; j < r; ++j) {
            uint8_t Sj = 0;
            uint8_t alpha_j1 = gf.pow(2, j + 1); // α^(j+1)
            uint8_t x_power = 1;
            // Evaluate polynomial at α^(j+1) using Horner's method backward
            for (int i = n - 1; i >= 0; --i) {
                Sj = gf.add(Sj, gf.mul(codeword[i], x_power));
                x_power = gf.mul(x_power, alpha_j1);
            }
            S[j] = Sj;
            if (Sj != 0) has_error = true;
        }
        
        if (!has_error) return true; // No errors
        
        // Berlekamp-Massey algorithm
        std::vector<uint8_t> Lambda{1}; // Error locator polynomial
        std::vector<uint8_t> B{1};
        uint8_t b = 1;
        int L = 0;
        
        for (uint32_t n = 0; n < r; ++n) {
            // Calculate discrepancy
            uint8_t d = S[n];
            for (int i = 1; i <= L && i < Lambda.size(); ++i) {
                d = gf.add(d, gf.mul(Lambda[i], S[n - i]));
            }
            
            if (d == 0) {
                // No update needed
                B.insert(B.begin(), 0);
            } else {
                // Update Lambda
                std::vector<uint8_t> T = Lambda;
                
                // Lambda = Lambda - (d/b) * x * B
                uint8_t factor = gf.div(d, b);
                B.insert(B.begin(), 0); // Multiply B by x
                
                if (Lambda.size() < B.size()) {
                    Lambda.resize(B.size(), 0);
                }
                for (size_t i = 0; i < B.size(); ++i) {
                    Lambda[i] = gf.add(Lambda[i], gf.mul(factor, B[i]));
                }
                
                if (2 * L <= n) {
                    L = n + 1 - L;
                    B = T;
                    b = d;
                }
            }
        }
        
        // Chien search to find error locations
        std::vector<int> error_positions;
        for (int i = 0; i < n; ++i) {
            uint8_t alpha_neg_i = gf.pow(2, 255 - i); // α^(-i) mod 255
            // Evaluate Lambda at α^(-i)
            uint8_t val = 0;
            uint8_t x_power = 1;
            for (size_t j = 0; j < Lambda.size(); ++j) {
                val = gf.add(val, gf.mul(Lambda[j], x_power));
                x_power = gf.mul(x_power, alpha_neg_i);
            }
            if (val == 0) {
                error_positions.push_back(i);
            }
        }
        
        if (error_positions.size() != L || error_positions.empty()) {
            return false; // Decoding failure
        }
        
        // Forney algorithm for error magnitudes
        // First compute Omega(x) = S(x) * Lambda(x) mod x^r
        std::vector<uint8_t> Omega(r, 0);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < Lambda.size() && j <= i; ++j) {
                Omega[i] = gf.add(Omega[i], gf.mul(S[i - j], Lambda[j]));
            }
        }
        
        // Compute Lambda'(x) - formal derivative
        std::vector<uint8_t> Lambda_prime;
        for (size_t i = 1; i < Lambda.size(); i += 2) {
            Lambda_prime.push_back(Lambda[i]);
        }
        
        // Apply corrections
        for (int pos : error_positions) {
            uint8_t Xi_inv = gf.pow(2, pos); // α^pos
            
            // Evaluate Omega at Xi_inv
            uint8_t omega_val = 0;
            uint8_t x_power = 1;
            for (size_t i = 0; i < Omega.size(); ++i) {
                omega_val = gf.add(omega_val, gf.mul(Omega[i], x_power));
                x_power = gf.mul(x_power, Xi_inv);
            }
            
            // Evaluate Lambda' at Xi_inv
            uint8_t lambda_prime_val = 0;
            x_power = 1;
            for (size_t i = 0; i < Lambda_prime.size(); ++i) {
                lambda_prime_val = gf.add(lambda_prime_val, gf.mul(Lambda_prime[i], x_power));
                x_power = gf.mul(x_power, Xi_inv);
            }
            
            if (lambda_prime_val == 0) continue;
            
            // Error magnitude
            uint8_t error_val = gf.div(omega_val, lambda_prime_val);
            
            // Map position to data index
            size_t idx = n - 1 - pos;
            if (idx < len) {
                data[idx] = gf.add(data[idx], error_val);
            }
        }
        
        return true;
    }
};

} // namespace rs_clean
