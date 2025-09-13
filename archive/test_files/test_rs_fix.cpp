// Simple RS tail block fix - testing the correct syndrome computation
// The key insight: for shortened RS codes, we need to adjust syndrome computation
// to account for the virtual leading zeros

#include <vector>
#include <cstdint>
#include <cstring>

// For tail block of length L < k:
// The encoder treats it as [data_L | zeros_(k-L)] and computes parity
// The decoder needs to compute syndromes accounting for this

std::vector<uint8_t> compute_syndromes_for_tail(
    const uint8_t* codeword,  // Full codeword [data|zeros|parity]
    uint32_t n,                // n = k + r (255)
    uint32_t k,                // k = 223
    uint32_t r,                // r = 32
    uint32_t chunk_len,       // Actual data length < k
    const uint8_t* logt,
    const uint8_t* alogt)
{
    std::vector<uint8_t> S(r, 0);
    uint32_t pad = k - chunk_len;  // Number of virtual leading zeros
    
    // Key fix: The syndrome computation needs to account for shortened code
    // The conventional RS syndrome is S_j = C(α^j) where C(x) = Σ c_i * x^i
    // But our encoder uses roots α^1 to α^32, so S_j = C(α^(j+1))
    
    for (uint32_t j = 0; j < r; ++j) {
        uint8_t Sj = 0;
        
        // Process actual data bytes [0..chunk_len)
        for (uint32_t i = 0; i < chunk_len; ++i) {
            uint8_t c = codeword[i];
            if (c) {
                // Position in full codeword (accounting for virtual zeros)
                // The encoder sees this as position (pad + i) in a full block
                int power = ((n - 1 - (pad + i)) * (j + 1)) % 255;
                if (power < 0) power += 255;
                Sj ^= alogt[(logt[c] + power) % 255];
            }
        }
        
        // Virtual zeros don't contribute (they're 0)
        // Skip positions [chunk_len..k)
        
        // Process parity bytes [k..n)
        for (uint32_t i = k; i < n; ++i) {
            uint8_t c = codeword[i];
            if (c) {
                int power = ((n - 1 - i) * (j + 1)) % 255;
                if (power < 0) power += 255;
                Sj ^= alogt[(logt[c] + power) % 255];
            }
        }
        
        S[j] = Sj;
    }
    
    return S;
}

// Alternative approach: Process as if zeros were present
std::vector<uint8_t> compute_syndromes_tail_v2(
    const uint8_t* recv,       // [data_L | zeros | parity]
    uint32_t n,
    uint32_t k, 
    uint32_t r,
    uint32_t chunk_len,
    const uint8_t* logt,
    const uint8_t* alogt)
{
    std::vector<uint8_t> S(r, 0);
    
    // Standard syndrome: S_j = Σ recv[i] * α^((n-1-i)*(j+1))
    for (uint32_t j = 0; j < r; ++j) {
        uint8_t Sj = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint8_t c = recv[i];
            if (c) {
                int exp = ((n - 1 - i) * (j + 1)) % 255;
                if (exp < 0) exp += 255;
                Sj ^= alogt[(logt[c] + exp) % 255];
            }
        }
        S[j] = Sj;
    }
    
    return S;
}
