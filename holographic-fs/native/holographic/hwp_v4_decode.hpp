#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace holo {

// Decode modern .hwp payloads (H4K8/HWP4V001) to original bytes.
// Throws std::runtime_error on unsupported formats (e.g., H4M1 header-only).
std::vector<std::uint8_t> decode_hwp_v4_to_bytes(const std::uint8_t* data, std::size_t size);

// Convenience: decode from file path
std::vector<std::uint8_t> decode_hwp_v4_file_to_bytes(const std::string& path);

}

