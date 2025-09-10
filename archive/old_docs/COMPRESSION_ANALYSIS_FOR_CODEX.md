# Compression Analysis & Download Issues - Codex Review

## 🚨 CRITICAL FINDINGS

### 1. **H4M1 vs H4K8 Format Confusion**
- **H4M1 files**: Header-only, **NON-RECONSTRUCTIBLE**
  - Examples: ARUBA.docx (21KB→18B), PDF (2.6MB→19B)
  - Magic: `H4M1` + metadata only
  - **Cannot be downloaded** - no spectral data stored

- **H4K8 files**: Header + sparse coefficients, **RECONSTRUCTIBLE**
  - Examples: test files (44B→52B, 87B→52B)
  - Magic: `H4K8` + metadata + up to 8 spectral coefficients
  - **Can be downloaded** - contains actual data

### 2. **Storage Logic Bug**
```python
# Current logic in app.py:692-694
if fmt == "micro" and not has3d:
    fmt = "microK8"  # Should create H4K8
# But H4M1 files were created with write_hwp_v4_micro()
```

**Problem**: Large files getting H4M1 (header-only) instead of H4K8 (reconstructible)

### 3. **Compression Ratio Display Issues**
- **Impossible ratios**: 2.6MB→19B = 140,147x compression
- **Expansion shown as compression**: 44B→52B = 0.8x (should show as expansion)
- **UI calculation**: `originalSize / holoSize` - incorrect for expansion

## 🔧 REQUIRED FIXES

### Fix 1: Storage Format Selection
```python
# In app.py store() function
if fmt == "micro":
    if len(data) < 1000:  # Small files
        write_hwp_v4_micro()  # H4M1 header-only
    else:  # Larger files need reconstruction capability
        write_hwp_v4_micro_k8()  # H4K8 with coefficients
```

### Fix 2: Download Error Handling
```python
# In app.py download() function
if p.suffix.lower() == ".hwp":
    # Check file format first
    with open(p, 'rb') as f:
        magic = f.read(4)
    if magic == b'H4M1':
        return {"error": "File stored as header-only (H4M1), cannot reconstruct"}
    elif magic == b'H4K8':
        # Proceed with H4K8 decode
```

### Fix 3: Compression Ratio Display
```javascript
// In app.js
const ratio = holoSize > 0 ? (originalSize / holoSize).toFixed(1) : '—'
const displayRatio = ratio < 1 ? `0.${ratio.split('.')[1]}x` : `${ratio}x`
const compressionType = ratio < 1 ? 'expansion' : 'compression'
```

## 🧮 MATHEMATICAL VALIDATION

### H4M1 Format (Header-Only)
- **Mathematically correct**: Cannot reconstruct from metadata alone
- **Use case**: Tiny files where storage overhead > data size
- **Limitation**: Permanent data loss unless 3D backend available

### H4K8 Format (Sparse Coefficients)
- **Mathematically sound**: Sparse spectral reconstruction
- **Reconstruction**: Inverse FFT from K≤8 coefficients
- **Accuracy**: Depends on sparsity and original signal characteristics

## 📊 CURRENT STATE
- ✅ **New files (H4K8)**: Store and retrieve correctly
- ❌ **Old files (H4M1)**: Cannot be retrieved (by design)
- ❌ **Compression display**: Shows impossible ratios
- ❌ **Storage logic**: Wrong format selection for large files

## 🎯 RECOMMENDATIONS

1. **Immediate**: Fix storage format selection logic
2. **UI**: Correct compression ratio display and add format indicators
3. **Migration**: Consider converting H4M1 files to H4K8 where possible
4. **Documentation**: Clarify H4M1 vs H4K8 use cases and limitations

## 🔍 TESTING VALIDATION

**Files to test after fixes:**
- Small files (<1KB): Should use H4M1, show as "header-only"
- Medium files (1KB-100KB): Should use H4K8, show realistic compression
- Large files (>100KB): Should use H4K8, show realistic compression

**Expected outcomes:**
- All H4K8 files should download successfully
- H4M1 files should show clear "non-reconstructible" message
- Compression ratios should be mathematically plausible
