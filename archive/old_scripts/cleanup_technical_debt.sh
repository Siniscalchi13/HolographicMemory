#!/bin/bash
# HolographicMemory Technical Debt Cleanup Script
# WARNING: This will delete identified unnecessary files
# Review carefully before running!

echo "HolographicMemory Technical Debt Cleanup"
echo "========================================"
echo ""
echo "This script will remove identified technical debt:"
echo "- Test files in root directory"
echo "- Redundant demo directories"
echo "- Embedded Python runtime (302MB)"
echo ""
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Create backup directory
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Function to safely move files
safe_move() {
    if [ -e "$1" ]; then
        echo "Moving $1 to backup..."
        mv "$1" "$BACKUP_DIR/" 2>/dev/null || echo "  Failed to move $1"
    fi
}

echo ""
echo "Phase 1: Removing test files from root..."
echo "-----------------------------------------"
safe_move "test_codex_fixes.txt"
safe_move "test_compression.txt"
safe_move "test_consistent.txt"
safe_move "test_download.docx"
safe_move "test_download2.docx"
safe_move "test_download3.docx"
safe_move "test_download4.docx"
safe_move "test_download5.docx"
safe_move "test_file.txt"
safe_move "test_file2.txt"
safe_move "test_final_comprehensive.txt"
safe_move "test_final.txt"
safe_move "test_identity.txt"
safe_move "test_knowledge.txt"
safe_move "test_math_validation.txt"
safe_move "test_mathematical_solution.txt"
safe_move "test_pagination.html"
safe_move "test_preference.txt"
safe_move "test_real_waves.txt"
safe_move "test_v4_large.bin"
safe_move "test_v4_reconstructed.txt"
safe_move "test_v4_small.txt"
safe_move "test_vault.txt"
safe_move "knowledge_reconstructed.txt"
safe_move "vault_reconstructed.txt"
safe_move "small_test.txt"

echo ""
echo "Phase 2: Cleaning up demo directories..."
echo "-----------------------------------------"
# Keep demo/ as the main demo, remove others
safe_move "demo_index2"
safe_move "demo_live"
safe_move "demo_native"
safe_move "demo_recall"

echo ""
echo "Phase 3: Embedded Python Runtime (302MB)..."
echo "-----------------------------------------"
echo "WARNING: This will remove the macOS app's embedded Python runtime."
echo "The app will need to be rebuilt with system Python instead."
read -p "Remove embedded Python runtime? (yes/no): " remove_python

if [ "$remove_python" = "yes" ]; then
    safe_move "macos/HolographicMemoryApp/Runtime"
    echo "Embedded Python runtime moved to backup."
else
    echo "Skipping Python runtime removal."
fi

echo ""
echo "Phase 4: Creating proper .gitignore entries..."
echo "----------------------------------------------"
cat >> .gitignore << 'EOF'

# Test outputs
test_*.txt
test_*.html
test_*.docx
test_*.bin
*_reconstructed.txt

# Demo directories (keep only demo/)
demo_*/
!demo/

# macOS embedded runtime (should not be in repo)
macos/HolographicMemoryApp/Runtime/

# Backup directories
backup_*/
EOF

echo "Updated .gitignore file"

echo ""
echo "Phase 5: Summary"
echo "----------------"
echo "Backup created in: $BACKUP_DIR"
echo ""
echo "Space freed:"
du -sh "$BACKUP_DIR" 2>/dev/null || echo "Unable to calculate"

echo ""
echo "IMPORTANT NEXT STEPS:"
echo "1. Review the backup directory to ensure nothing critical was moved"
echo "2. Implement the C++ quantum core as specified in TECHNICAL_AUDIT_REPORT.md"
echo "3. Create services/math_core implementation"
echo "4. Rebuild macOS app with system Python instead of embedded runtime"
echo "5. Run 'git add -A' and commit the cleanup"
echo ""
echo "To restore files: mv $BACKUP_DIR/* ."
echo ""
echo "Cleanup complete!"
