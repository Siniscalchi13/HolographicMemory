#!/usr/bin/env python3
"""
TAI Enterprise Test Runner

This script runs the complete enterprise test suite for the TAI system,
including system startup, E2E integration, and performance/load testing.

Usage:
    python run_enterprise_tests.py [--quick] [--performance] [--load] [--all]
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class EnterpriseTestRunner:
    """Enterprise test runner for TAI system"""
    
    def __init__(self):
        self.test_modules = {
            "startup": "test_tai_system_startup.py",
            "e2e": "test_tai_e2e_integration.py", 
            "performance": "test_tai_performance_load.py"
        }
        
        self.test_results: Dict[str, Any] = {}
    
    def run_test_module(self, module_name: str, verbose: bool = True) -> bool:
        """Run a specific test module"""
        if module_name not in self.test_modules:
            print(f"❌ Unknown test module: {module_name}")
            return False
        
        print(f"🧪 Running {module_name} tests...")
        
        cmd = [
            "python", "-m", "pytest", 
            str(Path(__file__).parent / self.test_modules[module_name]),
            "-v" if verbose else "",
            "--tb=short",
            "--durations=10"
        ]
        
        # Remove empty strings
        cmd = [arg for arg in cmd if arg]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            self.test_results[module_name] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"✅ {module_name} tests passed")
                return True
            else:
                print(f"❌ {module_name} tests failed")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {module_name} tests timed out")
            self.test_results[module_name] = {
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out",
                "success": False
            }
            return False
        except Exception as e:
            print(f"💥 Error running {module_name} tests: {e}")
            self.test_results[module_name] = {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
            return False
    
    def run_quick_tests(self) -> bool:
        """Run quick validation tests"""
        print("🚀 Running quick validation tests...")
        
        # Run startup tests only
        return self.run_test_module("startup")
    
    def run_performance_tests(self) -> bool:
        """Run performance and load tests"""
        print("⚡ Running performance and load tests...")
        
        # Run performance tests
        return self.run_test_module("performance")
    
    def run_e2e_tests(self) -> bool:
        """Run end-to-end integration tests"""
        print("🔗 Running end-to-end integration tests...")
        
        # Run E2E tests
        return self.run_test_module("e2e")
    
    def run_all_tests(self) -> bool:
        """Run complete enterprise test suite"""
        print("🏢 Running complete enterprise test suite...")
        
        all_passed = True
        
        # Run tests in order
        test_order = ["startup", "e2e", "performance"]
        
        for test_module in test_order:
            if not self.run_test_module(test_module):
                all_passed = False
                print(f"❌ Stopping test suite due to {test_module} failure")
                break
            
            # Small delay between test modules
            time.sleep(5)
        
        return all_passed
    
    def generate_report(self) -> str:
        """Generate test report"""
        report = []
        report.append("=" * 80)
        report.append("TAI ENTERPRISE TEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        
        report.append(f"Total Test Modules: {total_tests}")
        report.append(f"Passed: {passed_tests}")
        report.append(f"Failed: {total_tests - passed_tests}")
        report.append(f"Success Rate: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        report.append("")
        
        for module_name, result in self.test_results.items():
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            report.append(f"{module_name.upper()}: {status}")
            
            if not result["success"]:
                report.append(f"  Return Code: {result['returncode']}")
                if result["stderr"]:
                    report.append(f"  Error: {result['stderr'][:200]}...")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "enterprise_test_report.txt"):
        """Save test report to file"""
        report = self.generate_report()
        
        with open(filename, "w") as f:
            f.write(report)
        
        print(f"📄 Test report saved to: {filename}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="TAI Enterprise Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance and load tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end integration tests")
    parser.add_argument("--all", action="store_true", help="Run complete enterprise test suite")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Default to quick tests if no specific option is provided
    if not any([args.quick, args.performance, args.e2e, args.all]):
        args.quick = True
    
    runner = EnterpriseTestRunner()
    
    print("🏢 TAI Enterprise Test Runner")
    print("=" * 50)
    
    success = False
    
    try:
        if args.quick:
            success = runner.run_quick_tests()
        elif args.performance:
            success = runner.run_performance_tests()
        elif args.e2e:
            success = runner.run_e2e_tests()
        elif args.all:
            success = runner.run_all_tests()
        
        # Generate report
        if args.report or not success:
            runner.save_report()
            print(runner.generate_report())
        
        if success:
            print("\n🎉 All tests passed! TAI system is enterprise-ready.")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed. Check the report for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
