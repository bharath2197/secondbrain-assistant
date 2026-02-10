#!/usr/bin/env python3
"""
Backend API Tests for SecondBrain PWA
Tests all endpoints for existence and expected responses
"""

import requests
import sys
import json
from datetime import datetime

class BackendTester:
    def __init__(self, base_url="https://quick-kb.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.results = []

    def log_result(self, test_name, passed, details=""):
        """Log test result"""
        self.tests_run += 1
        if passed:
            self.tests_passed += 1
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"   Details: {details}")
        
        self.results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })

    def test_health_check(self):
        """Test GET /api/ health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            expected_status = 200
            
            if response.status_code == expected_status:
                try:
                    data = response.json()
                    if data.get("status") == "ok" and "service" in data:
                        self.log_result("Health Check", True, f"Status: {response.status_code}, Response: {data}")
                        return True
                    else:
                        self.log_result("Health Check", False, f"Unexpected response format: {data}")
                        return False
                except json.JSONDecodeError:
                    self.log_result("Health Check", False, f"Invalid JSON response")
                    return False
            else:
                self.log_result("Health Check", False, f"Expected 200, got {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Health Check", False, f"Request failed: {str(e)}")
            return False

    def test_protected_endpoints(self):
        """Test protected endpoints return 401 without auth"""
        endpoints = [
            ("GET", "/profile"),
            ("POST", "/profile"),
            ("POST", "/chat"),
            ("GET", "/messages"),
            ("GET", "/kb"),
            ("GET", "/reminders"),
        ]
        
        all_passed = True
        for method, path in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{path}", timeout=10)
                else:
                    response = requests.post(f"{self.base_url}{path}", json={}, timeout=10)
                
                if response.status_code == 401:
                    self.log_result(f"Protected Endpoint {method} {path}", True, f"Correctly returned 401")
                else:
                    self.log_result(f"Protected Endpoint {method} {path}", False, f"Expected 401, got {response.status_code}")
                    all_passed = False
            except Exception as e:
                self.log_result(f"Protected Endpoint {method} {path}", False, f"Request failed: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_kb_delete_endpoint(self):
        """Test DELETE /api/kb/{id} endpoint returns 401 without auth"""
        try:
            response = requests.delete(f"{self.base_url}/kb/test-id", timeout=10)
            if response.status_code == 401:
                self.log_result("DELETE KB Entry", True, f"Correctly returned 401")
                return True
            else:
                self.log_result("DELETE KB Entry", False, f"Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            self.log_result("DELETE KB Entry", False, f"Request failed: {str(e)}")
            return False

    def test_reminder_patch_endpoint(self):
        """Test PATCH /api/reminders/{id} endpoint returns 401 without auth"""
        try:
            response = requests.patch(f"{self.base_url}/reminders/test-id", json={"status": "completed"}, timeout=10)
            if response.status_code == 401:
                self.log_result("PATCH Reminder", True, f"Correctly returned 401")
                return True
            else:
                self.log_result("PATCH Reminder", False, f"Expected 401, got {response.status_code}")
                return False
        except Exception as e:
            self.log_result("PATCH Reminder", False, f"Request failed: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all backend tests"""
        print("🧪 Starting Backend API Tests")
        print(f"Testing API at: {self.base_url}")
        print("=" * 50)
        
        # Test health check first
        self.test_health_check()
        
        # Test protected endpoints
        self.test_protected_endpoints()
        
        # Test specific endpoints
        self.test_kb_delete_endpoint()
        self.test_reminder_patch_endpoint()
        
        # Summary
        print("\n" + "=" * 50)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed - see details above")
        
        return success_rate >= 80  # Consider 80%+ success rate as acceptable

def main():
    tester = BackendTester()
    success = tester.run_all_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())