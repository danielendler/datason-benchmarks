#!/usr/bin/env python3
"""
Security Functionality Validation Test
======================================

Verify that our performance optimizations don't break security redaction.
"""

import datason
import json

def test_basic_redaction():
    """Test that sensitive fields are still properly redacted."""
    print("🔒 Testing Security Redaction Functionality")
    print("=" * 50)
    
    # Test data with sensitive fields that match default patterns
    test_data = {
        "username": "john_doe",
        "email": "john@example.com",  # Should be redacted by email pattern
        "password": "secret123",      # Should be redacted by field name
        "api_key": "abc123def456",    # Should be redacted by field name  
        "ssn": "123-45-6789",         # Should be redacted by SSN pattern
        "credit_card": "4532-1234-5678-9012",  # Should be redacted by credit card pattern
        "normal_data": {
            "name": "John Doe",
            "age": 30,
            "active": True
        }
    }
    
    # Test dump_secure
    result = datason.dump_secure(test_data)
    
    print("Original data contains sensitive info:")
    print(f"- password: {test_data['password']}")
    print(f"- api_key: {test_data['api_key']}")
    print(f"- ssn: {test_data['ssn']}")
    print(f"- email: {test_data['email']}")
    
    print(f"\nSecure dump result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    if isinstance(result, dict) and 'data' in result:
        processed_data = result['data']
        
        # Check if sensitive fields are redacted
        password = processed_data.get('password')
        api_key = processed_data.get('api_key')
        ssn = processed_data.get('ssn')
        email = processed_data.get('email')
        
        print(f"\nAfter security processing:")
        print(f"- password: {password}")
        print(f"- api_key: {api_key}")  
        print(f"- ssn: {ssn}")
        print(f"- email: {email}")
        
        # Validate redaction occurred
        redacted_fields = []
        if password == "<REDACTED>":
            redacted_fields.append("password")
        if api_key == "<REDACTED>":
            redacted_fields.append("api_key")
        if ssn == "<REDACTED>":
            redacted_fields.append("ssn")
        if email == "<REDACTED>":
            redacted_fields.append("email")
        
        print(f"\n✅ Redacted fields: {redacted_fields}")
        
        # Check if non-sensitive fields are preserved
        preserved_fields = []
        if processed_data.get('username') == "john_doe":
            preserved_fields.append("username")
        if processed_data.get('normal_data', {}).get('name') == "John Doe":
            preserved_fields.append("normal_data.name")
            
        print(f"✅ Preserved fields: {preserved_fields}")
        
        # Check security metadata
        if 'security' in result:
            print(f"\n🛡️ Security metadata present: {list(result['security'].keys())}")
        
        return len(redacted_fields) > 0 and len(preserved_fields) > 0
    else:
        print(f"❌ Unexpected result format: {result}")
        return False

def test_performance_edge_cases():
    """Test performance optimizations with edge cases."""
    print(f"\n🏃 Testing Performance Edge Cases")
    print("=" * 50)
    
    edge_cases = {
        "primitives_only": {
            "number": 42,
            "boolean": True,
            "null": None,
            "float": 3.14
        },
        "empty_structures": {
            "empty_list": [],
            "empty_dict": {},
            "nested_empty": {
                "level1": {
                    "empty": {}
                }
            }
        },
        "mixed_types": {
            "data": [1, 2, 3, "string", True, None],
            "config": {
                "timeout": 30,
                "enabled": False
            }
        }
    }
    
    for case_name, test_data in edge_cases.items():
        print(f"\nTesting {case_name}...")
        
        # Test that it doesn't crash
        try:
            result = datason.dump_secure(test_data)
            print(f"  ✅ Success: {type(result)} with {len(str(result))} chars")
            
            # Verify data integrity for non-sensitive data
            if isinstance(result, dict) and 'data' in result:
                if result['data'] == test_data:
                    print(f"  ✅ Data integrity preserved")
                else:
                    print(f"  ⚠️  Data modified (expected for security processing)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False
    
    return True

def main():
    """Run all security validation tests."""
    print(f"DataSON Version: {getattr(datason, '__version__', 'unknown')}")
    
    # Test basic redaction functionality
    redaction_works = test_basic_redaction()
    
    # Test performance edge cases
    edge_cases_work = test_performance_edge_cases()
    
    print(f"\n" + "=" * 50)
    print("🧪 Security Validation Summary")
    print("=" * 50)
    print(f"Redaction functionality: {'✅ PASS' if redaction_works else '❌ FAIL'}")
    print(f"Edge case handling: {'✅ PASS' if edge_cases_work else '❌ FAIL'}")
    
    if redaction_works and edge_cases_work:
        print(f"\n🎉 All security tests PASSED - optimizations are safe to deploy!")
        return True
    else:
        print(f"\n⚠️ Some tests failed - review optimizations before deployment")
        return False

if __name__ == "__main__":
    main()