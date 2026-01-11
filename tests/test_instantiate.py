import unittest
from trainer.instantiate import instantiate

class MockObject:
    def __init__(self, value=0):
        self.value = value
        self.attr = None

    def method(self, arg):
        return arg * 2

    def side_effect(self, arg):
        self.attr = arg

class TestInstantiate(unittest.TestCase):
    def test_call_keyword(self):
        # Test explicit call returning value
        obj = MockObject()
        config = {
            "root": {
                "call": {
                    "object": obj,
                    "method": "method",
                    "args": [10]
                }
            }
        }
        instantiate(config, [], config)
        self.assertEqual(config["root"], 20)

    def test_object_property_setting(self):
        # Test object property setting (no replacement)
        obj = MockObject()
        config = {
            "root": {
                "object": obj,
                "value": 100
            }
        }
        instantiate(config, [], config)
        self.assertEqual(obj.value, 100)
        # Verify node is NOT replaced
        self.assertIsInstance(config["root"], dict)
        self.assertEqual(config["root"]["object"], obj)

    def test_object_side_effect(self):
        # Test method call for side effect (no replacement)
        obj = MockObject()
        config = {
            "root": {
                "object": obj,
                "side_effect": {"arg": "test"}
            }
        }
        instantiate(config, [], config)
        self.assertEqual(obj.attr, "test")
        # Verify node is NOT replaced
        self.assertIsInstance(config["root"], dict)

    def test_call_keyword_args_dict(self):
        obj = MockObject()
        config = {
            "root": {
                "call": {
                    "object": obj,
                    "method": "method",
                    "args": {"arg": 5}
                }
            }
        }
        instantiate(config, [], config)
        self.assertEqual(config["root"], 10)

if __name__ == '__main__':
    unittest.main()
