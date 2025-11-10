import logging
import unittest
import warnings

from optimum.rbln import __version__
from optimum.rbln.utils.depreacate_utils import deprecate_kwarg


INFINITE_VERSION = "9999.0.0"


class DeprecationDecoratorTester(unittest.TestCase):
    def test_rename_kwarg(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            @deprecate_kwarg("deprecated_name", new_name="new_name", version=INFINITE_VERSION)
            def dummy_function(new_name=None, other_name=None):
                return new_name, other_name

            # Test keyword argument is renamed
            value, other_value = dummy_function(deprecated_name="old_value")
            self.assertEqual(value, "old_value")
            self.assertIsNone(other_value)

            # Test deprecated and new args are passed, the new one should be returned
            value, other_value = dummy_function(deprecated_name="old_value", new_name="new_value")
            self.assertEqual(value, "new_value")
            self.assertIsNone(other_value)

    def test_rename_multiple_kwargs(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            @deprecate_kwarg("deprecated_name1", new_name="new_name1", version=INFINITE_VERSION)
            @deprecate_kwarg("deprecated_name2", new_name="new_name2", version=INFINITE_VERSION)
            def dummy_function(new_name1=None, new_name2=None, other_name=None):
                return new_name1, new_name2, other_name

            # Test keyword argument is renamed
            value1, value2, other_value = dummy_function(deprecated_name1="old_value1", deprecated_name2="old_value2")
            self.assertEqual(value1, "old_value1")
            self.assertEqual(value2, "old_value2")
            self.assertIsNone(other_value)

    def test_warnings(self):
        # 1. Test warning is raised for future version
        @deprecate_kwarg(
            "deprecated_name", new_name="new_name", version=INFINITE_VERSION, raise_if_greater_or_equal_version=False
        )
        def dummy_function(new_name=None, other_name=None):
            return new_name, other_name

        # Capture logging warnings - assertLogs will fail if no warning is logged
        with self.assertLogs("optimum.rbln.utils.depreacate_utils", level=logging.WARNING):
            value, other_value = dummy_function(deprecated_name="old_value")
        self.assertEqual(value, "old_value")
        self.assertIsNone(other_value)

        # 2. Test warning is raised for past version when raise_if_greater_or_equal_version=False
        # But arg is still renamed
        @deprecate_kwarg(
            "deprecated_name", new_name="new_name", version="0.0.0", raise_if_greater_or_equal_version=False
        )
        def dummy_function(new_name=None, other_name=None):
            return new_name, other_name

        # Capture logging warnings - assertLogs will fail if no warning is logged
        with self.assertLogs("optimum.rbln.utils.depreacate_utils", level=logging.WARNING):
            value, other_value = dummy_function(deprecated_name="old_value")
        self.assertEqual(value, "old_value")
        self.assertIsNone(other_value)

        # 3. Test warning is raised for past version if raise_if_greater_or_equal_version is False
        # Scenario C: Deletion - deprecated arg is removed from kwargs
        @deprecate_kwarg("deprecated_name", version="0.0.0", raise_if_greater_or_equal_version=False)
        def dummy_function(deprecated_name=None):
            return deprecated_name

        # 4.Capture logging warnings - assertLogs will fail if no warning is logged
        with self.assertLogs("optimum.rbln.utils.depreacate_utils", level=logging.WARNING):
            value = dummy_function(deprecated_name="deprecated_value")
        # In Scenario C (Deletion), the deprecated arg is removed, so function receives None
        self.assertIsNone(value)

        # 5. Test warning is raised for future version if raise_if_greater_or_equal_version is False
        # Scenario C: Deletion - deprecated arg is removed from kwargs
        @deprecate_kwarg("deprecated_name", version=INFINITE_VERSION, raise_if_greater_or_equal_version=False)
        def dummy_function(deprecated_name=None):
            return deprecated_name

        # Capture logging warnings - assertLogs will fail if no warning is logged
        with self.assertLogs("optimum.rbln.utils.depreacate_utils", level=logging.WARNING):
            value = dummy_function(deprecated_name="deprecated_value")
        # In Scenario C (Deletion), the deprecated arg is removed, so function receives None
        self.assertIsNone(value)

    def test_raises(self):
        # 1. Test if deprecated name and new name are both passed and raise_if_both_names is set -> raise error
        @deprecate_kwarg("deprecated_name", new_name="new_name", version=INFINITE_VERSION, raise_if_both_names=True)
        def dummy_function(new_name=None, other_name=None):
            return new_name, other_name

        with self.assertRaises(ValueError):
            dummy_function(deprecated_name="old_value", new_name="new_value")

        # 2. Test for current version == deprecation version (default raise_if_greater_or_equal_version=True)
        @deprecate_kwarg("deprecated_name", version=__version__)
        def dummy_function(deprecated_name=None):
            return deprecated_name

        with self.assertRaises(ValueError):
            dummy_function(deprecated_name="old_value")

        # 3. Test for current version > deprecation version (default raise_if_greater_or_equal_version=True)
        @deprecate_kwarg("deprecated_name", version="0.0.0")
        def dummy_function(deprecated_name=None):
            return deprecated_name

        with self.assertRaises(ValueError):
            dummy_function(deprecated_name="old_value")

    def test_value_deprecation(self):
        # Scenario B: Value Type Change (e.g., bool -> str)

        # 1. Test ValueError is raised when deprecated_type is set without value_replacer
        @deprecate_kwarg(
            "value", deprecated_type=bool, version=INFINITE_VERSION, raise_if_greater_or_equal_version=False
        )
        def dummy_function(value=None):
            return value

        # ValueError should be raised when deprecated_type is used without value_replacer
        with self.assertRaises(ValueError) as context:
            dummy_function(value=True)
        self.assertIn("value_replacer should be provided", str(context.exception))

        # 2. Test warning is raised and value is automatically replaced when value_replacer is provided
        def bool_to_str(value):
            return "true" if value else "false"

        @deprecate_kwarg(
            "value",
            deprecated_type=bool,
            value_replacer=bool_to_str,
            version=INFINITE_VERSION,
            raise_if_greater_or_equal_version=False,
        )
        def dummy_function(value=None):
            return value

        # Capture logging warnings - assertLogs will fail if no warning is logged
        with self.assertLogs("optimum.rbln.utils.depreacate_utils", level=logging.WARNING) as log:
            result = dummy_function(value=True)
            # Warning should mention automatic replacement
            self.assertTrue("automatically replaced" in log.records[0].getMessage())
            self.assertTrue("true" in log.records[0].getMessage())
        # Value should be replaced
        self.assertEqual(result, "true")

    def test_additional_message(self):
        # Test additional message is added to the warning
        # Scenario C: Deletion - deprecated arg is removed from kwargs
        @deprecate_kwarg(
            "deprecated_name",
            version=INFINITE_VERSION,
            additional_message="Additional message",
            raise_if_greater_or_equal_version=False,
        )
        def dummy_function(deprecated_name=None):
            return deprecated_name

        # Capture logging warnings - assertLogs will fail if no warning is logged
        with self.assertLogs("optimum.rbln.utils.depreacate_utils", level=logging.WARNING) as log:
            value = dummy_function(deprecated_name="old_value")
            # Check that additional message is included
            self.assertTrue("Additional message" in log.records[0].getMessage())
        # In Scenario C (Deletion), the deprecated arg is removed, so function receives None
        self.assertIsNone(value)
