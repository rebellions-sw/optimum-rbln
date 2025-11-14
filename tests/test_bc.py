import logging
import unittest
import warnings

from optimum.rbln import __version__
from optimum.rbln.utils.deprecation import deprecate_kwarg


INFINITE_VERSION = "9999.0.0"


class DeprecationDecoratorTester(unittest.TestCase):
    """Test cases for deprecate_kwarg decorator organized by scenario."""

    # ============================================================================
    # Scenario A: Rename (Key Name Change)
    # ============================================================================

    def test_scenario_a_rename_basic(self):
        """Test basic rename functionality."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            @deprecate_kwarg("deprecated_name", new_name="new_name", version=INFINITE_VERSION)
            def dummy_function(new_name=None, other_name=None):
                return new_name, other_name

            # Test: deprecated name is renamed to new name
            value, other_value = dummy_function(deprecated_name="old_value")
            self.assertEqual(value, "old_value")
            self.assertIsNone(other_value)

            # Test: when both names provided, new name takes precedence
            value, other_value = dummy_function(deprecated_name="old_value", new_name="new_value")
            self.assertEqual(value, "new_value")
            self.assertIsNone(other_value)

    def test_scenario_a_rename_multiple(self):
        """Test multiple keyword arguments can be renamed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            @deprecate_kwarg("deprecated_name1", new_name="new_name1", version=INFINITE_VERSION)
            @deprecate_kwarg("deprecated_name2", new_name="new_name2", version=INFINITE_VERSION)
            def dummy_function(new_name1=None, new_name2=None, other_name=None):
                return new_name1, new_name2, other_name

            value1, value2, other_value = dummy_function(deprecated_name1="old_value1", deprecated_name2="old_value2")
            self.assertEqual(value1, "old_value1")
            self.assertEqual(value2, "old_value2")
            self.assertIsNone(other_value)

    def test_scenario_a_warning(self):
        """Test warning is raised when using deprecated name."""

        @deprecate_kwarg(
            "deprecated_name",
            new_name="new_name",
            version=INFINITE_VERSION,
            raise_if_greater_or_equal_version=False,
        )
        def dummy_function(new_name=None, other_name=None):
            return new_name, other_name

        with self.assertLogs("optimum.rbln.utils.deprecation", level=logging.WARNING) as log:
            value, other_value = dummy_function(deprecated_name="old_value")
            # Verify warning message content
            message = log.records[0].getMessage()
            self.assertIn("will be removed", message)
        self.assertEqual(value, "old_value")
        self.assertIsNone(other_value)

    def test_scenario_a_raise_if_both_names(self):
        """Test ValueError is raised when both names provided and raise_if_both_names=True."""

        @deprecate_kwarg(
            "deprecated_name",
            new_name="new_name",
            version=INFINITE_VERSION,
            raise_if_both_names=True,
        )
        def dummy_function(new_name=None, other_name=None):
            return new_name, other_name

        with self.assertRaises(ValueError) as context:
            dummy_function(deprecated_name="old_value", new_name="new_value")
        # Verify error message content
        message = str(context.exception)
        self.assertIn("ignoring deprecated", message)

    # ============================================================================
    # Scenario B: Value Type Change
    # ============================================================================

    def test_scenario_b_value_replacer_required(self):
        """Test ValueError is raised when deprecated_type is set without value_replacer."""

        @deprecate_kwarg(
            "value",
            deprecated_type=bool,
            version=INFINITE_VERSION,
            raise_if_greater_or_equal_version=False,
        )
        def dummy_function(value=None):
            return value

        with self.assertRaises(ValueError) as context:
            dummy_function(value=True)
        self.assertIn("value_replacer should be provided", str(context.exception))

    def test_scenario_b_automatic_replacement(self):
        """Test value is automatically replaced when value_replacer is provided."""

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

        with self.assertLogs("optimum.rbln.utils.deprecation", level=logging.WARNING) as log:
            result = dummy_function(value=True)
            # Verify warning message content
            message = log.records[0].getMessage()
            self.assertIn("automatically replaced", message)
        self.assertEqual(result, "true")

    # ============================================================================
    # Scenario C: Deletion
    # ============================================================================

    def test_scenario_c_deletion_warning(self):
        """Test warning is raised and deprecated arg is removed from kwargs."""

        @deprecate_kwarg(
            "deprecated_name",
            version=INFINITE_VERSION,
            raise_if_greater_or_equal_version=False,
        )
        def dummy_function(deprecated_name=None):
            return deprecated_name

        with self.assertLogs("optimum.rbln.utils.deprecation", level=logging.WARNING) as log:
            value = dummy_function(deprecated_name="deprecated_value")
            # Verify warning message content
            message = log.records[0].getMessage()
            self.assertIn("will be removed", message)

        # Deprecated arg is removed, so function receives None
        self.assertIsNone(value)

    def test_scenario_c_additional_message(self):
        """Test additional message is included in warning."""

        @deprecate_kwarg(
            "deprecated_name",
            version=INFINITE_VERSION,
            additional_message="Additional message",
            raise_if_greater_or_equal_version=False,
        )
        def dummy_function(deprecated_name=None):
            return deprecated_name

        with self.assertLogs("optimum.rbln.utils.deprecation", level=logging.WARNING) as log:
            value = dummy_function(deprecated_name="old_value")
            # Verify warning message content including additional message
            message = log.records[0].getMessage()
            self.assertIn("will be removed", message)
            self.assertIn("Additional message", message)
        self.assertIsNone(value)

    # ============================================================================
    # Version-based Error Handling
    # ============================================================================

    def test_raise_for_current_version(self):
        """Test ValueError is raised when current version >= deprecated version (default behavior)."""

        # Test: current version == deprecation version
        @deprecate_kwarg("deprecated_name", version=__version__)
        def dummy_function(deprecated_name=None):
            return deprecated_name

        with self.assertRaises(ValueError) as context:
            dummy_function(deprecated_name="old_value")
        # Verify error message content
        message = str(context.exception)
        self.assertIn("deprecated_name", message)
        self.assertIn("removed starting from", message)

        # Test: current version > deprecation version
        @deprecate_kwarg("deprecated_name", version="0.0.0")
        def dummy_function(deprecated_name=None):
            return deprecated_name

        with self.assertRaises(ValueError) as context:
            dummy_function(deprecated_name="old_value")
        # Verify error message content
        message = str(context.exception)
        self.assertIn("removed starting from", message)
