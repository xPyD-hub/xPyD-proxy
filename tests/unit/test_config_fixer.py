# SPDX-License-Identifier: Apache-2.0
"""Tests for xpyd.config_fixer."""

from __future__ import annotations

import os
import tempfile

import yaml

from xpyd.config_fixer import ConfigFixer, run_fix_config


class TestAutoFixRules:
    """Test each auto-fix rule individually."""

    def test_missing_default_port_legacy(self):
        data = {
            "model": "llama-3",
            "prefill": ["10.0.0.1"],
            "decode": ["10.0.0.2:8000"],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("default port" in f.description for f in report.fixes)
        assert fixer.fixed_data["prefill"] == ["10.0.0.1:8000"]

    def test_missing_default_port_instances(self):
        data = {
            "instances": [
                {"address": "10.0.0.1", "role": "prefill", "model": "m"},
                {"address": "10.0.0.2:8000", "role": "decode", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("default port" in f.description for f in report.fixes)
        assert fixer.fixed_data["instances"][0]["address"] == "10.0.0.1:8000"

    def test_role_case_normalization(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "Prefill", "model": "m"},
                {"address": "10.0.0.2:8000", "role": "DUAL", "model": "m2"},
            ],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["instances"][0]["role"] == "prefill"
        assert fixer.fixed_data["instances"][1]["role"] == "dual"

    def test_role_typo_fuzzy(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "prefil", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert fixer.fixed_data["instances"][0]["role"] == "prefill"
        assert any("role normalized" in f.description for f in report.fixes)

    def test_extra_whitespace_address(self):
        data = {
            "model": "llama-3",
            "prefill": [" 10.0.0.1:8000 "],
            "decode": ["10.0.0.2:8000"],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["prefill"] == ["10.0.0.1:8000"]

    def test_scheduler_typo(self):
        data = {
            "model": "llama-3",
            "decode": ["10.0.0.1:8000"],
            "scheduling": "round_robbin",
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert fixer.fixed_data["scheduling"] == "round_robin"
        assert any("scheduler typo" in f.description for f in report.fixes)

    def test_model_name_whitespace(self):
        data = {
            "model": " llama-3 ",
            "decode": ["10.0.0.1:8000"],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["model"] == "llama-3"

    def test_model_name_whitespace_instances(self):
        data = {
            "instances": [
                {
                    "address": "10.0.0.1:8000",
                    "role": "dual",
                    "model": " llama-3 ",
                },
            ],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["instances"][0]["model"] == "llama-3"

    def test_models_shorthand_address_fix(self):
        data = {
            "models": [
                {
                    "name": "llama-3",
                    "dual": ["10.0.0.1"],
                },
            ],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["models"][0]["dual"] == ["10.0.0.1:8000"]

    def test_models_shorthand_scheduler_typo(self):
        data = {
            "models": [
                {
                    "name": "llama-3",
                    "dual": ["10.0.0.1:8000"],
                    "scheduler": "round_robbin",
                },
            ],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["models"][0]["scheduler"] == "round_robin"

    def test_models_shorthand_name_whitespace(self):
        data = {
            "models": [
                {
                    "name": " llama-3 ",
                    "dual": ["10.0.0.1:8000"],
                },
            ],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["models"][0]["name"] == "llama-3"


class TestShortStringRejection:
    """Fuzzy matching must not match very short strings."""

    def test_short_role_not_matched(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "p", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        fixer.run()
        # "p" should NOT be fuzzy-matched to "prefill"
        assert fixer.fixed_data["instances"][0]["role"] == "p"

    def test_short_scheduler_not_matched(self):
        data = {
            "model": "m",
            "decode": ["10.0.0.1:8000"],
            "scheduling": "rr",
        }
        fixer = ConfigFixer(data)
        fixer.run()
        assert fixer.fixed_data["scheduling"] == "rr"


class TestSuggestions:
    """Test suggest-only rules."""

    def test_dual_pd_mix(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "dual", "model": "m"},
                {"address": "10.0.0.2:8000", "role": "prefill", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("both dual and prefill/decode" in s.message
                    for s in report.suggestions)

    def test_address_conflict(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "dual", "model": "m1"},
                {"address": "10.0.0.1:8000", "role": "dual", "model": "m2"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("multiple models" in s.message for s in report.suggestions)

    def test_unbalanced_pd(self):
        data = {
            "instances": [
                {"address": f"10.0.0.{i}:8000", "role": "prefill", "model": "m"}
                for i in range(1, 6)
            ] + [
                {"address": "10.0.0.10:8000", "role": "decode", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("rebalancing" in s.message for s in report.suggestions)

    def test_missing_decode(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "prefill", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("no decode" in s.message for s in report.suggestions)

    def test_missing_prefill(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "decode", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("no prefill" in s.message for s in report.suggestions)


class TestCleanConfig:
    """A valid config should produce no fixes or suggestions."""

    def test_no_false_positives_legacy(self):
        data = {
            "model": "llama-3",
            "prefill": ["10.0.0.1:8000"],
            "decode": ["10.0.0.2:8000"],
            "scheduling": "loadbalanced",
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert len(report.fixes) == 0
        assert len(report.suggestions) == 0

    def test_no_false_positives_instances(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "prefill", "model": "m"},
                {"address": "10.0.0.2:8000", "role": "decode", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert len(report.fixes) == 0
        assert len(report.suggestions) == 0

    def test_no_false_positives_dual(self):
        data = {
            "instances": [
                {"address": "10.0.0.1:8000", "role": "dual", "model": "m"},
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert len(report.fixes) == 0
        assert len(report.suggestions) == 0


class TestCombinedIssues:
    """Multiple fixes in one config."""

    def test_multiple_fixes(self):
        data = {
            "model": " llama-3 ",
            "prefill": [" 10.0.0.1 "],
            "decode": ["10.0.0.2:8000"],
            "scheduling": "round_robbin",
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        # model whitespace + address whitespace + port + scheduler typo
        assert len(report.fixes) >= 3
        assert fixer.fixed_data["model"] == "llama-3"
        assert fixer.fixed_data["prefill"] == ["10.0.0.1:8000"]
        assert fixer.fixed_data["scheduling"] == "round_robin"


class TestWriteMode:
    """Test --write creates a timestamped backup."""

    def test_write_creates_backup(self):
        data = {
            "model": " llama-3 ",
            "decode": ["10.0.0.1:8000"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test.yaml")
            with open(config_path, "w") as f:
                yaml.dump(data, f)

            exit_code = run_fix_config(config_path, write=True)
            assert exit_code == 0

            # Check backup exists
            bak_files = [
                f for f in os.listdir(tmpdir) if f.endswith(".bak")
            ]
            assert len(bak_files) == 1

            # Check fixed file
            with open(config_path) as f:
                fixed = yaml.safe_load(f)
            assert fixed["model"] == "llama-3"

    def test_file_not_found(self):
        exit_code = run_fix_config("/nonexistent/path.yaml")
        assert exit_code == 1


class TestModelsShorthandSuggestions:
    """Test suggestions for models shorthand format."""

    def test_models_address_conflict(self):
        data = {
            "models": [
                {
                    "name": "m1",
                    "dual": ["10.0.0.1:8000"],
                },
                {
                    "name": "m2",
                    "dual": ["10.0.0.1:8000"],
                },
            ],
        }
        fixer = ConfigFixer(data)
        report = fixer.run()
        assert any("multiple models" in s.message for s in report.suggestions)
