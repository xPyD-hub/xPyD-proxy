# SPDX-License-Identifier: Apache-2.0
"""Unit tests for NodeDiscovery model auto-detection (S5)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from xpyd.discovery import NodeDiscovery
from xpyd.registry import InstanceRegistry


class TestProbeModels:
    """Tests for _probe_models updating registry from /v1/models."""

    @pytest.fixture()
    def registry(self):
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="")
        reg.mark_healthy("10.0.0.1:8000")
        return reg

    @pytest.fixture()
    def discovery(self, registry):
        return NodeDiscovery(
            prefill_instances=["10.0.0.1:8000"],
            decode_instances=[],
            registry=registry,
        )

    @pytest.mark.asyncio()
    async def test_probe_models_updates_registry(self, discovery, registry):
        """_probe_models calls /v1/models and updates registry model field."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"data": [{"id": "llama-3", "object": "model"}]}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await discovery._probe_models(mock_session, "10.0.0.1:8000")

        info = registry.get_instance_info("10.0.0.1:8000")
        assert info.model == "llama-3"

    @pytest.mark.asyncio()
    async def test_probe_models_non_200_graceful(self, discovery, registry):
        """_probe_models handles non-200 response gracefully without updating."""
        mock_resp = AsyncMock()
        mock_resp.status = 404
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await discovery._probe_models(mock_session, "10.0.0.1:8000")

        info = registry.get_instance_info("10.0.0.1:8000")
        assert info.model == ""

    @pytest.mark.asyncio()
    async def test_probe_models_unexpected_format(self, discovery, registry):
        """_probe_models handles unexpected JSON format gracefully."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"unexpected": "format"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await discovery._probe_models(mock_session, "10.0.0.1:8000")

        info = registry.get_instance_info("10.0.0.1:8000")
        assert info.model == ""

    @pytest.mark.asyncio()
    async def test_probe_models_empty_data_list(self, discovery, registry):
        """_probe_models handles empty data list gracefully."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"data": []})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)

        await discovery._probe_models(mock_session, "10.0.0.1:8000")

        info = registry.get_instance_info("10.0.0.1:8000")
        assert info.model == ""

    @pytest.mark.asyncio()
    async def test_probe_models_connection_error(self, discovery, registry):
        """_probe_models handles connection errors gracefully."""
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Connection refused"))

        await discovery._probe_models(mock_session, "10.0.0.1:8000")

        info = registry.get_instance_info("10.0.0.1:8000")
        assert info.model == ""

    @pytest.mark.asyncio()
    async def test_probe_models_skips_when_no_registry(self):
        """_probe_models returns immediately when registry is None."""
        discovery = NodeDiscovery(
            prefill_instances=["10.0.0.1:8000"],
            decode_instances=[],
            registry=None,
        )
        mock_session = MagicMock()
        # Should not call session.get at all
        await discovery._probe_models(mock_session, "10.0.0.1:8000")
        mock_session.get.assert_not_called()

    @pytest.mark.asyncio()
    async def test_probe_models_skips_when_model_already_set(self):
        """_probe_models skips /v1/models probe when model is already known."""
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="llama-3")
        reg.mark_healthy("10.0.0.1:8000")

        discovery = NodeDiscovery(
            prefill_instances=["10.0.0.1:8000"],
            decode_instances=[],
            registry=reg,
        )
        mock_session = MagicMock()
        await discovery._probe_models(mock_session, "10.0.0.1:8000")

        # Should not probe since model is already set
        mock_session.get.assert_not_called()
        info = reg.get_instance_info("10.0.0.1:8000")
        assert info.model == "llama-3"


class TestRegistryUpdateModel:
    """Tests for registry.update_model()."""

    def test_update_model_changes_field(self):
        """update_model correctly modifies an existing instance's model."""
        reg = InstanceRegistry()
        reg.add("prefill", "10.0.0.1:8000", model="")
        reg.update_model("10.0.0.1:8000", "llama-3")
        info = reg.get_instance_info("10.0.0.1:8000")
        assert info.model == "llama-3"

    def test_update_model_overwrites_existing(self):
        """update_model overwrites a previously set model name."""
        reg = InstanceRegistry()
        reg.add("decode", "10.0.0.2:8000", model="old-model")
        reg.update_model("10.0.0.2:8000", "new-model")
        info = reg.get_instance_info("10.0.0.2:8000")
        assert info.model == "new-model"

    def test_update_model_unknown_address_raises(self):
        """update_model raises KeyError for unregistered address."""
        reg = InstanceRegistry()
        with pytest.raises(KeyError):
            reg.update_model("10.0.0.99:8000", "llama-3")
