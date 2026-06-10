"""Dispatch tests for buspirate-mcp.

Verifies that server.py call_tool correctly routes every tool name to the
corresponding tools.tool_* function without TypeError or missing arguments.
Tool functions are patched to AsyncMock so this tests ONLY the dispatch
routing and argument unpacking, not tool logic.
"""

import json

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mcp.types import TextContent

from buspirate_mcp import server, tools

TOOL_ARGS = {
    "list_devices": {},
    "verify_connection": {"tx_pin": 4, "rx_pin": 5},
    "scan_baud": {},
    "open_uart": {"baud": 115200, "engagement_name": "test"},
    "read_output": {"session_id": "t"},
    "send_command": {"session_id": "t", "command": "help"},
    "close_uart": {"session_id": "t"},
    "set_voltage": {"voltage_v": 3.3, "current_limit_ma": 100, "_confirmed": True},
    "set_power": {"enable": True, "_confirmed": True},
    "measure_voltage": {"pin": 0},
    "enter_download_mode": {"boot_pin": 0, "reset_pin": 1},
    "read_flash": {"size": 0x1000, "output_path": "/tmp/test.bin"},
    "open_spi": {"engagement_name": "test"},
    "spi_probe": {"session_id": "t"},
    "spi_read": {"session_id": "t", "address": 0, "length": 256},
    "spi_dump": {"session_id": "t"},
    "spi_write": {"session_id": "t", "input_path": "/tmp/fw.bin", "_confirmed": True},
    "spi_transfer": {"session_id": "t", "write_hex": "9f"},
    "close_spi": {"session_id": "t"},
    "open_i2c": {"engagement_name": "test"},
    "i2c_scan": {"session_id": "t"},
    "i2c_read": {"session_id": "t", "device_addr": "0x50"},
    "i2c_write": {
        "session_id": "t",
        "device_addr": "0x50",
        "register_addr": 0,
        "data_hex": "ff",
        "_confirmed": True,
    },
    "i2c_dump": {"session_id": "t", "device_addr": "0x50"},
    "close_i2c": {"session_id": "t"},
    "open_1wire": {"engagement_name": "test"},
    "onewire_search": {"session_id": "t"},
    "onewire_read": {"session_id": "t", "write_hex": "33", "read_bytes": 8},
    "close_1wire": {"session_id": "t"},
    "la_prepare": {"engagement_name": "test", "protocol": "spi"},
    "la_command": {"session_id": "t", "command": "[0x9f r:3]"},
    "la_analyze": {"session_id": "t"},
    "la_identify": {"session_id": "t"},
    "la_cleanup": {"session_id": "t"},
}


@pytest.fixture(autouse=True)
def _mock_globals():
    """Patch module-level globals and all tool functions."""
    mock_hw = MagicMock()
    patches = [
        patch.object(server, "session_manager", MagicMock()),
        patch.object(server, "_hardware", mock_hw),
        patch.object(server, "_hardware_port", "/dev/ttyACM1"),
        patch("buspirate_mcp.server._get_hardware", return_value=mock_hw),
        # read_flash calls BusPirateHardware.find_terminal_port() inline
        patch(
            "buspirate_mcp.server.BusPirateHardware.find_terminal_port",
            return_value="/dev/ttyACM0",
        ),
    ]
    # Patch all tool_* functions
    for name in dir(tools):
        if name.startswith("tool_"):
            patches.append(
                patch.object(
                    tools, name, new_callable=AsyncMock, return_value={"ok": True}
                )
            )
    for p in patches:
        p.start()
    yield
    patch.stopall()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,args", TOOL_ARGS.items())
async def test_dispatch(tool_name, args):
    """call_tool should route {tool_name} without crashing."""
    result = await server.call_tool(tool_name, args)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    data = json.loads(result[0].text)
    assert "Unknown tool" not in data.get("error", "")


async def test_unknown_tool():
    """Unknown tool names raise ValueError from classify_tool."""
    with pytest.raises(ValueError, match="Unknown tool"):
        await server.call_tool("nonexistent_tool", {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confirmed_value",
    [False, None, 0, "", "true", "yes", 1, "True"],
)
async def test_confirmed_requires_literal_true(confirmed_value):
    """Approval-write tools must reject anything that isn't the literal bool True."""
    args = {
        "voltage_v": 3.3,
        "current_limit_ma": 100,
        "_confirmed": confirmed_value,
    }
    result = await server.call_tool("set_voltage", args)
    data = json.loads(result[0].text)
    assert data.get("confirmation_required") is True
