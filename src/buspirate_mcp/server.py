"""BusPirate MCP server -- stdio transport.

Registers all tools from tools.py with the MCP SDK and runs the
server. Claude Code spawns this process and communicates over stdin/stdout.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from buspirate_mcp.hardware import BusPirateHardware
from buspirate_mcp.safety import classify_tool, SafetyTier
from buspirate_mcp.session import SessionManager
from buspirate_mcp import tools

logger = logging.getLogger("buspirate-mcp")

# Engagements dir: env var overrides, fallback to package root.
# In standalone mode: defaults to <repo>/engagements/
# When submoduled: parent repo sets PIDEV_ENGAGEMENTS_DIR via .mcp.json env.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENGAGEMENTS_DIR = Path(
    os.environ.get("PIDEV_ENGAGEMENTS_DIR", str(_PACKAGE_ROOT / "engagements"))
)

app = Server("buspirate-mcp")
session_manager = SessionManager(engagements_dir=ENGAGEMENTS_DIR)

# Hardware connection -- initialized lazily on first tool call that needs it
_hardware: BusPirateHardware | None = None
_hardware_port: str = ""


def _get_hardware(port: str | None = None) -> BusPirateHardware:
    global _hardware, _hardware_port
    # Detect dead connection (uart set to None by disconnect or stale state)
    if _hardware is not None and _hardware.uart is None:
        _hardware = None
    if _hardware is None:
        if port is None:
            devices = BusPirateHardware.list_devices()
            if not devices:
                raise RuntimeError("No BusPirate found on USB")
            # Find the binary protocol port; fall back to first device
            binary = [d for d in devices if d["role"] == "binary"]
            port = binary[0]["path"] if binary else devices[0]["path"]
        _hardware = BusPirateHardware.connect(port)
        _hardware_port = port
    return _hardware


TOOL_DEFINITIONS = [
    Tool(
        name="list_devices",
        description="Find BusPirate devices on USB CDC serial ports. [read-only]",
        inputSchema={"type": "object", "properties": {}, "required": []},
    ),
    Tool(
        name="verify_connection",
        description=(
            "Check for signal activity on specified pins. "
            "User must provide pin mapping. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tx_pin": {"type": "integer", "description": "BP IO pin number for TX (e.g., 4)"},
                "rx_pin": {"type": "integer", "description": "BP IO pin number for RX (e.g., 5)"},
                "sample_duration_ms": {
                    "type": "integer", "default": 2000,
                    "description": "How long to sample for activity (ms)",
                },
            },
            "required": ["tx_pin", "rx_pin"],
        },
    ),
    Tool(
        name="scan_baud",
        description=(
            "Scan common baud rates and score by readable ASCII output. "
            "Returns candidates sorted by confidence. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="open_uart",
        description=(
            "Open a persistent UART connection and start an engagement. "
            "Creates engagement folder and raw logging. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "baud": {"type": "integer", "description": "Baud rate"},
                "tx_pin": {"type": "integer", "description": "BP IO pin number for TX (e.g., 4)", "default": 4},
                "rx_pin": {"type": "integer", "description": "BP IO pin number for RX (e.g., 5)", "default": 5},
                "engagement_name": {"type": "string", "description": "Target device name"},
                "project_path": {
                    "type": "string",
                    "description": "Path to a project folder (from project-mcp). If provided, writes to <project_path>/uart/ instead of creating a standalone engagement.",
                },
            },
            "required": ["baud", "engagement_name"],
        },
    ),
    Tool(
        name="read_output",
        description=(
            "Read available data from an open UART session. "
            "Blocks up to timeout. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "timeout_ms": {"type": "integer", "default": 5000},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="send_command",
        description=(
            "Send a command over UART and capture the response. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "command": {"type": "string"},
                "timeout_ms": {"type": "integer", "default": 3000},
            },
            "required": ["session_id", "command"],
        },
    ),
    Tool(
        name="close_uart",
        description="Close a UART session and finalize logs. [allowed-write]",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="set_voltage",
        description=(
            "Set power supply voltage (0.8-5.0V) and current limit (0-500mA). "
            "DANGEROUS: wrong voltage can damage target hardware. [approval-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "voltage_v": {"type": "number", "minimum": 0.8, "maximum": 5.0},
                "current_limit_ma": {"type": "integer", "minimum": 0, "maximum": 500},
                "_confirmed": {"type": "boolean", "default": False, "description": "Set true to confirm execution"},
            },
            "required": ["voltage_v", "current_limit_ma"],
        },
        annotations={"destructiveHint": True, "readOnlyHint": False},
    ),
    Tool(
        name="set_power",
        description=(
            "Enable or disable the power supply. "
            "DANGEROUS: affects power to target device. [approval-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "enable": {"type": "boolean"},
                "_confirmed": {"type": "boolean", "default": False, "description": "Set true to confirm execution"},
            },
            "required": ["enable"],
        },
        annotations={"destructiveHint": True, "readOnlyHint": False},
    ),
    Tool(
        name="measure_voltage",
        description=(
            "Read the DC voltage (mV and V) on a BP IO pin via its ADC. "
            "Use this to read back an analog level — e.g. confirm the PSU "
            "output by jumpering VOUT to an IO pin, or check any DC node. "
            "Returns the pin's voltage plus the full 8-pin snapshot. "
            "(Note: this, not verify_connection, is the voltmeter — "
            "verify_connection only samples one pin for digital activity.) "
            "[read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pin": {
                    "type": "integer", "minimum": 0, "maximum": 7,
                    "description": "BP IO pin number to measure (0-7)",
                },
            },
            "required": ["pin"],
        },
        annotations={"readOnlyHint": True},
    ),
    Tool(
        name="enter_download_mode",
        description=(
            "Put target into UART download mode by toggling GPIO pins. "
            "Holds boot pin LOW, pulses reset pin, releases. "
            "Works for ESP32, ESP8266, and similar chips. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "boot_pin": {
                    "type": "integer",
                    "description": "BP IO pin connected to target boot select (e.g., GPIO0 on ESP32)",
                },
                "reset_pin": {
                    "type": "integer",
                    "description": "BP IO pin connected to target EN/RST",
                },
            },
            "required": ["boot_pin", "reset_pin"],
        },
    ),
    Tool(
        name="read_flash",
        description=(
            "Read flash memory from target via esptool through BP6 UART bridge. "
            "Optionally enters download mode first. Requires esptool installed. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "offset": {
                    "type": "integer", "default": 0,
                    "description": "Flash offset to start reading from",
                },
                "size": {
                    "type": "integer",
                    "description": "Number of bytes to read (e.g., 0x400000 for 4MB)",
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save the flash dump",
                },
                "boot_pin": {
                    "type": "integer", "default": -1,
                    "description": "BP IO pin for boot select (-1 to skip download mode entry)",
                },
                "reset_pin": {
                    "type": "integer", "default": -1,
                    "description": "BP IO pin for reset (-1 to skip download mode entry)",
                },
                "baud": {
                    "type": "integer", "default": 115200,
                    "description": "Baud rate for flash read (115200 or 460800 recommended)",
                },
            },
            "required": ["size", "output_path"],
        },
    ),
    # --- SPI tools ---
    Tool(
        name="open_spi",
        description=(
            "Open an SPI connection and start an engagement. "
            "Configures SPI mode on the BusPirate, creates engagement folder. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "engagement_name": {"type": "string", "description": "Target device name"},
                "speed": {"type": "integer", "default": 1000000, "description": "SPI clock speed in Hz"},
                "clock_polarity": {"type": "boolean", "default": False, "description": "CPOL: clock idle state"},
                "clock_phase": {"type": "boolean", "default": False, "description": "CPHA: data sampling edge"},
                "chip_select_idle": {"type": "boolean", "default": True, "description": "CS idle state (true=high)"},
                "voltage_mv": {"type": "integer", "description": "Target voltage in mV (optional, enables PSU)"},
                "current_ma": {"type": "integer", "description": "Current limit in mA"},
                "project_path": {
                    "type": "string",
                    "description": "Path to a project folder (from project-mcp). If provided, writes to <project_path>/spi/ instead of creating a standalone engagement.",
                },
            },
            "required": ["engagement_name"],
        },
    ),
    Tool(
        name="spi_probe",
        description=(
            "Read JEDEC ID and status register from an SPI flash chip. "
            "Returns manufacturer, capacity, and write-protect status. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="spi_read",
        description=(
            "Read a region of SPI flash memory starting at a given address. "
            "Saves to artifacts. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "address": {"type": "integer", "description": "Flash address to start reading from"},
                "length": {"type": "integer", "description": "Number of bytes to read"},
            },
            "required": ["session_id", "address", "length"],
        },
    ),
    Tool(
        name="spi_dump",
        description=(
            "Dump the entire SPI flash to a file. Auto-detects size from JEDEC ID "
            "if not provided. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "size": {"type": "integer", "description": "Flash size in bytes (auto-detected if omitted)"},
                "output_filename": {"type": "string", "default": "flash_dump.bin", "description": "Output filename"},
                "chunk_size": {"type": "integer", "default": 512, "description": "Read chunk size in bytes"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="spi_write",
        description=(
            "Write a binary file to SPI flash. Optionally erases first and verifies. "
            "DANGEROUS: overwrites flash contents irreversibly. [approval-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "input_path": {"type": "string", "description": "Path to binary file to write"},
                "erase": {"type": "boolean", "default": True, "description": "Chip erase before write"},
                "verify": {"type": "boolean", "default": True, "description": "Read-back verify after write"},
                "_confirmed": {"type": "boolean", "default": False, "description": "Set true to confirm execution"},
            },
            "required": ["session_id", "input_path"],
        },
        annotations={"destructiveHint": True, "readOnlyHint": False},
    ),
    Tool(
        name="spi_transfer",
        description=(
            "Send a raw SPI transaction. Write hex bytes and optionally read back. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "write_hex": {"type": "string", "description": "Hex string of bytes to send (e.g. '9F')"},
                "read_bytes": {"type": "integer", "default": 0, "description": "Number of bytes to read back"},
            },
            "required": ["session_id", "write_hex"],
        },
    ),
    Tool(
        name="close_spi",
        description="Close an SPI session and reset the BusPirate to HiZ mode. [allowed-write]",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    # --- I2C tools ---
    Tool(
        name="open_i2c",
        description=(
            "Open an I2C connection and start an engagement. "
            "Configures I2C mode with pullups enabled. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "engagement_name": {"type": "string", "description": "Target device name"},
                "speed": {"type": "integer", "default": 400000, "description": "I2C clock speed in Hz"},
                "clock_stretch": {"type": "boolean", "default": False, "description": "Enable clock stretching"},
                "voltage_mv": {"type": "integer", "description": "Target voltage in mV (optional, enables PSU)"},
                "current_ma": {"type": "integer", "description": "Current limit in mA"},
                "project_path": {
                    "type": "string",
                    "description": "Path to a project folder (from project-mcp). If provided, writes to <project_path>/i2c/ instead of creating a standalone engagement.",
                },
            },
            "required": ["engagement_name"],
        },
    ),
    Tool(
        name="i2c_scan",
        description=(
            "Scan the I2C bus for devices. Returns 7-bit addresses with device hints. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="i2c_read",
        description=(
            "Read bytes from an I2C device, optionally from a specific register. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "device_addr": {
                    "type": ["string", "integer"],
                    "description": "7-bit device address (hex string like '0x50' or integer)",
                },
                "register_addr": {"type": "integer", "description": "Register address to read from (optional)"},
                "length": {"type": "integer", "default": 1, "description": "Number of bytes to read"},
            },
            "required": ["session_id", "device_addr"],
        },
    ),
    Tool(
        name="i2c_write",
        description=(
            "Write bytes to an I2C device register. "
            "DANGEROUS: may alter device configuration or EEPROM contents. [approval-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "device_addr": {
                    "type": ["string", "integer"],
                    "description": "7-bit device address (hex string like '0x50' or integer)",
                },
                "register_addr": {"type": "integer", "description": "Register address to write to"},
                "data_hex": {"type": "string", "description": "Hex string of data bytes to write"},
                "_confirmed": {"type": "boolean", "default": False, "description": "Set true to confirm execution"},
            },
            "required": ["session_id", "device_addr", "register_addr", "data_hex"],
        },
        annotations={"destructiveHint": True, "readOnlyHint": False},
    ),
    Tool(
        name="i2c_dump",
        description=(
            "Dump memory from an I2C device by reading all registers sequentially. "
            "Saves to artifacts. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "device_addr": {
                    "type": ["string", "integer"],
                    "description": "7-bit device address (hex string like '0x50' or integer)",
                },
                "size": {"type": "integer", "default": 256, "description": "Number of bytes to dump"},
                "register_bytes": {
                    "type": "integer",
                    "enum": [1, 2],
                    "default": 1,
                    "description": (
                        "Register/memory address width. 1 for most small EEPROMs "
                        "and sensors (up to 256 bytes). 2 for 24C32+ and larger "
                        "EEPROMs (up to 64KB)."
                    ),
                },
            },
            "required": ["session_id", "device_addr"],
        },
    ),
    Tool(
        name="close_i2c",
        description="Close an I2C session and reset the BusPirate to HiZ mode. [allowed-write]",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    # --- 1-Wire tools ---
    Tool(
        name="open_1wire",
        description=(
            "Open a 1-Wire connection and start an engagement. "
            "Configures 1-Wire mode with pullup enabled. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "engagement_name": {"type": "string", "description": "Target device name"},
                "voltage_mv": {"type": "integer", "description": "Target voltage in mV (optional, enables PSU)"},
                "current_ma": {"type": "integer", "description": "Current limit in mA"},
                "project_path": {
                    "type": "string",
                    "description": "Path to a project folder (from project-mcp). If provided, writes to <project_path>/1wire/ instead of creating a standalone engagement.",
                },
            },
            "required": ["engagement_name"],
        },
    ),
    Tool(
        name="onewire_search",
        description=(
            "Search for a device on the 1-Wire bus using Read ROM (0x33). "
            "Returns family code, serial number, and CRC validity. "
            "Works when exactly one device is present. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="onewire_read",
        description=(
            "Send arbitrary bytes on the 1-Wire bus and read back a response. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "write_hex": {"type": "string", "description": "Hex string of bytes to send (e.g. 'CC44')"},
                "read_bytes": {"type": "integer", "description": "Number of bytes to read back"},
            },
            "required": ["session_id", "write_hex", "read_bytes"],
        },
    ),
    Tool(
        name="close_1wire",
        description="Close a 1-Wire session and reset the BusPirate to HiZ mode. [allowed-write]",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    # --- Logic Analyzer (FALA) tools ---
    Tool(
        name="la_prepare",
        description=(
            "Switch to FALA (Follow Along Logic Analyzer) mode and enter a bus "
            "protocol for signal capture. Disconnects BPIO2. Captures all 8 IO pins "
            "at up to 75 MHz every time a bus command runs. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "engagement_name": {
                    "type": "string",
                    "description": "Name for the LA engagement",
                },
                "protocol": {
                    "type": "string",
                    "enum": ["spi", "i2c", "uart", "1wire"],
                    "description": "Bus protocol to use for terminal commands",
                },
                "protocol_config": {
                    "type": "object",
                    "description": "Protocol-specific configuration (speed, mode, etc.)",
                },
                "project_path": {
                    "type": "string",
                    "description": "Path to a project folder (from project-mcp)",
                },
            },
            "required": ["engagement_name", "protocol"],
        },
    ),
    Tool(
        name="la_command",
        description=(
            "Execute a bus command on the terminal and capture FALA data. "
            "Uses BP6 terminal syntax: SPI [0x9f r:3], I2C [0xa0 0x00 r:16], etc. "
            "Returns both decoded terminal output and raw signal capture. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID from la_prepare",
                },
                "command": {
                    "type": "string",
                    "description": "Bus command in BP6 terminal syntax",
                },
            },
            "required": ["session_id", "command"],
        },
    ),
    Tool(
        name="la_analyze",
        description=(
            "Analyze a FALA capture for signal characteristics: transitions, frequency, "
            "duty cycle, idle state, and role identification per channel. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "capture_file": {
                    "type": "string",
                    "description": "Capture file path relative to engagement (default: latest)",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Channel numbers to analyze (default: all 8)",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="la_identify",
        description=(
            "Auto-identify bus protocols from a FALA capture using signal heuristics. "
            "Returns ranked candidates with confidence scores and channel mappings. [read-only]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "capture_file": {
                    "type": "string",
                    "description": "Capture file path (default: latest)",
                },
            },
            "required": ["session_id"],
        },
    ),
    Tool(
        name="la_cleanup",
        description=(
            "Deactivate FALA and restore BPIO2 mode. After this call, BPIO2 tools "
            "(SPI, I2C, UART, 1-Wire) are available again. [allowed-write]"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
]


@app.list_tools()
async def list_tools():
    return TOOL_DEFINITIONS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    tier = classify_tool(name)
    logger.info("tool=%s tier=%s args=%s", name, tier.value, arguments)

    # Approval-write tools require explicit confirmation token.
    # MCP clients that respect destructiveHint will prompt the user,
    # but we enforce server-side too for clients that don't.
    if tier == SafetyTier.APPROVAL_WRITE:
        # Strict: _confirmed must be the literal boolean True. Truthy values
        # like 1, "true", or "yes" do NOT satisfy the gate -- they often leak
        # in from loose JSON decoders or mistakes in client code.
        if arguments.get("_confirmed") is not True:
            desc = f"{name}({', '.join(f'{k}={v}' for k, v in arguments.items())})"
            return [TextContent(
                type="text",
                text=json.dumps({
                    "confirmation_required": True,
                    "tool": name,
                    "arguments": arguments,
                    "message": f"APPROVAL REQUIRED: {desc}. "
                    f"Re-call with _confirmed=true (boolean) to execute.",
                }),
            )]
        # Remove the confirmation token before passing to the tool
        arguments = {k: v for k, v in arguments.items() if k != "_confirmed"}

    try:
        if name == "list_devices":
            result = await tools.tool_list_devices()

        elif name == "verify_connection":
            hw = _get_hardware()
            result = await tools.tool_verify_connection(
                hardware=hw,
                pins={"tx": arguments["tx_pin"], "rx": arguments["rx_pin"]},
                sample_duration_ms=arguments.get("sample_duration_ms", 2000),
            )

        elif name == "scan_baud":
            hw = _get_hardware()
            result = await tools.tool_scan_baud(hardware=hw)

        elif name == "open_uart":
            hw = _get_hardware()
            result = await tools.tool_open_uart(
                session_manager=session_manager,
                hardware=hw,
                baud=arguments["baud"],
                pins={
                    "tx": arguments.get("tx_pin", 4),
                    "rx": arguments.get("rx_pin", 5),
                },
                engagement_name=arguments["engagement_name"],
                device_path=_hardware_port,
                project_path=arguments.get("project_path"),
            )

        elif name == "read_output":
            result = await tools.tool_read_output(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                timeout_ms=arguments.get("timeout_ms", 5000),
            )

        elif name == "send_command":
            result = await tools.tool_send_command(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                command=arguments["command"],
                timeout_ms=arguments.get("timeout_ms", 3000),
            )

        elif name == "close_uart":
            result = await tools.tool_close_uart(
                session_manager=session_manager,
                session_id=arguments["session_id"],
            )

        elif name == "set_voltage":
            hw = _get_hardware()
            result = await tools.tool_set_voltage(
                hardware=hw,
                voltage_v=arguments["voltage_v"],
                current_limit_ma=arguments["current_limit_ma"],
            )

        elif name == "set_power":
            hw = _get_hardware()
            result = await tools.tool_set_power(
                hardware=hw,
                enable=arguments["enable"],
            )

        elif name == "measure_voltage":
            hw = _get_hardware()
            result = await tools.tool_measure_voltage(
                hardware=hw,
                pin=arguments["pin"],
            )

        elif name == "enter_download_mode":
            hw = _get_hardware()
            result = await tools.tool_enter_download_mode(
                hardware=hw,
                boot_pin=arguments["boot_pin"],
                reset_pin=arguments["reset_pin"],
            )

        elif name == "read_flash":
            hw = _get_hardware()
            terminal_port = BusPirateHardware.find_terminal_port()
            if not terminal_port:
                result = {"error": "Cannot find BP6 terminal port for bridge mode"}
            else:
                result = await tools.tool_read_flash(
                    hardware=hw,
                    terminal_port=terminal_port,
                    offset=arguments.get("offset", 0),
                    size=arguments["size"],
                    output_path=arguments["output_path"],
                    boot_pin=arguments.get("boot_pin", -1),
                    reset_pin=arguments.get("reset_pin", -1),
                    baud=arguments.get("baud", 115200),
                )

        # --- SPI ---
        elif name == "open_spi":
            hw = _get_hardware()
            result = await tools.tool_open_spi(
                session_manager=session_manager,
                hardware=hw,
                engagement_name=arguments["engagement_name"],
                speed=arguments.get("speed", 1_000_000),
                clock_polarity=arguments.get("clock_polarity", False),
                clock_phase=arguments.get("clock_phase", False),
                chip_select_idle=arguments.get("chip_select_idle", True),
                voltage_mv=arguments.get("voltage_mv"),
                current_ma=arguments.get("current_ma"),
                project_path=arguments.get("project_path"),
            )

        elif name == "spi_probe":
            result = await tools.tool_spi_probe(
                session_manager=session_manager,
                session_id=arguments["session_id"],
            )

        elif name == "spi_read":
            result = await tools.tool_spi_read(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                address=arguments["address"],
                length=arguments["length"],
            )

        elif name == "spi_dump":
            result = await tools.tool_spi_dump(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                size=arguments.get("size"),
                output_filename=arguments.get("output_filename", "flash_dump.bin"),
                chunk_size=arguments.get("chunk_size", 512),
            )

        elif name == "spi_write":
            result = await tools.tool_spi_write(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                input_path=arguments["input_path"],
                erase=arguments.get("erase", True),
                verify=arguments.get("verify", True),
            )

        elif name == "spi_transfer":
            result = await tools.tool_spi_transfer(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                write_hex=arguments["write_hex"],
                read_bytes=arguments.get("read_bytes", 0),
            )

        elif name == "close_spi":
            result = await tools.tool_close_spi(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                hardware=_get_hardware(),
            )

        # --- I2C ---
        elif name == "open_i2c":
            hw = _get_hardware()
            result = await tools.tool_open_i2c(
                session_manager=session_manager,
                hardware=hw,
                engagement_name=arguments["engagement_name"],
                speed=arguments.get("speed", 400_000),
                clock_stretch=arguments.get("clock_stretch", False),
                voltage_mv=arguments.get("voltage_mv"),
                current_ma=arguments.get("current_ma"),
                project_path=arguments.get("project_path"),
            )

        elif name == "i2c_scan":
            result = await tools.tool_i2c_scan(
                session_manager=session_manager,
                session_id=arguments["session_id"],
            )

        elif name == "i2c_read":
            result = await tools.tool_i2c_read(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                device_addr=arguments["device_addr"],
                register_addr=arguments.get("register_addr"),
                length=arguments.get("length", 1),
            )

        elif name == "i2c_write":
            result = await tools.tool_i2c_write(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                device_addr=arguments["device_addr"],
                register_addr=arguments["register_addr"],
                data_hex=arguments["data_hex"],
            )

        elif name == "i2c_dump":
            result = await tools.tool_i2c_dump(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                device_addr=arguments["device_addr"],
                size=arguments.get("size", 256),
                register_bytes=arguments.get("register_bytes", 1),
            )

        elif name == "close_i2c":
            result = await tools.tool_close_i2c(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                hardware=_get_hardware(),
            )

        # --- 1-Wire ---
        elif name == "open_1wire":
            hw = _get_hardware()
            result = await tools.tool_open_1wire(
                session_manager=session_manager,
                hardware=hw,
                engagement_name=arguments["engagement_name"],
                voltage_mv=arguments.get("voltage_mv"),
                current_ma=arguments.get("current_ma"),
                project_path=arguments.get("project_path"),
            )

        elif name == "onewire_search":
            result = await tools.tool_onewire_search(
                session_manager=session_manager,
                session_id=arguments["session_id"],
            )

        elif name == "onewire_read":
            result = await tools.tool_onewire_read(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                write_hex=arguments["write_hex"],
                read_bytes=arguments["read_bytes"],
            )

        elif name == "close_1wire":
            result = await tools.tool_close_1wire(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                hardware=_get_hardware(),
            )

        # --- Logic Analyzer (FALA) ---
        elif name == "la_prepare":
            # Pass hardware so tool can check active mode and find ports.
            # FALA activation switches binmode, releasing ACM1 from BPIO2.
            hw = _get_hardware()
            result = await tools.tool_la_prepare(
                session_manager=session_manager,
                hardware=hw,
                engagement_name=arguments["engagement_name"],
                protocol=arguments["protocol"],
                protocol_config=arguments.get("protocol_config"),
                project_path=arguments.get("project_path"),
            )
            # Null hardware so next BPIO2 call re-detects after FALA cleanup
            _hardware = None

        elif name == "la_command":
            result = await tools.tool_la_command(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                command=arguments["command"],
            )

        elif name == "la_analyze":
            result = await tools.tool_la_analyze(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                capture_file=arguments.get("capture_file"),
                channels=arguments.get("channels"),
            )

        elif name == "la_identify":
            result = await tools.tool_la_identify(
                session_manager=session_manager,
                session_id=arguments["session_id"],
                capture_file=arguments.get("capture_file"),
            )

        elif name == "la_cleanup":
            result = await tools.tool_la_cleanup(
                session_manager=session_manager,
                session_id=arguments["session_id"],
            )
            # Clear hardware so next BPIO2 call re-detects
            _hardware = None

        else:
            result = {"error": f"Unknown tool: {name}"}

    except Exception as exc:
        logger.error("tool=%s error=%s", name, exc)
        result = {"error": str(exc), "tool": name}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
