"""Three-tier safety model for BusPirate MCP tools.

Tiers:
  read-only       -- full autonomy, no side effects
  allowed-write   -- autonomous, all calls logged
  approval-write  -- blocks until human confirms
"""

from __future__ import annotations

from enum import Enum


class SafetyTier(Enum):
    READ_ONLY = "read-only"
    ALLOWED_WRITE = "allowed-write"
    APPROVAL_WRITE = "approval-write"


_TOOL_TIERS: dict[str, SafetyTier] = {
    "list_devices": SafetyTier.READ_ONLY,
    "verify_connection": SafetyTier.READ_ONLY,
    "scan_baud": SafetyTier.READ_ONLY,
    "read_output": SafetyTier.READ_ONLY,
    "open_uart": SafetyTier.ALLOWED_WRITE,
    "send_command": SafetyTier.ALLOWED_WRITE,
    "close_uart": SafetyTier.ALLOWED_WRITE,
    "set_voltage": SafetyTier.APPROVAL_WRITE,
    "set_power": SafetyTier.APPROVAL_WRITE,
    "measure_voltage": SafetyTier.READ_ONLY,
    "enter_download_mode": SafetyTier.ALLOWED_WRITE,
    "read_flash": SafetyTier.ALLOWED_WRITE,
    # SPI
    "open_spi": SafetyTier.ALLOWED_WRITE,
    "spi_probe": SafetyTier.READ_ONLY,
    "spi_read": SafetyTier.READ_ONLY,
    "spi_dump": SafetyTier.READ_ONLY,
    "spi_write": SafetyTier.APPROVAL_WRITE,
    "spi_transfer": SafetyTier.READ_ONLY,
    "close_spi": SafetyTier.ALLOWED_WRITE,
    # I2C
    "open_i2c": SafetyTier.ALLOWED_WRITE,
    "i2c_scan": SafetyTier.READ_ONLY,
    "i2c_read": SafetyTier.READ_ONLY,
    "i2c_write": SafetyTier.APPROVAL_WRITE,
    "i2c_dump": SafetyTier.READ_ONLY,
    "close_i2c": SafetyTier.ALLOWED_WRITE,
    # 1-Wire
    "open_1wire": SafetyTier.ALLOWED_WRITE,
    "onewire_search": SafetyTier.READ_ONLY,
    "onewire_read": SafetyTier.READ_ONLY,
    "close_1wire": SafetyTier.ALLOWED_WRITE,
    # Logic Analyzer (FALA)
    "la_prepare": SafetyTier.ALLOWED_WRITE,
    "la_command": SafetyTier.READ_ONLY,
    "la_analyze": SafetyTier.READ_ONLY,
    "la_identify": SafetyTier.READ_ONLY,
    "la_cleanup": SafetyTier.ALLOWED_WRITE,
}


def classify_tool(tool_name: str) -> SafetyTier:
    """Return the safety tier for a tool name."""
    tier = _TOOL_TIERS.get(tool_name)
    if tier is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return tier


def validate_voltage_range(voltage_v: float, current_limit_ma: int) -> None:
    """Raise ValueError if voltage/current params are out of safe range.

    BusPirate 6 supports 0.8-5.0V and 0-500mA.
    """
    if voltage_v < 0.8 or voltage_v > 5.0:
        raise ValueError(
            f"Voltage {voltage_v}V out of range. "
            f"BusPirate 6 supports 0.8V to 5.0V."
        )
    if current_limit_ma < 0:
        raise ValueError(
            f"Current limit must be >= 0, got {current_limit_ma}mA."
        )
    if current_limit_ma > 500:
        raise ValueError(
            f"Current limit {current_limit_ma}mA exceeds max 500mA."
        )
