"""MCP tool implementations for BusPirate UART operations.

Each function is an async tool handler. Registration with the MCP
server happens in server.py. These functions contain the logic only.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from buspirate_mcp.hardware import BusPirateHardware
from buspirate_mcp.la import FALASession
from buspirate_mcp import la_parsers
from buspirate_mcp.safety import validate_voltage_range
from buspirate_mcp.session import SessionManager

COMMON_BAUD_RATES = [115200, 57600, 38400, 19200, 9600, 4800, 2400, 1200]

ONEWIRE_FAMILIES = {
    0x01: "DS1990A (iButton)",
    0x10: "DS18S20 (Temperature)",
    0x22: "DS1822 (Temperature)",
    0x28: "DS18B20 (Temperature)",
    0x26: "DS2438 (Battery Monitor)",
    0x29: "DS2408 (8-ch Switch)",
    0x3A: "DS2413 (2-ch Switch)",
}
ONEWIRE_CMD_READ_ROM = 0x33

I2C_KNOWN_DEVICES = {
    range(0x50, 0x58): "EEPROM (24Cxx series)",
    range(0x68, 0x6A): "RTC/IMU (DS3231, MPU6050)",
    range(0x76, 0x78): "Pressure/Temp (BME280, BMP280)",
    range(0x3C, 0x3E): "OLED Display (SSD1306)",
    range(0x48, 0x50): "ADC (ADS1115, PCF8591)",
    range(0x20, 0x28): "IO Expander (PCF8574, MCP23017)",
    range(0x38, 0x40): "Touch/Humidity (FT6236, HTU21D)",
}

# SPI flash commands (JEDEC standard)
SPI_CMD_JEDEC_ID = 0x9F
SPI_CMD_READ_STATUS = 0x05
SPI_CMD_WRITE_ENABLE = 0x06
SPI_CMD_READ_DATA = 0x03
SPI_CMD_PAGE_PROGRAM = 0x02
SPI_CMD_CHIP_ERASE = 0xC7
SPI_STATUS_WIP = 0x01

SPI_MANUFACTURERS = {
    0xEF: "Winbond",
    0xC2: "Macronix",
    0x20: "Micron/Numonyx",
    0xC8: "GigaDevice",
    0x9D: "ISSI",
    0x01: "Spansion/Cypress",
    0xBF: "SST/Microchip",
    0x1F: "Atmel/Microchip",
    0x85: "Puya",
}
PROMPT_PATTERN = re.compile(r"^\s*[#$>\]]", re.MULTILINE)
BAUD_SCAN_READ_TIME_S = 0.5
SILENCE_THRESHOLD_S = 0.5


def _ascii_score(data: bytes) -> float:
    """Score data by ratio of printable ASCII to total bytes."""
    if not data:
        return 0.0
    printable = sum(
        1 for b in data if 32 <= b <= 126 or b in (9, 10, 13)
    )
    return printable / len(data)


async def tool_list_devices() -> dict[str, Any]:
    """List BusPirate devices on USB."""
    devices = BusPirateHardware.list_devices()
    result: dict[str, Any] = {"devices": devices}
    if not devices:
        result["hint"] = (
            "No BusPirate found. Check USB cable, run lsusb, "
            "verify permissions on /dev/ttyACM*."
        )
    return result


async def tool_verify_connection(
    hardware: Any,
    pins: dict[str, int],
    sample_duration_ms: int = 2000,
) -> dict[str, Any]:
    """Check for signal activity on the specified pins."""
    tx_pin = pins["tx"]

    # Ensure UART mode is active before reading pins
    hardware.configure_uart(speed=115200)

    samples = []
    transitions = 0
    start = time.monotonic()
    end_time = start + (sample_duration_ms / 1000.0)

    while time.monotonic() < end_time:
        voltages = hardware.get_pin_voltages()
        if voltages is None:
            await asyncio.sleep(0.05)
            continue
        samples.append(voltages[tx_pin])
        if len(samples) >= 2 and samples[-1] != samples[-2]:
            transitions += 1
        await asyncio.sleep(0.05)

    if not samples:
        return {
            "activity_detected": False,
            "transitions": 0,
            "approx_frequency_hz": 0.0,
            "samples_taken": 0,
            "voltage_range_mv": [0, 0],
            "message": (
                f"Could not read pin voltages on pin {tx_pin}. "
                f"UART mode may not be active or hardware not responding."
            ),
        }

    activity = transitions > 0
    duration_s = time.monotonic() - start
    freq_hz = round(transitions / duration_s, 1) if activity else 0.0

    return {
        "activity_detected": activity,
        "transitions": transitions,
        "approx_frequency_hz": freq_hz,
        "samples_taken": len(samples),
        "voltage_range_mv": [min(samples), max(samples)],
        "message": (
            f"Signal transitions detected on pin {tx_pin} "
            f"({transitions} transitions, ~{freq_hz} Hz)."
            if activity
            else f"No activity on pin {tx_pin}. Target may be "
            f"quiescent, powered off, or wiring incorrect."
        ),
    }


async def tool_scan_baud(
    hardware: Any,
) -> dict[str, Any]:
    """Scan common baud rates and score by readable ASCII output."""
    candidates = []

    for rate in COMMON_BAUD_RATES:
        hardware.configure_uart(speed=rate)
        hardware.read()  # flush stale bytes from previous rate
        await asyncio.sleep(BAUD_SCAN_READ_TIME_S)
        data = hardware.read() or b""
        score = _ascii_score(data)
        candidates.append({
            "baud": rate,
            "score": round(score, 3),
            "sample_bytes": len(data),
            "preview": data[:64].decode("utf-8", errors="replace"),
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0] if candidates else {"score": 0.0}

    return {
        "candidates": candidates,
        "recommended": candidates[0]["baud"] if best["score"] >= 0.7 else None,
        "best_score": best["score"],
        "message": (
            f"Recommended baud rate: {candidates[0]['baud']} "
            f"(confidence: {best['score']:.1%})"
            if best["score"] >= 0.7
            else "No baud rate scored above 0.7 threshold. "
            "Provide all scores for user decision."
        ),
    }


async def tool_open_uart(
    session_manager: SessionManager,
    hardware: Any,
    baud: int,
    pins: dict[str, str],
    engagement_name: str,
    device_path: str = "",
    project_path: str | None = None,
) -> dict[str, Any]:
    """Open a persistent UART session and start logging."""
    hardware.configure_uart(speed=baud)
    session = session_manager.create(
        name=engagement_name,
        hardware=hardware,
        baud=baud,
        pins=pins,
        device_path=device_path,
        project_path=project_path,
    )
    return {
        "session_id": session.session_id,
        "engagement_path": str(session.engagement_path),
        "baud": baud,
        "pins": pins,
    }


async def tool_read_output(
    session_manager: SessionManager,
    session_id: str,
    timeout_ms: int = 5000,
) -> dict[str, Any]:
    """Read available data from an open UART session."""
    session = session_manager.get(session_id)
    collected = b""
    end_time = time.monotonic() + (timeout_ms / 1000.0)

    while time.monotonic() < end_time:
        chunk = session.hardware.read() or b""
        if chunk:
            collected += chunk
            session.log_rx(chunk)
            if PROMPT_PATTERN.search(
                collected.decode("utf-8", errors="replace")
            ):
                break
        await asyncio.sleep(0.05)  # always yield to event loop

    text = collected.decode("utf-8", errors="replace")
    return {
        "text": text,
        "bytes_received": len(collected),
    }


async def tool_send_command(
    session_manager: SessionManager,
    session_id: str,
    command: str,
    timeout_ms: int = 3000,
) -> dict[str, Any]:
    """Send a command and capture the response."""
    session = session_manager.get(session_id)
    cmd_bytes = (command + "\r\n").encode("utf-8")
    session.hardware.write(cmd_bytes)
    session.log_tx(cmd_bytes)

    collected = b""
    last_rx_time = time.monotonic()
    end_time = time.monotonic() + (timeout_ms / 1000.0)

    while time.monotonic() < end_time:
        chunk = session.hardware.read() or b""
        if chunk:
            collected += chunk
            session.log_rx(chunk)
            last_rx_time = time.monotonic()

            decoded = collected.decode("utf-8", errors="replace")
            if PROMPT_PATTERN.search(decoded):
                break
        elif collected and (time.monotonic() - last_rx_time) > SILENCE_THRESHOLD_S:
            break
        await asyncio.sleep(0.05)  # always yield to event loop

    response = collected.decode("utf-8", errors="replace")
    return {
        "response": response,
        "bytes_received": len(collected),
    }


async def tool_close_uart(
    session_manager: SessionManager,
    session_id: str,
) -> dict[str, Any]:
    """Close a UART session and finalize logs."""
    session_manager.close(session_id)
    return {"closed": True, "session_id": session_id}


async def tool_set_voltage(
    hardware: Any,
    voltage_v: float,
    current_limit_ma: int,
) -> dict[str, Any]:
    """Set power supply voltage. Validates range before applying."""
    try:
        validate_voltage_range(voltage_v, current_limit_ma)
    except ValueError as e:
        return {"applied": False, "error": str(e)}

    result = hardware.set_voltage(voltage_v, current_limit_ma)
    return {
        "applied": result,
        "voltage_v": voltage_v,
        "current_limit_ma": current_limit_ma,
        "error": None,
    }


async def tool_set_power(
    hardware: Any,
    enable: bool,
) -> dict[str, Any]:
    """Enable or disable power supply."""
    result = hardware.set_power(enable)
    return {
        "applied": result,
        "power": "on" if enable else "off",
    }


async def tool_measure_voltage(
    hardware: Any,
    pin: int,
) -> dict[str, Any]:
    """Read the DC voltage on a BP IO pin via its ADC.

    This is the way to read back an analog level — e.g. confirm the PSU
    output by jumpering VOUT to an IO pin, or check any DC node. Returns the
    requested pin's millivolts plus the full 8-pin snapshot.
    """
    if not isinstance(pin, int) or pin < 0 or pin > 7:
        return {"pin": pin, "millivolts": None, "volts": None,
                "error": f"pin must be 0-7, got {pin}"}
    voltages = hardware.get_pin_voltages()
    if not voltages:
        return {"pin": pin, "millivolts": None, "volts": None,
                "error": "ADC read returned no data (device not responding)."}
    mv = voltages[pin]
    return {
        "pin": pin,
        "millivolts": mv,
        "volts": round(mv / 1000.0, 3),
        "all_pins_mv": list(voltages),
        "error": None,
    }


async def tool_enter_download_mode(
    hardware: Any,
    boot_pin: int,
    reset_pin: int,
) -> dict[str, Any]:
    """Put target into UART download mode via GPIO pin toggling.

    Holds boot_pin LOW, pulses reset_pin LOW then HIGH, then releases
    boot_pin. Works for ESP32, ESP8266, and other chips with UART
    bootloaders triggered by a strapping pin.
    """
    if not isinstance(boot_pin, int) or boot_pin < 0 or boot_pin > 7:
        return {"error": f"boot_pin must be 0-7, got: {boot_pin}"}
    if not isinstance(reset_pin, int) or reset_pin < 0 or reset_pin > 7:
        return {"error": f"reset_pin must be 0-7, got: {reset_pin}"}

    # Hold boot select pin LOW
    hardware.set_pin_output(boot_pin, high=False)
    await asyncio.sleep(0.3)

    # Pulse reset: LOW then HIGH
    hardware.set_pin_output(reset_pin, high=False)
    await asyncio.sleep(0.2)
    hardware.set_pin_output(reset_pin, high=True)
    await asyncio.sleep(0.5)

    # Release boot pin
    hardware.release_pin(boot_pin)
    hardware.release_pin(reset_pin)

    return {
        "status": "download_mode_entered",
        "boot_pin": boot_pin,
        "reset_pin": reset_pin,
        "message": (
            f"Boot pin IO{boot_pin} toggled, reset pin IO{reset_pin} pulsed. "
            f"Target should be in UART download mode."
        ),
    }


def _enter_bridge_mode(terminal_port: str) -> bool:
    """Enter UART bridge mode on the BP6 terminal port.

    The bridge command makes ACM0 a transparent serial passthrough
    to the target. Handles VT100 prompt, detects current mode, and
    enters bridge. Returns True on success.
    """
    import serial
    ser = serial.Serial(terminal_port, 115200, timeout=2)
    try:
        time.sleep(0.5)
        ser.read(ser.in_waiting or 1)

        # Send enter to see what prompt we get
        ser.write(b'\r')
        time.sleep(0.5)
        resp = ser.read(ser.in_waiting or 1).decode('utf-8', errors='replace')

        # Handle VT100 prompt (appears after fresh USB plug)
        if 'VT100' in resp or resp.strip() == '':
            ser.write(b'n\r')
            time.sleep(0.5)
            resp = ser.read(ser.in_waiting or 1).decode('utf-8', errors='replace')

        # Check if we're already in UART mode
        if 'UART>' in resp or 'UART>' in resp.upper():
            # Already in UART mode, send bridge
            ser.write(b'bridge\r')
            time.sleep(1)
            resp = ser.read(ser.in_waiting or 1).decode('utf-8', errors='replace')
            return 'bridge' in resp.lower()

        if 'HiZ>' in resp:
            # Need to enter UART mode first
            ser.write(b'm\r')
            time.sleep(0.5)
            ser.read(ser.in_waiting or 1)
            ser.write(b'3\r')
            time.sleep(1)
            ser.read(ser.in_waiting or 1)
            ser.write(b'y\r')
            time.sleep(0.5)
            ser.read(ser.in_waiting or 1)

        # Try bridge
        ser.write(b'bridge\r')
        time.sleep(1)
        resp = ser.read(ser.in_waiting or 1).decode('utf-8', errors='replace')
        return 'bridge' in resp.lower()
    except Exception:
        return False
    finally:
        ser.close()


def _exit_bridge_mode(terminal_port: str) -> None:
    """Attempt to exit UART bridge mode on the BP6.

    Bridge mode exits when the BP6 button is pressed, or when the
    serial port is closed with a break condition. We send a serial
    break and toggle DTR/RTS to signal the BP6 to drop bridge mode.
    """
    import serial
    try:
        ser = serial.Serial(terminal_port, 115200, timeout=1)
        ser.send_break(duration=0.5)
        ser.dtr = False
        ser.rts = False
        time.sleep(0.3)
        ser.dtr = True
        ser.rts = True
        time.sleep(0.3)
        ser.close()
    except Exception:
        pass  # best effort


async def tool_read_flash(
    hardware: Any,
    terminal_port: str,
    offset: int,
    size: int,
    output_path: str,
    boot_pin: int = -1,
    reset_pin: int = -1,
    baud: int = 115200,
) -> dict[str, Any]:
    """Read flash memory from target via esptool through BP6 UART bridge.

    Optionally enters download mode first if boot_pin and reset_pin are
    provided. Then enters bridge mode on the terminal port and runs
    esptool read-flash.
    """
    # Validate pin numbers when they are being used
    if boot_pin >= 0 and (not isinstance(boot_pin, int) or boot_pin > 7):
        return {"error": f"boot_pin must be 0-7, got: {boot_pin}"}
    if reset_pin >= 0 and (not isinstance(reset_pin, int) or reset_pin > 7):
        return {"error": f"reset_pin must be 0-7, got: {reset_pin}"}

    # Step 1: Enter download mode if pins provided
    if boot_pin >= 0 and reset_pin >= 0:
        dl_result = await tool_enter_download_mode(
            hardware, boot_pin, reset_pin,
        )
        if dl_result["status"] != "download_mode_entered":
            return {"error": "Failed to enter download mode", "detail": dl_result}

    # Step 2: Enter bridge mode.
    # If bridge is already active from a previous read_flash call,
    # _enter_bridge_mode will send 'bridge\r' which the target ignores.
    # esptool will re-sync either way.
    _enter_bridge_mode(terminal_port)

    # Step 3: Run esptool
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Find esptool in the same venv as the MCP server
    esptool_bin = shutil.which('esptool')
    if not esptool_bin:
        # Fall back to looking next to the Python interpreter
        esptool_bin = str(Path(sys.executable).parent / 'esptool')
    cmd = [
        esptool_bin, '--port', terminal_port,
        '--before', 'no-reset', '--after', 'no-reset',
        '--no-stub', '--baud', str(baud),
        'read-flash', str(offset), str(size), str(output),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=max(60, size // 1000),  # ~1s per KB at 115200 + margin
        )
    except FileNotFoundError:
        return {
            "error": "esptool not found. Install with: pip install esptool",
        }
    except subprocess.TimeoutExpired:
        return {
            "error": "esptool timed out",
            "command": ' '.join(cmd),
        }

    if result.returncode != 0:
        return {
            "error": "esptool failed",
            "returncode": result.returncode,
            "stderr": result.stderr[-500:],
            "command": ' '.join(cmd),
        }

    file_size = output.stat().st_size if output.exists() else 0

    # Step 4: Attempt cleanup.
    # Bridge mode on ACM0 cannot be exited programmatically (only via
    # the BP6 physical button). BPIO2 commands hang while bridge is
    # active. The user must USB-replug the BP6 after flash operations
    # to restore normal operation.
    # TODO: find a way to exit bridge mode without physical button

    return {
        "status": "success",
        "output_path": str(output),
        "bytes_read": file_size,
        "offset": offset,
        "size_requested": size,
        "baud": baud,
        "stdout": result.stdout[-300:],
        "cleanup": (
            "Bridge mode is still active. Press the BP6 button or "
            "USB-replug to exit bridge mode and restore normal operation."
        ),
    }


def _onewire_crc8(data: bytes) -> int:
    """CRC-8 Dallas/Maxim 1-Wire (polynomial 0x31, reflected as 0x8C)."""
    crc = 0
    for byte in data:
        for _ in range(8):
            mix = (crc ^ byte) & 0x01
            crc >>= 1
            if mix:
                crc ^= 0x8C
            byte >>= 1
    return crc


async def tool_open_1wire(
    session_manager: SessionManager,
    hardware: Any,
    engagement_name: str,
    voltage_mv: int | None = None,
    current_ma: int | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    """Configure 1-Wire mode and open a transaction session."""
    hardware.configure_1wire(voltage_mv, current_ma)
    session = session_manager.create(
        name=engagement_name,
        hardware=hardware,
        protocol="1wire",
        protocol_config={
            "voltage_mv": voltage_mv,
            "current_ma": current_ma,
        },
        project_path=project_path,
    )
    return {
        "session_id": session.session_id,
        "engagement_path": str(session.engagement_path),
    }


async def tool_onewire_search(
    session_manager: SessionManager,
    session_id: str,
) -> dict[str, Any]:
    """Search for a device on the 1-Wire bus using Read ROM (0x33).

    Works when exactly one device is present. Returns family code, serial,
    CRC validity, and full ROM code. Returns {"present": False} if no device
    asserts a presence pulse.
    """
    session = session_manager.get(session_id)

    present = session.hardware.onewire_reset()
    if not present:
        return {"present": False}

    rom_bytes = session.hardware.onewire_transfer([ONEWIRE_CMD_READ_ROM], 8)
    if not rom_bytes or len(rom_bytes) < 8:
        return {"present": False}

    rom = bytes(rom_bytes)
    family_code = rom[0]
    serial_bytes = rom[1:7]
    crc_byte = rom[7]

    # CRC covers the first 7 bytes; the 8th byte is the CRC itself.
    crc_calc = _onewire_crc8(rom[:7])
    crc_valid = crc_calc == crc_byte

    family_name = ONEWIRE_FAMILIES.get(family_code, f"Unknown (0x{family_code:02X})")
    serial_hex = serial_bytes.hex().upper()
    rom_code = rom.hex().upper()

    session.log_transaction(
        operation="READ_ROM",
        write_hex=f"{ONEWIRE_CMD_READ_ROM:02X}",
        read_hex=rom.hex().upper(),
        metadata={
            "family_code": f"0x{family_code:02X}",
            "family_name": family_name,
            "serial": serial_hex,
            "crc_valid": crc_valid,
        },
    )

    return {
        "present": True,
        "family_code": f"0x{family_code:02X}",
        "family_name": family_name,
        "serial": serial_hex,
        "crc_valid": crc_valid,
        "rom_code": rom_code,
    }


async def tool_onewire_read(
    session_manager: SessionManager,
    session_id: str,
    write_hex: str,
    read_bytes: int,
) -> dict[str, Any]:
    """Send arbitrary bytes on the 1-Wire bus and read back a response.

    write_hex: hex string of bytes to write (e.g. "CC44").
    read_bytes: number of bytes to read after the write.
    """
    session = session_manager.get(session_id)

    data = bytes.fromhex(write_hex)
    result = session.hardware.onewire_transfer(list(data), read_bytes)
    result_bytes = bytes(result) if result else b""
    result_hex = result_bytes.hex().upper()

    session.log_transaction(
        operation="TRANSFER",
        write_hex=write_hex.upper(),
        read_hex=result_hex,
    )

    return {
        "tx": write_hex.upper(),
        "rx": result_hex,
        "bytes_read": len(result_bytes),
    }


async def tool_close_1wire(
    session_manager: SessionManager,
    session_id: str,
    hardware: Any,
) -> dict[str, Any]:
    """Close a 1-Wire session and reset the BusPirate to HiZ mode."""
    session_manager.close(session_id)
    hardware.reset_mode()
    return {"closed": True}


# ---------------------------------------------------------------------------
# I2C tools
# ---------------------------------------------------------------------------

def _i2c_hint(addr_7bit: int) -> str | None:
    """Look up a hint for a 7-bit I2C address from I2C_KNOWN_DEVICES."""
    for addr_range, hint in I2C_KNOWN_DEVICES.items():
        if addr_7bit in addr_range:
            return hint
    return None


def _parse_device_addr(device_addr: str | int) -> int:
    """Parse device_addr as hex string or int, return 7-bit int."""
    if isinstance(device_addr, str):
        return int(device_addr, 16)
    return device_addr


async def tool_open_i2c(
    session_manager: SessionManager,
    hardware: Any,
    engagement_name: str,
    speed: int = 400000,
    clock_stretch: bool = False,
    voltage_mv: int | None = None,
    current_ma: int | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    """Open a persistent I2C session and start logging."""
    hardware.configure_i2c(speed, clock_stretch, voltage_mv, current_ma)
    session = session_manager.create(
        name=engagement_name,
        hardware=hardware,
        protocol="i2c",
        protocol_config={
            "speed": speed,
            "clock_stretch": clock_stretch,
            "voltage_mv": voltage_mv,
            "current_ma": current_ma,
        },
        project_path=project_path,
    )
    return {
        "session_id": session.session_id,
        "engagement_path": str(session.engagement_path),
        "i2c_config": {
            "speed": speed,
            "clock_stretch": clock_stretch,
            "voltage_mv": voltage_mv,
            "current_ma": current_ma,
        },
    }


async def tool_i2c_scan(
    session_manager: SessionManager,
    session_id: str,
) -> dict[str, Any]:
    """Scan the I2C bus for devices and return addresses with hints."""
    session = session_manager.get(session_id)
    raw_addresses = session.hardware.i2c_scan(0x00, 0x7F)

    # SDK returns raw (shifted) addresses -- convert to 7-bit and deduplicate
    seen: set[int] = set()
    devices = []
    for raw_addr in raw_addresses:
        addr_7bit = raw_addr >> 1
        if addr_7bit in seen:
            continue
        seen.add(addr_7bit)
        hint = _i2c_hint(addr_7bit)
        devices.append({
            "address": f"0x{addr_7bit:02X}",
            "address_7bit": addr_7bit,
            "hint": hint,
        })

    devices.sort(key=lambda d: d["address_7bit"])

    session.log_transaction(
        operation="scan",
        metadata={"found": len(devices), "addresses": [d["address"] for d in devices]},
    )

    return {"devices": devices, "count": len(devices)}


async def tool_i2c_read(
    session_manager: SessionManager,
    session_id: str,
    device_addr: str | int,
    register_addr: int | None = None,
    length: int = 1,
) -> dict[str, Any]:
    """Read bytes from an I2C device, optionally from a specific register."""
    session = session_manager.get(session_id)
    addr_7bit = _parse_device_addr(device_addr)
    addr_write = (addr_7bit << 1) & 0xFE

    if register_addr is not None:
        write_data = [addr_write, register_addr]
    else:
        write_data = [addr_write]

    read_bytes_raw = session.hardware.i2c_transfer(write_data, read_bytes=length)

    data_hex = read_bytes_raw.hex() if read_bytes_raw else ""
    data_list = list(read_bytes_raw) if read_bytes_raw else []

    write_hex = bytes(write_data).hex()
    session.log_transaction(
        operation="read",
        write_hex=write_hex,
        read_hex=data_hex,
        metadata={
            "address": f"0x{addr_7bit:02X}",
            "register": f"0x{register_addr:02X}" if register_addr is not None else None,
            "length": length,
        },
    )

    return {
        "address": f"0x{addr_7bit:02X}",
        "register": f"0x{register_addr:02X}" if register_addr is not None else None,
        "data_hex": data_hex,
        "data_bytes": data_list,
        "length": len(data_list),
    }


async def tool_i2c_write(
    session_manager: SessionManager,
    session_id: str,
    device_addr: str | int,
    register_addr: int,
    data_hex: str,
) -> dict[str, Any]:
    """Write bytes to an I2C device register."""
    session = session_manager.get(session_id)
    addr_7bit = _parse_device_addr(device_addr)
    addr_write = (addr_7bit << 1) & 0xFE

    data_bytes = list(bytes.fromhex(data_hex))
    write_data = [addr_write, register_addr] + data_bytes

    session.hardware.i2c_transfer(write_data, read_bytes=0)

    write_hex = bytes(write_data).hex()
    session.log_transaction(
        operation="write",
        write_hex=write_hex,
        metadata={
            "address": f"0x{addr_7bit:02X}",
            "register": f"0x{register_addr:02X}",
            "bytes_written": len(data_bytes),
        },
    )

    return {
        "written": True,
        "address": f"0x{addr_7bit:02X}",
        "register": f"0x{register_addr:02X}",
        "bytes_written": len(data_bytes),
    }


async def tool_i2c_dump(
    session_manager: SessionManager,
    session_id: str,
    device_addr: str | int,
    size: int = 256,
    register_bytes: int = 1,
) -> dict[str, Any]:
    """Dump memory from an I2C device by reading all registers sequentially.

    register_bytes=1 for devices with 8-bit register/memory addresses (most small
    EEPROMs, sensors). register_bytes=2 for 16-bit addresses (24C32+, larger
    EEPROMs) -- address sent MSB first.
    """
    if register_bytes not in (1, 2):
        return {"error": f"register_bytes must be 1 or 2, got {register_bytes}"}
    max_addr = 256 if register_bytes == 1 else 65536
    if size > max_addr:
        return {
            "error": f"size {size} exceeds {max_addr}-byte address space "
                     f"for register_bytes={register_bytes}"
        }

    session = session_manager.get(session_id)
    addr_7bit = _parse_device_addr(device_addr)
    addr_write = (addr_7bit << 1) & 0xFE

    chunk_size = 32
    all_data = bytearray()

    for offset in range(0, size, chunk_size):
        count = min(chunk_size, size - offset)
        if register_bytes == 1:
            write_data = [addr_write, offset & 0xFF]
        else:
            write_data = [addr_write, (offset >> 8) & 0xFF, offset & 0xFF]
        chunk = session.hardware.i2c_transfer(write_data, read_bytes=count)
        if chunk:
            all_data.extend(chunk)
        else:
            all_data.extend(b"\xff" * count)

    artifacts_dir = session.engagement_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump_file = artifacts_dir / f"i2c_dump_{addr_7bit:02x}.bin"
    dump_file.write_bytes(bytes(all_data))

    hex_preview = all_data[:64].hex()

    session.log_transaction(
        operation="dump",
        metadata={
            "address": f"0x{addr_7bit:02X}",
            "size": size,
            "register_bytes": register_bytes,
            "bytes_read": len(all_data),
            "file": str(dump_file),
        },
    )

    return {
        "bytes_read": len(all_data),
        "hex_preview": hex_preview,
        "file_path": str(dump_file),
    }


async def tool_close_i2c(
    session_manager: SessionManager,
    session_id: str,
    hardware: Any,
) -> dict[str, Any]:
    """Close an I2C session and reset the bus mode."""
    session_manager.close(session_id)
    hardware.reset_mode()
    return {"closed": True}


# ---------------------------------------------------------------------------
# SPI tools
# ---------------------------------------------------------------------------


def _human_size(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.0f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


# Cap JEDEC-derived sizes to 64 MB. Capacity code > 26 (2^26) is almost
# always a bogus read on a disconnected or misbehaving chip, not a real
# 128 MB+ device. Without this cap, a bad byte causes a multi-GB allocation.
SPI_MAX_CAPACITY_BYTES = 64 * 1024 * 1024


def _jedec_capacity_bytes(capacity_code: int) -> int:
    """Convert JEDEC capacity byte (jedec_bytes[2]) to bytes, capped at 64MB."""
    if capacity_code <= 0:
        return 0
    size = 1 << capacity_code
    if size > SPI_MAX_CAPACITY_BYTES:
        return SPI_MAX_CAPACITY_BYTES
    return size


async def tool_open_spi(
    session_manager: SessionManager,
    hardware: Any,
    engagement_name: str,
    speed: int = 1_000_000,
    clock_polarity: bool = False,
    clock_phase: bool = False,
    chip_select_idle: bool = True,
    voltage_mv: int | None = None,
    current_ma: int | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    """Configure SPI mode on the BusPirate and create an engagement session."""
    hardware.configure_spi(
        speed=speed,
        clock_polarity=clock_polarity,
        clock_phase=clock_phase,
        chip_select_idle=chip_select_idle,
        voltage_mv=voltage_mv,
        current_ma=current_ma,
    )
    session = session_manager.create(
        name=engagement_name,
        hardware=hardware,
        protocol="spi",
        protocol_config={
            "speed": speed,
            "cpol": clock_polarity,
            "cpha": clock_phase,
        },
        project_path=project_path,
    )
    return {
        "session_id": session.session_id,
        "engagement_path": str(session.engagement_path),
        "spi_config": {
            "speed": speed,
            "cpol": clock_polarity,
            "cpha": clock_phase,
            "cs_idle": chip_select_idle,
        },
    }


async def tool_spi_probe(
    session_manager: SessionManager,
    session_id: str,
) -> dict[str, Any]:
    """Read JEDEC ID and status register from the SPI flash chip."""
    session = session_manager.get(session_id)
    hw = session.hardware

    # JEDEC ID: send 0x9F, read 3 bytes (manufacturer, type, capacity)
    jedec = hw.spi_transfer([SPI_CMD_JEDEC_ID], 3)
    jedec_bytes = bytes(jedec)
    session.log_transaction(
        "jedec_id",
        write_hex=f"{SPI_CMD_JEDEC_ID:02x}",
        read_hex=jedec_bytes.hex(),
    )

    # Status register: send 0x05, read 1 byte
    status = hw.spi_transfer([SPI_CMD_READ_STATUS], 1)
    status_byte = bytes(status)[0]
    session.log_transaction(
        "read_status",
        write_hex=f"{SPI_CMD_READ_STATUS:02x}",
        read_hex=f"{status_byte:02x}",
    )

    manufacturer = SPI_MANUFACTURERS.get(
        jedec_bytes[0], f"Unknown (0x{jedec_bytes[0]:02X})"
    )
    capacity_bytes = _jedec_capacity_bytes(jedec_bytes[2])

    return {
        "jedec_id": jedec_bytes.hex().upper(),
        "manufacturer": manufacturer,
        "device_type": f"0x{jedec_bytes[1]:02X}",
        "capacity_bytes": capacity_bytes,
        "capacity_human": _human_size(capacity_bytes),
        "status_register": f"0x{status_byte:02X}",
        "write_protected": bool(status_byte & 0x0C),
    }


async def tool_spi_read(
    session_manager: SessionManager,
    session_id: str,
    address: int,
    length: int,
) -> dict[str, Any]:
    """Read a region of SPI flash memory."""
    session = session_manager.get(session_id)
    hw = session.hardware
    chunk_size = 512
    collected = bytearray()

    remaining = length
    addr = address
    while remaining > 0:
        n = min(chunk_size, remaining)
        cmd = [
            SPI_CMD_READ_DATA,
            (addr >> 16) & 0xFF,
            (addr >> 8) & 0xFF,
            addr & 0xFF,
        ]
        data = hw.spi_transfer(cmd, n)
        collected.extend(data)
        addr += n
        remaining -= n

    # Save to artifacts
    artifacts_dir = session.engagement_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / f"spi_read_{address:06x}_{length}.bin"
    out_path.write_bytes(bytes(collected))

    session.log_transaction(
        "read_data",
        write_hex=f"03 {address:06x}",
        read_hex=f"{len(collected)} bytes",
        metadata={"address": address, "length": length},
    )

    hex_preview = bytes(collected[:64]).hex()
    return {
        "bytes_read": len(collected),
        "hex_preview": hex_preview,
        "file_path": str(out_path),
    }


async def tool_spi_dump(
    session_manager: SessionManager,
    session_id: str,
    size: int | None = None,
    output_filename: str = "flash_dump.bin",
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Dump the entire SPI flash to a file."""
    session = session_manager.get(session_id)
    hw = session.hardware

    # Auto-detect size via JEDEC ID if not provided
    if size is None:
        jedec = hw.spi_transfer([SPI_CMD_JEDEC_ID], 3)
        jedec_bytes = bytes(jedec)
        size = _jedec_capacity_bytes(jedec_bytes[2])
        if size == 0:
            return {
                "error": "Could not auto-detect flash size. "
                "Provide size explicitly.",
            }

    artifacts_dir = session.engagement_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(output_filename).name
    out_path = artifacts_dir / safe_name

    collected = bytearray()
    addr = 0
    t0 = time.monotonic()

    while addr < size:
        n = min(chunk_size, size - addr)
        cmd = [
            SPI_CMD_READ_DATA,
            (addr >> 16) & 0xFF,
            (addr >> 8) & 0xFF,
            addr & 0xFF,
        ]
        data = hw.spi_transfer(cmd, n)
        collected.extend(data)
        addr += n

    elapsed = time.monotonic() - t0
    out_path.write_bytes(bytes(collected))

    speed_kbps = (len(collected) / 1024) / elapsed if elapsed > 0 else 0.0

    session.log_transaction(
        "dump",
        write_hex="full flash",
        read_hex=f"{len(collected)} bytes",
        metadata={
            "size": size,
            "elapsed_s": round(elapsed, 3),
            "speed_kbps": round(speed_kbps, 1),
        },
    )

    return {
        "bytes_read": len(collected),
        "elapsed_s": round(elapsed, 3),
        "speed_kbps": round(speed_kbps, 1),
        "file_path": str(out_path),
    }


async def tool_spi_write(
    session_manager: SessionManager,
    session_id: str,
    input_path: str,
    erase: bool = True,
    verify: bool = True,
) -> dict[str, Any]:
    """Write a binary file to SPI flash. Optionally erases first and verifies."""
    session = session_manager.get(session_id)
    hw = session.hardware

    data = Path(input_path).read_bytes()
    t0 = time.monotonic()

    # Chip erase
    if erase:
        hw.spi_transfer([SPI_CMD_WRITE_ENABLE], 0)
        hw.spi_transfer([SPI_CMD_CHIP_ERASE], 0)
        # Poll WIP bit until erase completes (120s deadline for damaged chips)
        erase_deadline = time.monotonic() + 120
        while True:
            if time.monotonic() > erase_deadline:
                return {"error": "Chip erase timeout: WIP bit stuck after 120s"}
            status = hw.spi_transfer([SPI_CMD_READ_STATUS], 1)
            if not (bytes(status)[0] & SPI_STATUS_WIP):
                break
            await asyncio.sleep(0.1)
        session.log_transaction("chip_erase", write_hex="06 c7", read_hex="done")

    # Page program (256-byte pages)
    page_size = 256
    addr = 0
    while addr < len(data):
        page = data[addr : addr + page_size]
        hw.spi_transfer([SPI_CMD_WRITE_ENABLE], 0)
        cmd = [
            SPI_CMD_PAGE_PROGRAM,
            (addr >> 16) & 0xFF,
            (addr >> 8) & 0xFF,
            addr & 0xFF,
        ] + list(page)
        hw.spi_transfer(cmd, 0)
        # Poll WIP (5s deadline per page)
        page_deadline = time.monotonic() + 5
        while True:
            if time.monotonic() > page_deadline:
                return {
                    "error": f"Page program timeout at addr 0x{addr:06x}: WIP bit stuck after 5s"
                }
            status = hw.spi_transfer([SPI_CMD_READ_STATUS], 1)
            if not (bytes(status)[0] & SPI_STATUS_WIP):
                break
            await asyncio.sleep(0.01)
        addr += page_size

    session.log_transaction(
        "page_program",
        write_hex=f"{len(data)} bytes",
        read_hex="",
        metadata={"pages": (len(data) + page_size - 1) // page_size},
    )

    # Verify
    verified = False
    if verify:
        read_back = bytearray()
        addr = 0
        while addr < len(data):
            n = min(512, len(data) - addr)
            cmd = [
                SPI_CMD_READ_DATA,
                (addr >> 16) & 0xFF,
                (addr >> 8) & 0xFF,
                addr & 0xFF,
            ]
            chunk = hw.spi_transfer(cmd, n)
            read_back.extend(chunk)
            addr += n
        verified = bytes(read_back[: len(data)]) == data
        session.log_transaction(
            "verify",
            write_hex="read-back compare",
            read_hex="match" if verified else "MISMATCH",
        )

    elapsed = time.monotonic() - t0
    return {
        "bytes_written": len(data),
        "erased": erase,
        "verified": verified if verify else None,
        "elapsed_s": round(elapsed, 3),
    }


async def tool_spi_transfer(
    session_manager: SessionManager,
    session_id: str,
    write_hex: str,
    read_bytes: int = 0,
) -> dict[str, Any]:
    """Send a raw SPI transaction."""
    session = session_manager.get(session_id)
    hw = session.hardware

    tx_data = list(bytes.fromhex(write_hex))
    rx_data = hw.spi_transfer(tx_data, read_bytes)
    rx_hex = bytes(rx_data).hex() if rx_data else ""

    session.log_transaction(
        "raw_transfer",
        write_hex=write_hex,
        read_hex=rx_hex,
    )

    return {
        "tx": write_hex,
        "rx": rx_hex,
        "bytes_read": len(rx_data) if rx_data else 0,
    }


async def tool_close_spi(
    session_manager: SessionManager,
    session_id: str,
    hardware: Any,
) -> dict[str, Any]:
    """Close an SPI session and reset the BusPirate mode."""
    session_manager.close(session_id)
    hardware.reset_mode()
    return {"closed": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# FALA (Follow Along Logic Analyzer) tools
# ---------------------------------------------------------------------------


async def tool_la_prepare(
    session_manager: SessionManager,
    hardware: BusPirateHardware,
    engagement_name: str,
    protocol: str,
    protocol_config: dict | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    """Switch to FALA mode and enter a bus protocol for capture."""
    try:
        # Check no BPIO2 mode is active
        if hardware._active_mode is not None:
            return {"error": f"Close the active {hardware._active_mode} session before starting LA"}

        terminal_port = hardware.find_terminal_port()
        if terminal_port is None:
            return {"error": "Terminal port not found"}

        # Find the binary/FALA port (the other port)
        devices = hardware.list_devices()
        fala_port = None
        for d in devices:
            if d["path"] != terminal_port:
                fala_port = d["path"]
                break
        if fala_port is None:
            return {"error": "FALA port not found"}

        fala = FALASession(terminal_port, fala_port)
        result = fala.activate(protocol, protocol_config)

        session = session_manager.create(
            name=engagement_name,
            hardware=fala,
            protocol="la",
            protocol_config={"bus_protocol": protocol, **result},
            project_path=project_path,
        )

        return {
            "session_id": session.session_id,
            "engagement_path": str(session.engagement_path),
            "protocol": protocol,
            "sample_rate_hz": result.get("sample_rate_hz", 0),
        }
    except Exception as exc:
        return {"error": str(exc)}


_LA_COMMAND_RE = re.compile(r"^[\x20-\x7E]+$")


def _validate_la_command(command: str) -> str | None:
    """Validate a BP6 bus command before sending to the terminal.

    BP6 bus commands use square brackets for transactions (e.g. `[0x9F r:3]`).
    Return an error string if invalid, None if OK.
    """
    if not command or not command.strip():
        return "la_command: empty command"
    if len(command) > 512:
        return f"la_command: command too long ({len(command)} > 512)"
    if not _LA_COMMAND_RE.fullmatch(command):
        return "la_command: command contains control or non-ASCII characters"
    opens = command.count("[")
    closes = command.count("]")
    if opens != closes:
        return f"la_command: unbalanced brackets ({opens} '[' vs {closes} ']')"
    if opens == 0:
        return "la_command: missing bus transaction brackets '[...]'"
    return None


async def tool_la_command(
    session_manager: SessionManager,
    session_id: str,
    command: str,
) -> dict[str, Any]:
    """Execute a bus command and capture FALA data."""
    err = _validate_la_command(command)
    if err is not None:
        return {"error": err}
    try:
        session = session_manager.get(session_id)
        fala = session.hardware  # FALASession stored here

        result = fala.execute(command)

        # Save raw capture to artifacts if we got samples
        capture_info = result.get("capture", {})
        raw = capture_info.get("raw", b'')
        raw_file = None
        if raw:
            artifacts_dir = session.engagement_path / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            # Find next capture number
            existing = list(artifacts_dir.glob("capture_*.bin"))
            num = len(existing) + 1
            capture_path = artifacts_dir / f"capture_{num:03d}.bin"
            capture_path.write_bytes(raw)
            raw_file = str(capture_path.relative_to(session.engagement_path))

        # Log the transaction
        notification = capture_info.get("notification", {})
        session.log_transaction(
            operation="la_command",
            write_hex=command,
            read_hex="",
            metadata={
                "terminal_output": result.get("terminal_output", ""),
                "samples": notification.get("samples", 0) if notification else 0,
                "sample_rate_hz": notification.get("sample_rate_hz", 0) if notification else 0,
                "raw_file": raw_file,
            },
        )

        return {
            "terminal_output": result.get("terminal_output", ""),
            "capture": {
                "samples": notification.get("samples", 0) if notification else 0,
                "sample_rate_hz": notification.get("sample_rate_hz", 0) if notification else 0,
                "duration_us": round(notification["samples"] / notification["sample_rate_hz"] * 1_000_000, 2) if notification and notification.get("sample_rate_hz", 0) > 0 else 0,
                "raw_file": raw_file,
                "raw_bytes": len(raw),
            },
        }
    except KeyError:
        return {"error": f"Session not found: {session_id}"}
    except Exception as exc:
        return {"error": str(exc)}


async def tool_la_analyze(
    session_manager: SessionManager,
    session_id: str,
    capture_file: str | None = None,
    channels: list[int] | None = None,
) -> dict[str, Any]:
    """Analyze a FALA capture for signal characteristics."""
    try:
        session = session_manager.get(session_id)
        artifacts_dir = session.engagement_path / "artifacts"

        # Find capture file
        if capture_file:
            path = Path(capture_file).resolve() if Path(capture_file).is_absolute() else (session.engagement_path / capture_file).resolve()
            if not path.is_relative_to(session.engagement_path.resolve()):
                return {"error": f"Path must be under engagement directory: {capture_file}"}
        else:
            # Use latest capture
            captures = sorted(artifacts_dir.glob("capture_*.bin"))
            if not captures:
                return {"error": "No capture files found"}
            path = captures[-1]

        raw = path.read_bytes()
        if not raw:
            return {"error": "Capture file is empty"}

        # Get sample rate from session config or default
        config = session.protocol if hasattr(session, 'protocol') else None
        sample_rate = 75_000_000  # default

        result = la_parsers.analyze_channels(raw, sample_rate, channels)
        result["capture_file"] = str(path.relative_to(session.engagement_path))
        return result
    except KeyError:
        return {"error": f"Session not found: {session_id}"}
    except Exception as exc:
        return {"error": str(exc)}


async def tool_la_identify(
    session_manager: SessionManager,
    session_id: str,
    capture_file: str | None = None,
) -> dict[str, Any]:
    """Auto-identify protocols from a FALA capture."""
    try:
        session = session_manager.get(session_id)
        artifacts_dir = session.engagement_path / "artifacts"

        if capture_file:
            path = Path(capture_file).resolve() if Path(capture_file).is_absolute() else (session.engagement_path / capture_file).resolve()
            if not path.is_relative_to(session.engagement_path.resolve()):
                return {"error": f"Path must be under engagement directory: {capture_file}"}
        else:
            captures = sorted(artifacts_dir.glob("capture_*.bin"))
            if not captures:
                return {"error": "No capture files found"}
            path = captures[-1]

        raw = path.read_bytes()
        if not raw:
            return {"error": "Capture file is empty"}

        sample_rate = 75_000_000
        analysis = la_parsers.analyze_channels(raw, sample_rate)
        candidates = la_parsers.identify_protocol(analysis)

        return {
            "candidates": candidates,
            "count": len(candidates),
            "capture_file": str(path.relative_to(session.engagement_path)),
        }
    except KeyError:
        return {"error": f"Session not found: {session_id}"}
    except Exception as exc:
        return {"error": str(exc)}


async def tool_la_cleanup(
    session_manager: SessionManager,
    session_id: str,
) -> dict[str, Any]:
    """Deactivate FALA and restore BPIO2 mode."""
    try:
        session = session_manager.get(session_id)
        fala = session.hardware

        fala.deactivate()
        session_manager.close(session_id)

        return {"restored": True, "session_id": session_id}
    except KeyError:
        return {"error": f"Session not found: {session_id}"}
    except Exception as exc:
        return {"error": str(exc)}
