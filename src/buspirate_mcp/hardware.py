"""Thin abstraction over the BPIO2 SDK.

Only this module imports from the BPIO2 SDK (bpio_client, bpio_uart, etc.).
Everything else in the codebase talks to this module. This makes
testing without hardware trivial -- mock this, not the SDK.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from typing import Any

from serial.tools import list_ports

# Add vendored BPIO2 SDK to sys.path so we can import it.
# The SDK lives at vendor/bpio2/python/pybpio/ relative to the package root.
_VENDOR_PATH = str(
    Path(__file__).resolve().parents[2] / "vendor" / "bpio2" / "python"
)
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

try:
    from pybpio.bpio_client import BPIOClient
    from pybpio.bpio_uart import BPIOUART
    from pybpio.bpio_spi import BPIOSPI
    from pybpio.bpio_i2c import BPIOI2C
    from pybpio.bpio_1wire import BPIO1Wire
except ImportError:
    # SDK not available -- tests use mocks, but fail fast at runtime
    BPIOClient = None  # type: ignore[assignment, misc]
    BPIOUART = None  # type: ignore[assignment, misc]
    BPIOSPI = None  # type: ignore[assignment, misc]
    BPIOI2C = None  # type: ignore[assignment, misc]
    BPIO1Wire = None  # type: ignore[assignment, misc]


class BusPirateHardware:
    """Wraps a single BusPirate 6 connection."""

    def __init__(self, client: Any, uart: Any) -> None:
        self.client = client
        self.uart = uart
        self.spi: Any = None
        self.i2c: Any = None
        self.onewire: Any = None
        self._active_mode: str | None = None
        self._last_voltage_mv: int = 0
        self._last_current_ma: int = 0

    @property
    def _active_protocol(self) -> Any:
        """Return the protocol object for the currently active mode.

        Falls back to self.uart when no mode is set (backward compat for
        PSU calls before any protocol is configured).
        """
        if self._active_mode == "spi":
            return self.spi
        if self._active_mode == "i2c":
            return self.i2c
        if self._active_mode == "1wire":
            return self.onewire
        # "uart" or None -- use uart as default
        return self.uart

    def _check_mode(self, target_mode: str) -> None:
        """Raise if a different mode is already active.

        Reconfiguring the same mode is allowed. Switching requires
        reset_mode() first.
        """
        if self._active_mode is not None and self._active_mode != target_mode:
            raise RuntimeError(
                f"Mode '{self._active_mode}' is active. "
                f"Call reset_mode() before switching to '{target_mode}'."
            )

    def reset_mode(self) -> None:
        """Clear the active mode so a different protocol can be configured."""
        self._active_mode = None

    # BP6 USB identifiers (pid.codes 1209:7331).
    _BP_VID = 0x1209
    _BP_PID = 0x7331

    @staticmethod
    def _usb_interface(port: Any) -> int:
        """Best-effort USB interface number, used to order the BP6's two ports.

        The terminal is the lower-numbered interface, the binary BPIO2 protocol
        the higher one (the BPIO2 port enumerates as interface 2). pyserial
        reports it as the suffix of .location (e.g. "1-8:x.2" -> 2). Defaults to
        0 when absent, in which case callers fall back to device-name ordering.
        """
        m = re.search(r"[.:](\d+)$", port.location or "")
        return int(m.group(1)) if m else 0

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        """Find BusPirate devices on USB CDC serial ports.

        The BP6 enumerates as a composite USB device with two CDC ports: the
        text terminal (lower USB interface) and the binary BPIO2 protocol
        (higher interface). We enumerate all serial ports via pyserial, keep the
        ones whose USB VID:PID match the BusPirate, and order them by interface
        number so the terminal comes first. This is cross-platform (Linux,
        Windows, macOS) and ignores unrelated CDC devices.
        """
        matched: list[tuple[int, str]] = []
        for p in list_ports.comports():
            if (p.vid, p.pid) != (BusPirateHardware._BP_VID, BusPirateHardware._BP_PID):
                continue
            matched.append((BusPirateHardware._usb_interface(p), p.device))

        matched.sort()
        n = len(matched)
        return [
            {
                "path": dev,
                "role": ("terminal" if i == 0 else "binary") if n >= 2 else "unknown",
            }
            for i, (_iface, dev) in enumerate(matched)
        ]

    @classmethod
    def connect(cls, port: str) -> BusPirateHardware:
        """Open a connection to a BusPirate 6 on the given serial port."""
        if BPIOClient is None:
            raise ImportError(
                "BPIO2 SDK not installed. "
                "See https://docs.buspirate.com/docs/binmode-reference/protocol-bpio2/"
            )
        try:
            client = BPIOClient(port)
            uart = BPIOUART(client)
            return cls(client=client, uart=uart)
        except Exception as exc:
            raise ConnectionError(str(exc)) from exc

    def configure_uart(
        self,
        speed: int = 115200,
        data_bits: int = 8,
        parity: bool = False,
        stop_bits: int = 1,
    ) -> None:
        """Configure UART mode on the BusPirate."""
        self._check_mode("uart")
        self.uart.configure(
            speed=speed,
            data_bits=data_bits,
            parity=parity,
            stop_bits=stop_bits,
            flow_control=False,
            signal_inversion=False,
            async_callback=None,
        )
        self._active_mode = "uart"

    def read(self) -> bytes:
        """Read whatever is in the UART receive buffer."""
        return self.uart.read_async()

    def write(self, data: bytes) -> None:
        """Send data over UART to the target."""
        self.uart.transfer(data, read_bytes=0)

    def _ensure_psu_ready(self) -> None:
        """Make sure the BP will actually accept a PSU request.

        The BPIO2 SDK gates every PSU set/enable behind ``config_check()``:
        until a mode has been configured, ``set_psu_enable`` silently returns
        ``None`` and the rail never turns on — the only symptom is ``applied:
        null`` over MCP (the SDK's "Not connected" message goes to stdout).
        When no bus mode is active, bring up a neutral UART configuration so a
        standalone ``set_voltage()`` / ``set_power()`` works in a single call.
        ``_active_mode`` is deliberately left ``None``: the PSU is mode-agnostic
        and not claiming a mode keeps SPI/I2C free to open later without
        ``reset_mode()``.
        """
        proto = self._active_protocol
        if getattr(proto, "configured", False):
            return
        self.uart.configure(
            speed=115200, data_bits=8, parity=False, stop_bits=1,
            flow_control=False, signal_inversion=False, async_callback=None,
        )

    def set_voltage(self, voltage_v: float, current_limit_ma: int) -> bool:
        """Set the PSU voltage + current limit and enable the rail (one step).

        The BPIO2 ``set_psu_enable`` request carries the voltage, current limit
        and the enable flag together, so this single call powers the output on
        — a separate ``set_power(True)`` is not needed.
        """
        self._ensure_psu_ready()
        proto = self._active_protocol
        voltage_mv = round(voltage_v * 1000)
        self._last_voltage_mv = voltage_mv
        self._last_current_ma = current_limit_ma
        return proto.set_psu_enable(
            voltage_mv=voltage_mv, current_ma=current_limit_ma,
        )

    def set_power(self, enable: bool) -> bool:
        """Enable or disable the power supply.

        When enabling, re-applies the last configured voltage/current. Raises
        RuntimeError if no voltage was previously configured. ``set_voltage()``
        already enables the rail, so this is only needed to toggle it off/on
        without re-specifying the voltage.
        """
        if enable and self._last_voltage_mv == 0:
            raise RuntimeError(
                "No voltage configured. Call set_voltage() before set_power(True)."
            )
        self._ensure_psu_ready()
        proto = self._active_protocol
        if enable:
            return proto.set_psu_enable(
                voltage_mv=self._last_voltage_mv,
                current_ma=self._last_current_ma,
            )
        return proto.set_psu_disable()

    @staticmethod
    def _validate_pin(pin: int) -> None:
        """Raise ValueError if pin is not a valid IO pin number (0-7)."""
        if not isinstance(pin, int) or pin < 0 or pin > 7:
            raise ValueError(f"Pin must be 0-7, got: {pin}")

    def configure_pin_input(self, pin: int) -> None:
        """Configure a pin as digital input for reading."""
        self._validate_pin(pin)
        proto = self._active_protocol
        mask = 1 << pin
        proto.set_io_direction(direction_mask=mask, direction=0)

    def get_pin_voltages(self) -> list[int]:
        """Read ADC millivolt values for all IO pins."""
        return self._active_protocol.get_adc_mv()

    def set_pin_output(self, pin: int, high: bool) -> None:
        """Set a pin as output and drive it high or low."""
        self._validate_pin(pin)
        proto = self._active_protocol
        mask = 1 << pin
        proto.set_io_direction(direction_mask=mask, direction=mask)
        proto.set_io_value(value_mask=mask, value=mask if high else 0)

    def release_pin(self, pin: int) -> None:
        """Release a pin back to input (high-impedance)."""
        self._validate_pin(pin)
        proto = self._active_protocol
        mask = 1 << pin
        proto.set_io_direction(direction_mask=mask, direction=0)

    # -- SPI --

    def configure_spi(
        self,
        speed: int = 1000000,
        clock_polarity: bool = False,
        clock_phase: bool = False,
        chip_select_idle: bool = True,
        voltage_mv: int | None = None,
        current_ma: int | None = None,
    ) -> None:
        """Configure SPI mode on the BusPirate."""
        self._check_mode("spi")
        if self.spi is None:
            self.spi = BPIOSPI(self.client)
        kwargs: dict[str, Any] = {}
        if voltage_mv is not None:
            kwargs["psu_enable"] = True
            kwargs["psu_set_mv"] = voltage_mv
            kwargs["psu_set_ma"] = current_ma or 100
        self.spi.configure(
            speed=speed,
            clock_polarity=clock_polarity,
            clock_phase=clock_phase,
            chip_select_idle=chip_select_idle,
            **kwargs,
        )
        self._active_mode = "spi"

    def spi_transfer(self, write_data: bytes, read_bytes: int = 0) -> bytes:
        """Send/receive data over SPI."""
        return self.spi.transfer(write_data=write_data, read_bytes=read_bytes)

    def spi_select(self) -> None:
        """Assert chip select (active low)."""
        return self.spi.select()

    def spi_deselect(self) -> None:
        """Deassert chip select."""
        return self.spi.deselect()

    # -- I2C --

    def configure_i2c(
        self,
        speed: int = 400000,
        clock_stretch: bool = False,
        voltage_mv: int | None = None,
        current_ma: int | None = None,
    ) -> None:
        """Configure I2C mode on the BusPirate."""
        self._check_mode("i2c")
        if self.i2c is None:
            self.i2c = BPIOI2C(self.client)
        kwargs: dict[str, Any] = {"pullup_enable": True}
        if voltage_mv is not None:
            kwargs["psu_enable"] = True
            kwargs["psu_set_mv"] = voltage_mv
            kwargs["psu_set_ma"] = current_ma or 100
        self.i2c.configure(speed=speed, clock_stretch=clock_stretch, **kwargs)
        self._active_mode = "i2c"

    def i2c_transfer(self, write_data: bytes, read_bytes: int = 0) -> bytes:
        """Send/receive data over I2C."""
        return self.i2c.transfer(write_data=write_data, read_bytes=read_bytes)

    def i2c_scan(self, start_addr: int = 0x00, end_addr: int = 0x7F) -> list:
        """Scan the I2C bus for devices.

        The SDK scan() has print() calls that would corrupt MCP stdio
        transport, so we redirect stdout during the call.
        """
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return self.i2c.scan(start_addr=start_addr, end_addr=end_addr)
        finally:
            sys.stdout = old_stdout

    # -- 1-Wire --

    def configure_1wire(
        self,
        voltage_mv: int | None = None,
        current_ma: int | None = None,
    ) -> None:
        """Configure 1-Wire mode on the BusPirate."""
        self._check_mode("1wire")
        if self.onewire is None:
            self.onewire = BPIO1Wire(self.client)
        kwargs: dict[str, Any] = {"pullup_enable": True}
        if voltage_mv is not None:
            kwargs["psu_enable"] = True
            kwargs["psu_set_mv"] = voltage_mv
            kwargs["psu_set_ma"] = current_ma or 100
        self.onewire.configure(**kwargs)
        self._active_mode = "1wire"

    def onewire_reset(self) -> Any:
        """Send a 1-Wire bus reset pulse."""
        return self.onewire.reset()

    def onewire_transfer(self, write_data: bytes, read_bytes: int = 0) -> bytes:
        """Send/receive data over 1-Wire."""
        return self.onewire.transfer(write_data=write_data, read_bytes=read_bytes)

    @staticmethod
    def find_terminal_port() -> str | None:
        """Find the BP6 terminal port (first ACM device)."""
        devices = BusPirateHardware.list_devices()
        for d in devices:
            if d["role"] == "terminal":
                return d["path"]
        return devices[0]["path"] if devices else None

    def disconnect(self) -> None:
        """Disable PSU and clean up the connection."""
        proto = self._active_protocol
        if proto is not None:
            try:
                proto.set_psu_disable()
            except Exception:
                pass  # best effort -- hardware may already be gone
        self.client = None
        self.uart = None
        self.spi = None
        self.i2c = None
        self.onewire = None
        self._active_mode = None
