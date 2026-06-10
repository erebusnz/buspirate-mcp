"""Tests for MCP tool implementations."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from buspirate_mcp.tools import (
    tool_list_devices,
    tool_verify_connection,
    tool_scan_baud,
    tool_open_uart,
    tool_read_output,
    tool_send_command,
    tool_close_uart,
    tool_set_voltage,
    tool_set_power,
    tool_measure_voltage,
    tool_enter_download_mode,
    tool_read_flash,
    tool_open_1wire,
    tool_onewire_search,
    tool_onewire_read,
    tool_close_1wire,
    tool_open_i2c,
    tool_i2c_scan,
    tool_i2c_read,
    tool_i2c_write,
    tool_i2c_dump,
    tool_close_i2c,
    tool_open_spi,
    tool_spi_probe,
    tool_spi_read,
    tool_spi_dump,
    tool_spi_write,
    tool_spi_transfer,
    tool_close_spi,
    _ascii_score,
    _onewire_crc8,
    _validate_la_command,
)
from buspirate_mcp.session import SessionManager


class TestAsciiScore:
    def test_empty_data(self):
        assert _ascii_score(b"") == 0.0

    def test_all_printable(self):
        assert _ascii_score(b"Hello World\n") == 1.0

    def test_all_garbage(self):
        assert _ascii_score(bytes(range(128, 160))) < 0.1

    def test_mixed(self):
        data = b"Hello" + bytes(range(128, 133))
        score = _ascii_score(data)
        assert 0.4 < score < 0.6

    def test_tabs_and_newlines_are_printable(self):
        assert _ascii_score(b"\t\n\r") == 1.0


class TestListDevices:
    @pytest.mark.asyncio
    async def test_returns_device_list(self):
        with patch("buspirate_mcp.tools.BusPirateHardware") as mock_hw:
            mock_hw.list_devices.return_value = [
                {"path": "/dev/ttyACM0"},
                {"path": "/dev/ttyACM1"},
            ]
            result = await tool_list_devices()
            assert len(result["devices"]) == 2

    @pytest.mark.asyncio
    async def test_returns_empty_with_hint(self):
        with patch("buspirate_mcp.tools.BusPirateHardware") as mock_hw:
            mock_hw.list_devices.return_value = []
            result = await tool_list_devices()
            assert result["devices"] == []
            assert "hint" in result


class TestVerifyConnection:
    @pytest.mark.asyncio
    async def test_detects_activity(self):
        mock_hw = MagicMock()
        mock_hw.get_pin_voltages.side_effect = [
            [0, 0, 0, 0, 3300, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3300, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        result = await tool_verify_connection(
            hardware=mock_hw, pins={"tx": 4, "rx": 5}, sample_duration_ms=100,
        )
        assert result["activity_detected"] is True

    @pytest.mark.asyncio
    async def test_no_activity_on_quiescent_target(self):
        mock_hw = MagicMock()
        mock_hw.get_pin_voltages.return_value = [0, 0, 0, 0, 3300, 0, 0, 0]
        result = await tool_verify_connection(
            hardware=mock_hw, pins={"tx": 4, "rx": 5}, sample_duration_ms=100,
        )
        assert result["activity_detected"] is False

    @pytest.mark.asyncio
    async def test_handles_none_voltages(self):
        mock_hw = MagicMock()
        mock_hw.get_pin_voltages.return_value = None
        result = await tool_verify_connection(
            hardware=mock_hw, pins={"tx": 4, "rx": 5}, sample_duration_ms=100,
        )
        assert result["activity_detected"] is False
        assert result["samples_taken"] == 0

    @pytest.mark.asyncio
    async def test_configures_uart_before_sampling(self):
        mock_hw = MagicMock()
        mock_hw.get_pin_voltages.return_value = [0, 0, 0, 0, 0, 0, 0, 0]
        await tool_verify_connection(
            hardware=mock_hw, pins={"tx": 4, "rx": 5}, sample_duration_ms=100,
        )
        mock_hw.configure_uart.assert_called_once_with(speed=115200)


class TestMeasureVoltage:
    @pytest.mark.asyncio
    async def test_reads_requested_pin(self):
        mock_hw = MagicMock()
        mock_hw.get_pin_voltages.return_value = [33, 0, 0, 0, 2500, 0, 0, 0]
        result = await tool_measure_voltage(hardware=mock_hw, pin=4)
        assert result["pin"] == 4
        assert result["millivolts"] == 2500
        assert result["volts"] == 2.5
        assert result["all_pins_mv"] == [33, 0, 0, 0, 2500, 0, 0, 0]
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_rejects_out_of_range_pin(self):
        mock_hw = MagicMock()
        result = await tool_measure_voltage(hardware=mock_hw, pin=9)
        assert result["millivolts"] is None
        assert "0-7" in result["error"]
        mock_hw.get_pin_voltages.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_no_adc_data(self):
        mock_hw = MagicMock()
        mock_hw.get_pin_voltages.return_value = None
        result = await tool_measure_voltage(hardware=mock_hw, pin=0)
        assert result["millivolts"] is None
        assert result["error"]


class TestScanBaud:
    @pytest.mark.asyncio
    async def test_finds_correct_baud(self):
        mock_hw = MagicMock()
        last_speed = [0]

        def track_configure(**kwargs):
            last_speed[0] = kwargs.get("speed", 0)

        def fake_read():
            if last_speed[0] == 9600:
                return b"Linux version 5.15.0\nroot@device:~# "
            return bytes(range(128, 160))

        mock_hw.configure_uart = MagicMock(side_effect=track_configure)
        mock_hw.read = MagicMock(side_effect=fake_read)

        result = await tool_scan_baud(hardware=mock_hw)
        assert result["recommended"] == 9600
        assert result["best_score"] >= 0.7
        assert result["candidates"][0]["baud"] == 9600

    @pytest.mark.asyncio
    async def test_reports_failure_when_no_readable_rate(self):
        mock_hw = MagicMock()
        mock_hw.configure_uart = MagicMock()
        mock_hw.read = MagicMock(return_value=bytes(range(128, 160)))

        result = await tool_scan_baud(hardware=mock_hw)
        assert result["best_score"] < 0.7
        assert result["recommended"] is None

    @pytest.mark.asyncio
    async def test_handles_none_reads(self):
        mock_hw = MagicMock()
        mock_hw.configure_uart = MagicMock()
        mock_hw.read = MagicMock(return_value=None)

        result = await tool_scan_baud(hardware=mock_hw)
        assert result["best_score"] == 0.0
        for c in result["candidates"]:
            assert c["sample_bytes"] == 0


class TestOpenUart:
    @pytest.mark.asyncio
    async def test_creates_session(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_uart(
            session_manager=mgr,
            hardware=mock_hw,
            baud=115200,
            pins={"tx": 4, "rx": 5},
            engagement_name="test-device",
        )
        assert "session_id" in result
        assert "engagement_path" in result

    @pytest.mark.asyncio
    async def test_passes_device_path(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_uart(
            session_manager=mgr,
            hardware=mock_hw,
            baud=115200,
            pins={"tx": 4, "rx": 5},
            engagement_name="test-device",
            device_path="/dev/ttyACM1",
        )
        import json
        config = json.loads(
            (tmp_path / result["engagement_path"].split("/")[-1] / "config.json").read_text()
        )
        assert config["device_path"] == "/dev/ttyACM1"


class TestReadOutput:
    @pytest.mark.asyncio
    async def test_reads_buffered_data(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.read.return_value = b"root@device:~# "
        session = mgr.create(
            name="test", hardware=mock_hw, baud=9600,
            pins={"tx": 4, "rx": 5},
        )
        result = await tool_read_output(
            session_manager=mgr, session_id=session.session_id, timeout_ms=100,
        )
        assert "root@device" in result["text"]

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.read.return_value = b""
        session = mgr.create(
            name="test", hardware=mock_hw, baud=9600,
            pins={"tx": 4, "rx": 5},
        )
        result = await tool_read_output(
            session_manager=mgr, session_id=session.session_id, timeout_ms=100,
        )
        assert result["text"] == ""
        assert result["bytes_received"] == 0

    @pytest.mark.asyncio
    async def test_handles_none_reads(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.read.return_value = None
        session = mgr.create(
            name="test", hardware=mock_hw, baud=9600,
            pins={"tx": 4, "rx": 5},
        )
        result = await tool_read_output(
            session_manager=mgr, session_id=session.session_id, timeout_ms=100,
        )
        assert result["bytes_received"] == 0


class TestSendCommand:
    @pytest.mark.asyncio
    async def test_sends_and_captures_response(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.read.side_effect = [
            b"Linux device 5.15.0\n",
            b"# ",
        ]
        session = mgr.create(
            name="test", hardware=mock_hw, baud=9600,
            pins={"tx": 4, "rx": 5},
        )
        result = await tool_send_command(
            session_manager=mgr,
            session_id=session.session_id,
            command="uname -a",
            timeout_ms=500,
        )
        assert "Linux" in result["response"]
        mock_hw.write.assert_called_once()


class TestCloseUart:
    @pytest.mark.asyncio
    async def test_closes_session(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        session = mgr.create(
            name="test", hardware=mock_hw, baud=9600,
            pins={"tx": 4, "rx": 5},
        )
        sid = session.session_id
        result = await tool_close_uart(session_manager=mgr, session_id=sid)
        assert result["closed"] is True
        with pytest.raises(KeyError):
            mgr.get(sid)


class TestSetVoltage:
    @pytest.mark.asyncio
    async def test_valid_voltage(self):
        mock_hw = MagicMock()
        mock_hw.set_voltage.return_value = True
        result = await tool_set_voltage(
            hardware=mock_hw, voltage_v=3.3, current_limit_ma=300,
        )
        assert result["applied"] is True
        mock_hw.set_voltage.assert_called_once_with(3.3, 300)

    @pytest.mark.asyncio
    async def test_out_of_range_voltage(self):
        mock_hw = MagicMock()
        result = await tool_set_voltage(
            hardware=mock_hw, voltage_v=12.0, current_limit_ma=300,
        )
        assert result["error"] is not None
        mock_hw.set_voltage.assert_not_called()


class TestSetPower:
    @pytest.mark.asyncio
    async def test_enable_power(self):
        mock_hw = MagicMock()
        mock_hw.set_power.return_value = True
        result = await tool_set_power(hardware=mock_hw, enable=True)
        assert result["applied"] is True

    @pytest.mark.asyncio
    async def test_disable_power(self):
        mock_hw = MagicMock()
        mock_hw.set_power.return_value = True
        result = await tool_set_power(hardware=mock_hw, enable=False)
        assert result["applied"] is True
        assert result["power"] == "off"


class TestEnterDownloadMode:
    @pytest.mark.asyncio
    async def test_toggles_pins(self):
        mock_hw = MagicMock()
        result = await tool_enter_download_mode(
            hardware=mock_hw, boot_pin=0, reset_pin=1,
        )
        assert result["status"] == "download_mode_entered"
        # Verify pin toggle sequence: boot LOW, reset LOW, reset HIGH, release both
        mock_hw.set_pin_output.assert_any_call(0, high=False)
        mock_hw.set_pin_output.assert_any_call(1, high=False)
        mock_hw.set_pin_output.assert_any_call(1, high=True)
        mock_hw.release_pin.assert_any_call(0)
        mock_hw.release_pin.assert_any_call(1)


class TestReadFlash:
    @pytest.mark.asyncio
    async def test_returns_error_when_esptool_missing(self, tmp_path):
        mock_hw = MagicMock()
        output = tmp_path / "dump.bin"
        with patch("buspirate_mcp.tools._enter_bridge_mode", return_value=True), \
             patch("buspirate_mcp.tools.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("esptool not found")
            result = await tool_read_flash(
                hardware=mock_hw,
                terminal_port="/dev/ttyACM0",
                offset=0,
                size=4096,
                output_path=str(output),
            )
            assert "error" in result

    @pytest.mark.asyncio
    async def test_proceeds_even_if_bridge_entry_returns_false(self, tmp_path):
        """Bridge may already be active from a previous call."""
        mock_hw = MagicMock()
        output = tmp_path / "dump.bin"
        output.write_bytes(b'\xff' * 4096)
        with patch("buspirate_mcp.tools._enter_bridge_mode", return_value=False), \
             patch("buspirate_mcp.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Read 4096 bytes", stderr="",
            )
            result = await tool_read_flash(
                hardware=mock_hw,
                terminal_port="/dev/ttyACM0",
                offset=0,
                size=4096,
                output_path=str(output),
            )
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_success_returns_output_path(self, tmp_path):
        mock_hw = MagicMock()
        output = tmp_path / "subdir" / "dump.bin"
        with patch("buspirate_mcp.tools._enter_bridge_mode", return_value=True), \
             patch("buspirate_mcp.tools.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Read 4096 bytes", stderr="",
            )
            result = await tool_read_flash(
                hardware=mock_hw,
                terminal_port="/dev/ttyACM0",
                offset=0,
                size=4096,
                output_path=str(output),
            )
            assert result["status"] == "success"
            # Path traversal prevention: directory components stripped
            assert Path(result["output_path"]).name == "dump.bin"


class TestOpen1Wire:
    @pytest.mark.asyncio
    async def test_creates_session_with_1wire_protocol(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_1wire(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-sensor",
        )
        assert "session_id" in result
        assert "engagement_path" in result
        mock_hw.configure_1wire.assert_called_once_with(None, None)

    @pytest.mark.asyncio
    async def test_passes_voltage_and_current(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        await tool_open_1wire(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-sensor",
            voltage_mv=3300,
            current_ma=200,
        )
        mock_hw.configure_1wire.assert_called_once_with(3300, 200)

    @pytest.mark.asyncio
    async def test_session_protocol_is_1wire(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_1wire(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-sensor",
        )
        session = mgr.get(result["session_id"])
        assert session.protocol == "1wire"


class TestOneWireSearch:
    @pytest.mark.asyncio
    async def test_finds_ds18b20(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        # DS18B20: family=0x28, serial=AA BB CC DD EE FF, crc over first 7 bytes
        rom_without_crc = [0x28, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]
        crc = _onewire_crc8(bytes(rom_without_crc))
        rom_bytes = rom_without_crc + [crc]

        mock_hw.onewire_reset.return_value = True
        mock_hw.onewire_transfer.return_value = rom_bytes

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        result = await tool_onewire_search(
            session_manager=mgr, session_id=session.session_id,
        )

        assert result["present"] is True
        assert result["family_code"] == "0x28"
        assert result["family_name"] == "DS18B20 (Temperature)"
        assert result["serial"] == "AABBCCDDEEFF"
        assert result["crc_valid"] is True
        assert len(result["rom_code"]) == 16  # 8 bytes = 16 hex chars

    @pytest.mark.asyncio
    async def test_logs_transaction(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        rom_without_crc = [0x28, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]
        crc = _onewire_crc8(bytes(rom_without_crc))
        rom_bytes = rom_without_crc + [crc]

        mock_hw.onewire_reset.return_value = True
        mock_hw.onewire_transfer.return_value = rom_bytes

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        await tool_onewire_search(
            session_manager=mgr, session_id=session.session_id,
        )

        log_file = session.engagement_path / "logs" / "1wire-commands.jsonl"
        assert log_file.exists()
        import json
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["operation"] == "READ_ROM"

    @pytest.mark.asyncio
    async def test_detects_bad_crc(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        # Intentionally corrupt the CRC byte
        mock_hw.onewire_reset.return_value = True
        mock_hw.onewire_transfer.return_value = [0x28, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        result = await tool_onewire_search(
            session_manager=mgr, session_id=session.session_id,
        )
        assert result["present"] is True
        assert result["crc_valid"] is False


class TestOneWireSearchNoDevice:
    @pytest.mark.asyncio
    async def test_no_device_when_reset_returns_false(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.onewire_reset.return_value = False

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        result = await tool_onewire_search(
            session_manager=mgr, session_id=session.session_id,
        )
        assert result == {"present": False}
        mock_hw.onewire_transfer.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_device_when_reset_returns_none(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.onewire_reset.return_value = None

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        result = await tool_onewire_search(
            session_manager=mgr, session_id=session.session_id,
        )
        assert result == {"present": False}
        mock_hw.onewire_transfer.assert_not_called()


class TestOneWireRead:
    @pytest.mark.asyncio
    async def test_converts_hex_and_returns_result(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.onewire_transfer.return_value = [0xAA, 0xBB]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        result = await tool_onewire_read(
            session_manager=mgr,
            session_id=session.session_id,
            write_hex="CC44",
            read_bytes=2,
        )

        assert result["tx"] == "CC44"
        assert result["rx"] == "AABB"
        assert result["bytes_read"] == 2
        mock_hw.onewire_transfer.assert_called_once_with([0xCC, 0x44], 2)

    @pytest.mark.asyncio
    async def test_logs_transaction(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.onewire_transfer.return_value = [0xAA]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        await tool_onewire_read(
            session_manager=mgr,
            session_id=session.session_id,
            write_hex="BE",
            read_bytes=1,
        )

        log_file = session.engagement_path / "logs" / "1wire-commands.jsonl"
        assert log_file.exists()
        import json
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        assert lines[-1]["operation"] == "TRANSFER"
        assert lines[-1]["tx"] == "BE"
        assert lines[-1]["rx"] == "AA"

    @pytest.mark.asyncio
    async def test_handles_empty_transfer_result(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.onewire_transfer.return_value = None

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        result = await tool_onewire_read(
            session_manager=mgr,
            session_id=session.session_id,
            write_hex="CC",
            read_bytes=0,
        )
        assert result["bytes_read"] == 0
        assert result["rx"] == ""


class TestClose1Wire:
    @pytest.mark.asyncio
    async def test_closes_session_and_resets_mode(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="1wire",
        )
        sid = session.session_id
        result = await tool_close_1wire(
            session_manager=mgr, session_id=sid, hardware=mock_hw,
        )

        assert result == {"closed": True}
        mock_hw.reset_mode.assert_called_once()
        with pytest.raises(KeyError):
            mgr.get(sid)


# ---------------------------------------------------------------------------
# SPI tool tests
# ---------------------------------------------------------------------------


class TestOpenSPI:
    @pytest.mark.asyncio
    async def test_creates_session(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_spi(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-flash",
        )
        assert "session_id" in result
        assert "engagement_path" in result
        mock_hw.configure_spi.assert_called_once_with(
            speed=1_000_000,
            clock_polarity=False,
            clock_phase=False,
            chip_select_idle=True,
            voltage_mv=None,
            current_ma=None,
        )

    @pytest.mark.asyncio
    async def test_returns_config(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_spi(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-flash",
            speed=2_000_000,
            clock_polarity=True,
            clock_phase=True,
            chip_select_idle=False,
        )
        cfg = result["spi_config"]
        assert cfg["speed"] == 2_000_000
        assert cfg["cpol"] is True
        assert cfg["cpha"] is True
        assert cfg["cs_idle"] is False

    @pytest.mark.asyncio
    async def test_session_protocol_is_spi(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_spi(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-flash",
        )
        session = mgr.get(result["session_id"])
        assert session.protocol == "spi"


class TestSPIProbe:
    @pytest.mark.asyncio
    async def test_decodes_winbond_jedec(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        # JEDEC: Winbond W25Q128 (0xEF, 0x40, 0x18)
        # Status: 0x00 (no protection)
        mock_hw.spi_transfer.side_effect = [
            [0xEF, 0x40, 0x18],  # JEDEC ID
            [0x00],              # status register
        ]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_probe(
            session_manager=mgr, session_id=session.session_id,
        )

        assert result["jedec_id"] == "EF4018"
        assert result["manufacturer"] == "Winbond"
        assert result["device_type"] == "0x40"
        assert result["capacity_bytes"] == 2 ** 0x18  # 16 MB
        assert result["capacity_human"] == "16 MB"
        assert result["status_register"] == "0x00"
        assert result["write_protected"] is False

    @pytest.mark.asyncio
    async def test_detects_write_protection(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        mock_hw.spi_transfer.side_effect = [
            [0xEF, 0x40, 0x18],
            [0x0C],  # BP1 and BP0 set
        ]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_probe(
            session_manager=mgr, session_id=session.session_id,
        )
        assert result["write_protected"] is True

    @pytest.mark.asyncio
    async def test_unknown_manufacturer(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        mock_hw.spi_transfer.side_effect = [
            [0xFF, 0x00, 0x10],
            [0x00],
        ]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_probe(
            session_manager=mgr, session_id=session.session_id,
        )
        assert "Unknown" in result["manufacturer"]

    @pytest.mark.asyncio
    async def test_bogus_jedec_capacity_capped_at_64mb(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        # Capacity code 0xFF would be 2^255 bytes without the cap
        mock_hw.spi_transfer.side_effect = [
            [0xEF, 0x40, 0xFF],
            [0x00],
        ]
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_probe(
            session_manager=mgr, session_id=session.session_id,
        )
        assert result["capacity_bytes"] == 64 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_logs_both_transactions(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        mock_hw.spi_transfer.side_effect = [
            [0xEF, 0x40, 0x18],
            [0x00],
        ]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        await tool_spi_probe(
            session_manager=mgr, session_id=session.session_id,
        )

        import json
        log_file = session.engagement_path / "logs" / "spi-commands.jsonl"
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        assert lines[0]["operation"] == "jedec_id"
        assert lines[1]["operation"] == "read_status"


class TestSPIRead:
    @pytest.mark.asyncio
    async def test_reads_and_saves_file(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        # Return 64 bytes of pattern data per chunk
        mock_hw.spi_transfer.return_value = list(range(64))

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_read(
            session_manager=mgr,
            session_id=session.session_id,
            address=0x1000,
            length=64,
        )

        assert result["bytes_read"] == 64
        assert len(result["hex_preview"]) > 0
        from pathlib import Path
        assert Path(result["file_path"]).exists()
        assert Path(result["file_path"]).stat().st_size == 64

    @pytest.mark.asyncio
    async def test_reads_in_chunks(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        # 1024 bytes = 2 chunks of 512
        mock_hw.spi_transfer.return_value = [0xAA] * 512

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_read(
            session_manager=mgr,
            session_id=session.session_id,
            address=0,
            length=1024,
        )

        assert result["bytes_read"] == 1024
        assert mock_hw.spi_transfer.call_count == 2

    @pytest.mark.asyncio
    async def test_hex_preview_is_first_64_bytes(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        mock_hw.spi_transfer.return_value = [0xDE] * 128

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_read(
            session_manager=mgr,
            session_id=session.session_id,
            address=0,
            length=128,
        )

        # 64 bytes = 128 hex chars
        assert len(result["hex_preview"]) == 128
        assert result["hex_preview"] == "de" * 64


class TestSPIDump:
    @pytest.mark.asyncio
    async def test_dumps_with_explicit_size(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        mock_hw.spi_transfer.return_value = [0xFF] * 512

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_dump(
            session_manager=mgr,
            session_id=session.session_id,
            size=1024,
        )

        assert result["bytes_read"] == 1024
        assert "elapsed_s" in result
        assert "speed_kbps" in result
        from pathlib import Path
        assert Path(result["file_path"]).exists()
        assert Path(result["file_path"]).stat().st_size == 1024

    @pytest.mark.asyncio
    async def test_auto_detects_size_from_jedec(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        call_count = [0]

        def fake_transfer(write_data, read_bytes):
            call_count[0] += 1
            if call_count[0] == 1:
                # JEDEC ID query: 2**10 = 1024 bytes
                return [0xEF, 0x40, 0x0A]
            return [0xFF] * read_bytes

        mock_hw.spi_transfer.side_effect = fake_transfer

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_dump(
            session_manager=mgr,
            session_id=session.session_id,
        )

        assert result["bytes_read"] == 1024

    @pytest.mark.asyncio
    async def test_custom_output_filename(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.spi_transfer.return_value = [0x00] * 512

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_dump(
            session_manager=mgr,
            session_id=session.session_id,
            size=512,
            output_filename="custom.bin",
        )

        assert result["file_path"].endswith("custom.bin")

    @pytest.mark.asyncio
    async def test_returns_error_when_size_unknown(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        # JEDEC returns capacity exponent 0 -> size 0
        mock_hw.spi_transfer.return_value = [0xFF, 0xFF, 0x00]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_dump(
            session_manager=mgr,
            session_id=session.session_id,
        )

        assert "error" in result


class TestSPIWrite:
    @pytest.mark.asyncio
    async def test_writes_with_erase_and_verify(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        # Data to write: 128 bytes
        input_file = tmp_path / "firmware.bin"
        payload = bytes(range(128))
        input_file.write_bytes(payload)

        call_log = []

        def fake_transfer(write_data, read_bytes):
            call_log.append((write_data[:1], read_bytes))
            # Write Enable -> no read
            if write_data[0] == 0x06:
                return []
            # Chip Erase -> no read
            if write_data[0] == 0xC7:
                return []
            # Read Status -> return WIP=1 first time, then WIP=0
            if write_data[0] == 0x05:
                # Use a counter to simulate busy then ready
                if not hasattr(fake_transfer, '_status_count'):
                    fake_transfer._status_count = 0
                fake_transfer._status_count += 1
                if fake_transfer._status_count % 2 == 1:
                    return [0x01]  # WIP set
                return [0x00]  # WIP clear
            # Page Program -> no read
            if write_data[0] == 0x02:
                return []
            # Read Data (verify) -> return the original payload
            if write_data[0] == 0x03:
                addr = (write_data[1] << 16) | (write_data[2] << 8) | write_data[3]
                return list(payload[addr : addr + read_bytes])
            return [0x00] * read_bytes

        mock_hw.spi_transfer.side_effect = fake_transfer

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_write(
            session_manager=mgr,
            session_id=session.session_id,
            input_path=str(input_file),
        )

        assert result["bytes_written"] == 128
        assert result["erased"] is True
        assert result["verified"] is True
        assert "elapsed_s" in result

    @pytest.mark.asyncio
    async def test_write_without_erase(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        input_file = tmp_path / "small.bin"
        input_file.write_bytes(b"\xAA" * 16)

        def fake_transfer(write_data, read_bytes):
            if write_data[0] == 0x06:
                return []
            if write_data[0] == 0x02:
                return []
            if write_data[0] == 0x05:
                return [0x00]  # always ready
            if write_data[0] == 0x03:
                return [0xAA] * read_bytes
            return [0x00] * read_bytes

        mock_hw.spi_transfer.side_effect = fake_transfer

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_write(
            session_manager=mgr,
            session_id=session.session_id,
            input_path=str(input_file),
            erase=False,
        )

        assert result["erased"] is False
        assert result["verified"] is True

    @pytest.mark.asyncio
    async def test_verify_detects_mismatch(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        input_file = tmp_path / "data.bin"
        input_file.write_bytes(b"\xBB" * 16)

        def fake_transfer(write_data, read_bytes):
            if write_data[0] == 0x06:
                return []
            if write_data[0] == 0x02:
                return []
            if write_data[0] == 0x05:
                return [0x00]
            if write_data[0] == 0x03:
                return [0xCC] * read_bytes  # wrong data
            return []

        mock_hw.spi_transfer.side_effect = fake_transfer

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_write(
            session_manager=mgr,
            session_id=session.session_id,
            input_path=str(input_file),
            erase=False,
        )

        assert result["verified"] is False


class TestSPITransfer:
    @pytest.mark.asyncio
    async def test_hex_conversion_both_ways(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.spi_transfer.return_value = [0xDE, 0xAD]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_transfer(
            session_manager=mgr,
            session_id=session.session_id,
            write_hex="9f",
            read_bytes=2,
        )

        assert result["tx"] == "9f"
        assert result["rx"] == "dead"
        assert result["bytes_read"] == 2
        mock_hw.spi_transfer.assert_called_once_with([0x9F], 2)

    @pytest.mark.asyncio
    async def test_write_only(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.spi_transfer.return_value = []

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        result = await tool_spi_transfer(
            session_manager=mgr,
            session_id=session.session_id,
            write_hex="0600",
            read_bytes=0,
        )

        assert result["rx"] == ""
        assert result["bytes_read"] == 0

    @pytest.mark.asyncio
    async def test_logs_transaction(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.spi_transfer.return_value = [0xAB]

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        await tool_spi_transfer(
            session_manager=mgr,
            session_id=session.session_id,
            write_hex="9f",
            read_bytes=1,
        )

        import json
        log_file = session.engagement_path / "logs" / "spi-commands.jsonl"
        lines = [json.loads(l) for l in log_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["operation"] == "raw_transfer"


class TestCloseSPI:
    @pytest.mark.asyncio
    async def test_closes_session_and_resets_mode(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()

        session = mgr.create(
            name="test", hardware=mock_hw, protocol="spi",
        )
        sid = session.session_id
        result = await tool_close_spi(
            session_manager=mgr, session_id=sid, hardware=mock_hw,
        )

        assert result["closed"] is True
        assert result["session_id"] == sid
        mock_hw.reset_mode.assert_called_once()
        with pytest.raises(KeyError):
            mgr.get(sid)


# ---------------------------------------------------------------------------
# I2C tool tests
# ---------------------------------------------------------------------------


class TestOpenI2C:
    @pytest.mark.asyncio
    async def test_creates_session_with_i2c_protocol(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_i2c(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-i2c-device",
        )
        assert "session_id" in result
        assert "engagement_path" in result
        assert "i2c_config" in result
        mock_hw.configure_i2c.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_speed_is_400khz(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_i2c(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="test-i2c-device",
        )
        assert result["i2c_config"]["speed"] == 400000

    @pytest.mark.asyncio
    async def test_config_written_to_disk(self, tmp_path):
        import json
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        result = await tool_open_i2c(
            session_manager=mgr,
            hardware=mock_hw,
            engagement_name="mydev",
            speed=100000,
            clock_stretch=True,
        )
        eng_path = result["engagement_path"]
        config = json.loads((tmp_path / eng_path.split("/")[-1] / "config.json").read_text())
        assert config["protocol"] == "i2c"
        assert config["protocol_config"]["speed"] == 100000
        assert config["protocol_config"]["clock_stretch"] is True


class TestI2CScan:
    @pytest.mark.asyncio
    async def test_deduplicates_read_write_addresses(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        # 0xA0 (write) and 0xA1 (read) both shift to 0x50; same for 0xD0/0xD1 -> 0x68
        mock_hw.i2c_scan.return_value = [0xA0, 0xA1, 0xD0, 0xD1]
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_scan(session_manager=mgr, session_id=session.session_id)
        assert result["count"] == 2
        addrs = [d["address_7bit"] for d in result["devices"]]
        assert 0x50 in addrs
        assert 0x68 in addrs

    @pytest.mark.asyncio
    async def test_hints_are_populated(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_scan.return_value = [0xA0, 0xA1, 0xD0, 0xD1]
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_scan(session_manager=mgr, session_id=session.session_id)
        hints = {d["address_7bit"]: d["hint"] for d in result["devices"]}
        assert hints[0x50] == "EEPROM (24Cxx series)"
        assert hints[0x68] == "RTC/IMU (DS3231, MPU6050)"

    @pytest.mark.asyncio
    async def test_unknown_address_hint_is_none(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        # 0x10 >> 1 = 0x08, not in any known range
        mock_hw.i2c_scan.return_value = [0x10]
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_scan(session_manager=mgr, session_id=session.session_id)
        assert result["devices"][0]["hint"] is None

    @pytest.mark.asyncio
    async def test_empty_bus(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_scan.return_value = []
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_scan(session_manager=mgr, session_id=session.session_id)
        assert result["count"] == 0
        assert result["devices"] == []

    @pytest.mark.asyncio
    async def test_scan_calls_full_range(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_scan.return_value = []
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        await tool_i2c_scan(session_manager=mgr, session_id=session.session_id)
        mock_hw.i2c_scan.assert_called_once_with(0x00, 0x7F)


class TestI2CRead:
    @pytest.mark.asyncio
    async def test_correct_address_shifting(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = bytes([0x42])
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        await tool_i2c_read(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            register_addr=0x00,
            length=1,
        )
        # 0x50 << 1 = 0xA0, write address
        call_args = mock_hw.i2c_transfer.call_args
        write_data = call_args[0][0]
        assert write_data[0] == 0xA0
        assert write_data[1] == 0x00

    @pytest.mark.asyncio
    async def test_returns_data_correctly(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = bytes([0xDE, 0xAD])
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_read(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr=0x50,
            register_addr=0x10,
            length=2,
        )
        assert result["data_bytes"] == [0xDE, 0xAD]
        assert result["data_hex"] == "dead"
        assert result["length"] == 2
        assert result["address"] == "0x50"
        assert result["register"] == "0x10"

    @pytest.mark.asyncio
    async def test_no_register_addr(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = bytes([0xFF])
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_read(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
        )
        call_args = mock_hw.i2c_transfer.call_args
        write_data = call_args[0][0]
        # Only addr_write byte, no register
        assert len(write_data) == 1
        assert result["register"] is None

    @pytest.mark.asyncio
    async def test_int_device_addr(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = bytes([0x00])
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_read(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr=0x68,
            register_addr=0x00,
        )
        assert result["address"] == "0x68"


class TestI2CWrite:
    @pytest.mark.asyncio
    async def test_writes_correct_bytes(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = b""
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        await tool_i2c_write(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            register_addr=0x00,
            data_hex="deadbeef",
        )
        call_args = mock_hw.i2c_transfer.call_args
        write_data = call_args[0][0]
        # [addr_write=0xA0, reg=0x00, 0xDE, 0xAD, 0xBE, 0xEF]
        assert write_data == [0xA0, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]
        assert call_args[1]["read_bytes"] == 0

    @pytest.mark.asyncio
    async def test_returns_written_true(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = b""
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_write(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            register_addr=0x10,
            data_hex="ff",
        )
        assert result["written"] is True
        assert result["bytes_written"] == 1
        assert result["address"] == "0x50"
        assert result["register"] == "0x10"


class TestI2CDump:
    @pytest.mark.asyncio
    async def test_saves_dump_file(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        # Return 32 bytes of data for each chunk
        mock_hw.i2c_transfer.return_value = bytes(range(32))
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_dump(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            size=64,
        )
        assert result["bytes_read"] == 64
        from pathlib import Path
        dump_path = Path(result["file_path"])
        assert dump_path.exists()
        assert dump_path.name == "i2c_dump_50.bin"
        assert dump_path.stat().st_size == 64

    @pytest.mark.asyncio
    async def test_reads_in_chunks(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = bytes(32)
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        await tool_i2c_dump(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            size=64,
        )
        # 64 / 32 = 2 calls
        assert mock_hw.i2c_transfer.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_none_chunk(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = None
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_dump(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            size=32,
        )
        # Should fill with 0xff when chunk is None
        assert result["bytes_read"] == 32
        from pathlib import Path
        data = Path(result["file_path"]).read_bytes()
        assert all(b == 0xFF for b in data)

    @pytest.mark.asyncio
    async def test_hex_preview_is_first_64_bytes(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = bytes(range(32))
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_dump(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            size=256,
        )
        # Preview should be hex of first 64 bytes = 128 hex chars
        assert len(result["hex_preview"]) == 128

    @pytest.mark.asyncio
    async def test_register_bytes_2_sends_msb_first(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        mock_hw.i2c_transfer.return_value = bytes(32)
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        # Read 64 bytes starting at offset 0x1234 with 16-bit addressing
        await tool_i2c_dump(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            size=64,
            register_bytes=2,
        )
        # First call should have [addr_write, 0x00, 0x00] for offset 0
        first_call_args = mock_hw.i2c_transfer.call_args_list[0]
        assert first_call_args[0][0] == [0xA0, 0x00, 0x00]
        # Second call at offset 32 should be [addr_write, 0x00, 0x20]
        second_call_args = mock_hw.i2c_transfer.call_args_list[1]
        assert second_call_args[0][0] == [0xA0, 0x00, 0x20]

    @pytest.mark.asyncio
    async def test_register_bytes_invalid_rejected(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        result = await tool_i2c_dump(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            size=64,
            register_bytes=4,
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_register_bytes_1_rejects_oversize(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        # 8-bit addressing can't go beyond 256 bytes
        result = await tool_i2c_dump(
            session_manager=mgr,
            session_id=session.session_id,
            device_addr="0x50",
            size=512,
            register_bytes=1,
        )
        assert "error" in result


class TestCloseI2C:
    @pytest.mark.asyncio
    async def test_closes_session(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        sid = session.session_id
        result = await tool_close_i2c(
            session_manager=mgr, session_id=sid, hardware=mock_hw,
        )
        assert result["closed"] is True
        with pytest.raises(KeyError):
            mgr.get(sid)

    @pytest.mark.asyncio
    async def test_resets_mode(self, tmp_path):
        mgr = SessionManager(engagements_dir=tmp_path)
        mock_hw = MagicMock()
        session = mgr.create(
            name="test", hardware=mock_hw, protocol="i2c",
        )
        await tool_close_i2c(
            session_manager=mgr, session_id=session.session_id, hardware=mock_hw,
        )
        mock_hw.reset_mode.assert_called_once()


class TestValidateLaCommand:
    def test_valid_simple(self):
        assert _validate_la_command("[0x9f r:3]") is None

    def test_valid_multi_transaction(self):
        assert _validate_la_command("[0x9f r:3][0x05 r]") is None

    def test_empty(self):
        assert "empty" in _validate_la_command("")

    def test_whitespace_only(self):
        assert "empty" in _validate_la_command("   ")

    def test_too_long(self):
        assert "too long" in _validate_la_command("[" + "a" * 600 + "]")

    def test_newline_rejected(self):
        assert "control" in _validate_la_command("[0x9f]\n[0x05]")

    def test_non_ascii_rejected(self):
        assert "control" in _validate_la_command("[0x9f é r:3]")

    def test_unbalanced_brackets(self):
        assert "unbalanced" in _validate_la_command("[0x9f r:3")

    def test_no_brackets(self):
        assert "missing bus transaction brackets" in _validate_la_command("0x9f r:3")
