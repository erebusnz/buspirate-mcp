# buspirate-mcp

MCP server for BusPirate 6 hardware security testing. Exposes UART, SPI, I2C, 1-Wire, power supply, GPIO, and logic analyzer operations as [Model Context Protocol](https://modelcontextprotocol.io/) tools over stdio transport. 34 tools across 4 bus protocols + logic analyzer.

Built for use with Claude Code but works with any MCP client.

## What it does

- **UART probing:** detect baud rates, capture serial output, interact with debug consoles
- **SPI flash:** probe JEDEC ID, read/dump/write SPI flash chips (W25Q series, SOIC-8 clip)
- **I2C bus:** scan for devices, read/write registers, dump EEPROM contents
- **1-Wire:** enumerate devices, read ROM codes (DS18B20, iButton)
- **Power control:** voltage/current management with safety tiers
- **GPIO control:** toggle pins for bootloader entry (ESP32, ESP8266, etc.)
- **Flash extraction:** dump firmware through UART bridge via esptool
- **Logic analyzer:** follow-along capture at 75 MHz on all 8 IO pins with protocol identification
- **Engagement logging:** per-protocol logs (UART text, SPI/I2C/1-Wire JSONL), per-engagement folders

## Requirements

- Python 3.11+
- BusPirate 6 with BPIO2 binary mode enabled
- User in `dialout` group for serial access (`sudo usermod -aG dialout $USER`)

## Install

```bash
git clone --recurse-submodules https://github.com/mplogas/buspirate-mcp.git
cd buspirate-mcp
pip install -e ".[dev]"
```

## MCP Client Configuration

Copy the example config and adjust paths for your machine:

```bash
cp .mcp.json.example .mcp.json
# Edit .mcp.json with the absolute path to your venv's python
```

The `.mcp.json` is gitignored since paths are machine-specific. Example config:

```json
{
  "mcpServers": {
    "buspirate": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "buspirate_mcp"]
    }
  }
}
```

Set `PIDEV_ENGAGEMENTS_DIR` environment variable to control where engagement logs are written. Defaults to `./engagements/` relative to the package root.

## Tools

| Tool | Safety Tier | Description |
|---|---|---|
| `list_devices` | read-only | Find BusPirate devices on USB |
| `verify_connection` | read-only | Check for signal activity on pins |
| `scan_baud` | read-only | Auto-detect baud rate |
| `read_output` | read-only | Read UART data from target |
| `open_uart` | allowed-write | Open persistent UART session with logging |
| `send_command` | allowed-write | Send command and capture response |
| `close_uart` | allowed-write | Close session, finalize logs |
| `enter_download_mode` | allowed-write | GPIO toggle for bootloader entry |
| `read_flash` | allowed-write | Dump flash via esptool through UART bridge |
| `set_voltage` | approval-write | Set PSU voltage + current limit and enable, one call (requires confirmation) |
| `set_power` | approval-write | Enable/disable PSU (requires confirmation) |
| `measure_voltage` | read-only | Read DC voltage (mV) on an IO pin via ADC; e.g. read back VOUT |
| **SPI** | | |
| `open_spi` | allowed-write | Configure SPI mode, create session |
| `spi_probe` | read-only | Read JEDEC ID + status register, decode chip |
| `spi_read` | read-only | Read N bytes from SPI flash address |
| `spi_dump` | read-only | Full flash dump to binary file |
| `spi_write` | approval-write | Erase + program + verify SPI flash (requires confirmation) |
| `spi_transfer` | read-only | Raw SPI transfer (hex in/out) |
| `close_spi` | allowed-write | Close SPI session, reset mode |
| **I2C** | | |
| `open_i2c` | allowed-write | Configure I2C mode, create session |
| `i2c_scan` | read-only | Scan bus for devices (0x00-0x7F) |
| `i2c_read` | read-only | Read bytes from device + register |
| `i2c_write` | approval-write | Write bytes to device + register (requires confirmation) |
| `i2c_dump` | read-only | Dump EEPROM contents to file |
| `close_i2c` | allowed-write | Close I2C session, reset mode |
| **1-Wire** | | |
| `open_1wire` | allowed-write | Configure 1-Wire mode, create session |
| `onewire_search` | read-only | Reset bus, Read ROM, decode family code |
| `onewire_read` | read-only | Raw 1-Wire transaction |
| `close_1wire` | allowed-write | Close 1-Wire session, reset mode |
| **Logic Analyzer** | | |
| `la_prepare` | allowed-write | Switch to FALA mode, enter bus protocol for capture |
| `la_command` | read-only | Execute bus command and capture signal data |
| `la_analyze` | read-only | Analyze capture for signal characteristics |
| `la_identify` | read-only | Auto-identify bus protocols from capture |
| `la_cleanup` | allowed-write | Restore BPIO2 mode |

## Safety Model

Three tiers enforced at the MCP server boundary:

- **read-only:** full autonomy, no side effects
- **allowed-write:** autonomous execution, all calls logged
- **approval-write:** blocks until human confirms via `_confirmed` parameter. Wrong voltage fries chips. SPI flash writes and I2C writes overwrite data irreversibly.

## BusPirate 6 Setup

### Enable BPIO2 binary mode

Connect to the BP6 terminal port (ACM0) and run:

```
binmode -> select 2 (BPIO2 flatbuffer interface) -> y to save
```

This persists across reboots.

### Pin mapping

UART mode uses IO4 (TX->) and IO5 (RX<-). The BP6 display is the authoritative source for pin assignment.

Free pins (IO0-IO3, IO6-IO7) can be used for GPIO control (bootloader entry, reset) while UART is active.

### Project integration

The `open_uart` tool accepts an optional `project_path` parameter. When provided (from project-mcp's `create_project`), engagement data is written to `<project_path>/uart/` instead of creating a standalone folder. Omit it for standalone use.

### Known limitations

- **Bridge mode exit:** after flash dump operations, bridge mode stays active. Press the physical BP6 button or USB-replug to exit.
- **Dual-port operation:** BPIO2 (binary port) and bridge mode (terminal port) cannot operate simultaneously.
- **Baud scan timing:** fast-booting targets may finish output before the scan reaches the correct rate.
- **Port numbering:** the binary port number may change after USB replug (ACM1 -> ACM2). The server auto-detects.

## Tests

```bash
pytest              # 203 tests, no hardware needed
pytest -m hardware  # integration tests, BP6 must be connected
```

## License

MIT
