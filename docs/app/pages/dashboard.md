# Dashboard

The Dashboard is your mission control for monitoring GPU resources, training jobs, and system health.

## Widgets

The dashboard uses a customizable widget system. You can add, remove, and resize widgets to fit your workflow.

### GPU Sparklines

Real-time GPU utilization trends for each GPU. Shows the last 2 minutes of utilization as a sparkline chart.

**Use case**: Quickly spot GPU underutilization or bottlenecks.

### Power Meters

Displays current power draw and power limit for each GPU with historical sparklines.

**DGX Spark Support**: GPU 0 on DGX Spark systems shows calibrated 240W max power.

**Color coding**:
- Green: < 70% power usage
- Yellow: 70-85% power usage
- Red: > 85% power usage

### Temperature Heatmap

Visual heatmap of GPU temperatures. Hover to see exact temperature values.

### GPU Status

Detailed GPU information including:
- Memory usage (used/total)
- Utilization percentage
- Temperature
- Currently running jobs

### Network Traffic

Shows network RX/TX rates with sparklines. Useful for monitoring data pipeline I/O.

### Recent Jobs

List of recently started training jobs with status indicators.

## Customization

Click the gear icon on any widget to:
- Change widget size (small/medium/large)
- Remove from dashboard
- Configure refresh rate

Click **+ Add Widget** to add new widgets to your dashboard.

## Keyboard Shortcuts

- `Cmd/Ctrl + K`: Open command palette
- `Cmd/Ctrl + /`: Toggle sidebar
- `Escape`: Close modals
