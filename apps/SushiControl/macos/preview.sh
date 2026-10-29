#!/bin/sh
set -eu
cd "$(dirname "$0")"
output="${BUILD_DIR:-$PWD/build}"
mkdir -p "$output"
xcrun swiftc -O -parse-as-library -D DESIGN_PREVIEW Client.swift ServerStore.swift SystemStats.swift LaunchConfiguration.swift DesignSystem.swift ToolTask.swift ToolsPanel.swift ConnectionProfile.swift ManagedServer.swift SushiIcon.swift PerformancePanel.swift IntegrationsPanel.swift ModelSettings.swift ModelSettingsPanel.swift AboutPanel.swift Benchmark.swift BenchmarkPanel.swift SetupGuide.swift Conversion.swift ConversionPanel.swift TaskActivityPanel.swift UsageHistory.swift UsageHistoryPanel.swift LocalModelFiles.swift LocalModelFilesPanel.swift HubModels.swift HubBrowserPanel.swift SushiControl.swift DesignPreview.swift -o "$output/design-preview" -framework SwiftUI -framework AppKit -framework ServiceManagement -framework Security
"$output/design-preview" "$output"
