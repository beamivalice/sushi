#!/bin/sh
set -eu
cd "$(dirname "$0")"
output="${BUILD_DIR:-$PWD/build}/tests"
mkdir -p "$output"
xcrun swiftc -parse-as-library Client.swift LaunchConfiguration.swift ConnectionProfile.swift ClientTests.swift -o "$output/client-tests"
"$output/client-tests"
xcrun swiftc -parse-as-library Client.swift ToolTask.swift ToolTaskTests.swift -o "$output/tool-tests"
if [ "$#" -gt 0 ]; then "$output/tool-tests" "$1"; else "$output/tool-tests"; fi
xcrun swiftc -parse-as-library -D DESIGN_PREVIEW Client.swift ServerStore.swift SystemStats.swift LaunchConfiguration.swift DesignSystem.swift ToolTask.swift ToolsPanel.swift ConnectionProfile.swift ManagedServer.swift SushiIcon.swift PerformancePanel.swift IntegrationsPanel.swift ModelSettings.swift ModelSettingsPanel.swift AboutPanel.swift Benchmark.swift BenchmarkPanel.swift SetupGuide.swift Conversion.swift ConversionPanel.swift TaskActivityPanel.swift UsageHistory.swift UsageHistoryPanel.swift LocalModelFiles.swift LocalModelFilesPanel.swift HubModels.swift HubBrowserPanel.swift SushiControl.swift StoreTests.swift -o "$output/store-tests" -framework SwiftUI -framework AppKit -framework ServiceManagement -framework Security
if [ "$#" -gt 0 ]; then
    test_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
    "$output/store-tests" "$1" "$test_port"
else
    "$output/store-tests"
fi
python3 ../run-metrics-check.py "$output/store-tests"

if [ "$#" -gt 0 ]; then
    xcrun swiftc -parse-as-library Client.swift ToolTask.swift DownloadTaskTests.swift -o "$output/download-tests"
    python3 ../run-download-check.py "$1" "$output/download-tests"
fi

xcrun swiftc -parse-as-library Client.swift ModelSettings.swift ModelSettingsTests.swift -o "$output/model-settings-tests"
"$output/model-settings-tests"

xcrun swiftc -parse-as-library Client.swift Benchmark.swift BenchmarkTests.swift -o "$output/benchmark-tests"
"$output/benchmark-tests"

xcrun swiftc -parse-as-library Client.swift Benchmark.swift BenchmarkHTTPTests.swift -o "$output/benchmark-http-tests"
python3 ../run-benchmark-check.py "$output/benchmark-http-tests"

xcrun swiftc -parse-as-library Client.swift Conversion.swift ToolTask.swift ConversionTests.swift -o "$output/conversion-tests"
if [ "$#" -gt 0 ]; then "$output/conversion-tests" "$1"; else "$output/conversion-tests"; fi

xcrun swiftc -parse-as-library Client.swift ControllerHTTPTests.swift -o "$output/controller-http-tests"
python3 ../run-controller-check.py "$output/controller-http-tests"

xcrun swiftc -parse-as-library Client.swift UsageHistory.swift UsageHistoryTests.swift -o "$output/usage-tests"
"$output/usage-tests"

xcrun swiftc -parse-as-library Client.swift ControllerLifecycleTests.swift -o "$output/controller-lifecycle-tests"

xcrun swiftc -parse-as-library Client.swift LocalModelFiles.swift LocalModelFilesTests.swift -o "$output/local-model-tests"
"$output/local-model-tests"

xcrun swiftc -parse-as-library Client.swift ToolTask.swift HubModels.swift HubModelsTests.swift -o "$output/hub-tests"
"$output/hub-tests"

xcrun swiftc -parse-as-library Client.swift ManagedServer.swift ManagedServerTests.swift -o "$output/managed-server-tests"
"$output/managed-server-tests" --policy-only
