#!/bin/sh
set -eu
cd "$(dirname "$0")"
output="${BUILD_DIR:-$PWD/build}"
mkdir -p "$output"/'Sushi Control.app'/Contents/MacOS "$output"/'Sushi Control.app'/Contents/Resources
xcrun swiftc -O -parse-as-library -target arm64-apple-macosx14.0 Client.swift ServerStore.swift SystemStats.swift LaunchConfiguration.swift DesignSystem.swift ToolTask.swift ToolsPanel.swift ConnectionProfile.swift ManagedServer.swift SushiIcon.swift PerformancePanel.swift IntegrationsPanel.swift ModelSettings.swift ModelSettingsPanel.swift AboutPanel.swift Benchmark.swift BenchmarkPanel.swift SetupGuide.swift Conversion.swift ConversionPanel.swift TaskActivityPanel.swift UsageHistory.swift UsageHistoryPanel.swift LocalModelFiles.swift LocalModelFilesPanel.swift HubModels.swift HubBrowserPanel.swift SushiControl.swift -o "$output"/'Sushi Control.app/Contents/MacOS/SushiControl' -framework SwiftUI -framework AppKit -framework ServiceManagement -framework Security
xcrun swiftc -O -parse-as-library SushiIcon.swift ExportIcon.swift -o "$output/export-icon" -framework AppKit
"$output/export-icon" "$output/AppIcon.iconset"
iconutil -c icns "$output/AppIcon.iconset" -o "$output/Sushi Control.app/Contents/Resources/AppIcon.icns"
cat > "$output"/'Sushi Control.app/Contents/Info.plist' <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>org.sushi.SushiControl</string>
<key>CFBundleIconFile</key><string>AppIcon</string>
<key>CFBundleName</key><string>Sushi Control</string>
<key>CFBundleExecutable</key><string>SushiControl</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.1</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>LSUIElement</key><true/>
</dict></plist>
PLIST
license_dir="$(cd .. && pwd)"
while [ ! -f "$license_dir/LICENSE" ] && [ "$license_dir" != / ]; do license_dir="$(dirname "$license_dir")"; done
for license in LICENSE LICENSE-APACHE-2.0 NOTICE; do
    COPYFILE_DISABLE=1 cp -X "$license_dir/$license" "$output"/'Sushi Control.app/Contents/Resources/'
done
xattr -cr "$output/Sushi Control.app"
identity="${SIGNING_IDENTITY:--}"
if [ "$identity" = "-" ]; then
    codesign --force --options runtime --sign - "$output/Sushi Control.app"
else
    codesign --force --options runtime --timestamp --sign "$identity" "$output/Sushi Control.app"
fi

codesign --verify --strict "$output/Sushi Control.app"
printf '%s\n' "$output/Sushi Control.app"
