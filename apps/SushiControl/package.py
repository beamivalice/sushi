import hashlib, json, os, pathlib, plistlib, shutil, subprocess, tempfile, zipfile
root=pathlib.Path(__file__).resolve().parent
build=pathlib.Path(os.environ.get('BUILD_DIR','/private/tmp/sushi-control-release')).expanduser().resolve()
subprocess.run([str(root/'macos/build.sh')],env=dict(os.environ,BUILD_DIR=str(build)),check=True,timeout=180)
app=build/'Sushi Control.app'
with (app/'Contents/Info.plist').open('rb') as handle: version=plistlib.load(handle)['CFBundleShortVersionString']
dist=root/'dist'; dist.mkdir(exist_ok=True)
app_zip=dist/f'Sushi-Control-{version}-macos-arm64.zip'
subprocess.run(['/usr/bin/ditto','-c','-k','--keepParent','--norsrc',str(app),str(app_zip)],check=True,timeout=60)
source_zip=dist/f'Sushi-Control-{version}-source.zip'
files=[root/name for name in ['.gitignore','README.md','run-stock-check.py','run-download-check.py','run-benchmark-check.py','run-controller-check.py','run-metrics-check.py','package.py']]
license_root=next(parent for parent in [root,*root.parents] if all((parent/name).is_file() for name in ['LICENSE','LICENSE-APACHE-2.0','NOTICE']))
files+=sorted((root/'macos').glob('*.swift'))+sorted((root/'macos').glob('*.sh'))
with zipfile.ZipFile(source_zip,'w',compression=zipfile.ZIP_DEFLATED) as archive:
 for path in files: archive.write(path,f'Sushi-Control-{version}/'+str(path.relative_to(root)))
 for name in ['LICENSE','LICENSE-APACHE-2.0','NOTICE']: archive.write(license_root/name,f'Sushi-Control-{version}/'+name)
with tempfile.TemporaryDirectory(prefix='sushi-package-check-') as directory:
 subprocess.run(['/usr/bin/ditto','-x','-k',str(app_zip),directory],check=True,timeout=30)
 subprocess.run(['/usr/bin/codesign','--verify','--strict',str(pathlib.Path(directory)/app.name)],check=True,timeout=20)
with tempfile.TemporaryDirectory(prefix='sushi-source-check-') as directory:
 subprocess.run(['/usr/bin/ditto','-x','-k',str(source_zip),directory],check=True,timeout=30)
 source=pathlib.Path(directory)/f'Sushi-Control-{version}'
 subprocess.run([str(source/'macos/build.sh')],env=dict(os.environ,BUILD_DIR=str(pathlib.Path(directory)/'compiled')),check=True,timeout=180)
checksums={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [app_zip,source_zip]}
(dist/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in checksums.items()))
(dist/'manifest.json').write_text(json.dumps({'version':version,'architecture':'arm64','signature':'ad-hoc' if os.environ.get('SIGNING_IDENTITY','-') == '-' else 'developer-signed','notarized':False,'files':checksums},indent=2)+'\n')
print(json.dumps(checksums,indent=2))
