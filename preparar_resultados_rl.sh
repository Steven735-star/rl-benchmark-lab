#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, or pass the root directory as the first argument.
ROOT="${1:-$(pwd)}"
if [[ ! -d "$ROOT/.git" && ! -f "$ROOT/.git" ]]; then
  echo "ERROR: '$ROOT' no parece ser la raíz de un repositorio Git." >&2
  echo 'Ejecuta el script desde REPOSITORY o pasa su ruta como primer argumento.' >&2
  exit 1
fi
ROOT="$(cd "$ROOT" && pwd)"
PROJECT="$ROOT/rl_dqn_project"
ARCHIVE="$PROJECT/resultados_full_rl.zip"
TARGET="$PROJECT/results/full"
NESTED_IGNORE="$PROJECT/.gitignore"

for needed in "$PROJECT" "$ARCHIVE" "$NESTED_IGNORE"; do
  if [[ ! -e "$needed" ]]; then
    echo "ERROR: No se encuentra $needed" >&2
    exit 1
  fi
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: Se requiere python3 para inspeccionar el ZIP con seguridad." >&2
  exit 1
fi

printf '\n=== Estado previo de Git ===\n'
git -C "$ROOT" status --short

printf '\n=== Recuperación segura de resultados ===\n'
python3 - "$ARCHIVE" "$TARGET" "$NESTED_IGNORE" <<'PY'
import hashlib
from pathlib import Path
from collections import Counter
import sys
import zipfile

archive, target, nested_ignore = map(Path, sys.argv[1:])

# Update exactly this exclusion while leaving unrelated user rules intact.
text = nested_ignore.read_text(encoding='utf-8')
lines = text.splitlines(keepends=True)
remove = {'results/full/', '/results/full/', 'results/full', '/results/full'}
kept = [line for line in lines if line.strip() not in remove]
if len(kept) != len(lines):
    nested_ignore.write_text(''.join(kept), encoding='utf-8')
    print(f'.gitignore ajustado: {nested_ignore} (regla results/full/ retirada)')
else:
    print(f'Sin regla results/full/ que eliminar en {nested_ignore}')

def check_same(dest, member, zf):
    if dest.stat().st_size != member.file_size:
        return False
    h1 = hashlib.sha256()
    with open(dest, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h1.update(block)
    h2 = hashlib.sha256()
    with zf.open(member) as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h2.update(block)
    return h1.digest() == h2.digest()

with zipfile.ZipFile(archive) as zf:
    members = [m for m in zf.infolist() if not m.is_dir()]
    if not members:
        raise SystemExit('ERROR: El ZIP está vacío.')
    print(f'ZIP encontrado: {archive.name} ({len(members)} archivos)')

    # Recognize a standard project results archive. Do not extract unrelated files.
    anchors = {'metrics','models','figures','videos'}
    special = {'last_invocation.json'}
    mapped = []
    skipped = []
    for m in members:
        name = m.filename.replace('\\', '/')
        parts = name.split('/')
        if ('..' in parts or name.startswith('/') or not all(parts)):
            skipped.append((name, 'ruta insegura'))
            continue
        # 'results/full/...' / 'full/...' / 'metrics/...'; strip package prefixes.
        start = next((i for i, p in enumerate(parts) if p in anchors), None)
        if start is not None:
            rel = Path(*parts[start:])
        elif parts[-1] in special:
            rel = Path(parts[-1])
        else:
            skipped.append((name, 'fuera de estructura de resultados'))
            continue
        if any(p.startswith('.') for p in rel.parts):
            skipped.append((name, 'archivo oculto'))
            continue
        mapped.append((m, rel))

    if not mapped:
        print('NO SE EXTRAJO NADA: no se reconocieron metrics/, models/, figures/ o videos/.')
        print('Ejemplos de rutas internas (primeros 20):')
        for m in members[:20]:
            print(' -', m.filename)
        raise SystemExit('Se necesita revisar manualmente la estructura del ZIP.')

    # Refuse contradictory collisions inside the archive itself.
    duplicates = Counter(str(rel) for _, rel in mapped)
    repeated = [rel for rel, count in duplicates.items() if count > 1]
    if repeated:
        print('ERROR: existen rutas de salida duplicadas dentro del ZIP:')
        for rel in repeated[:15]:
            print(' -', rel)
        raise SystemExit(1)

    extracted = already = conflicts = 0
    target.mkdir(parents=True, exist_ok=True)
    for m, rel in mapped:
        dest = target / rel
        if dest.exists():
            if check_same(dest, m, zf):
                already += 1
            else:
                conflicts += 1
                print(f'CONFLICTO (no sobrescrito): {rel}')
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + '.partial-rlaudit')
        try:
            with zf.open(m) as src, tmp.open('xb') as output:
                import shutil
                shutil.copyfileobj(src, output, 1024*1024)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        extracted += 1

    print(f'Extraídos: {extracted} | Ya existentes e idénticos: {already} | Conflictos: {conflicts}')
    print(f'Ignorados fuera del conjunto de resultados: {len(skipped)}')
    print(f'Carpeta de salida: {target}')
    summaries = list((target / 'metrics').glob('*_summary.json'))
    print(f'Resúmenes de experimentos detectados: {len(summaries)} / 33 esperados')
    if len(summaries) != 33:
        print('AVISO: falta alguna ejecución o la estructura difiere de la esperada.')
    if conflicts:
        print('AVISO: existen archivos con el mismo nombre pero contenido distinto; no fueron sobrescritos.')
PY

printf '\n=== Inventario y archivos demasiado grandes ===\n'
python3 - "$ROOT" "$TARGET" <<'PY'
import subprocess, sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
files = [p for p in target.rglob('*') if p.is_file()]
size = sum(p.stat().st_size for p in files)
print(f'Total en results/full/: {len(files)} archivos, {size/1024**2:.1f} MiB')
large = [p for p in files if p.stat().st_size >= 50*1024**2]
for p in sorted(large, key=lambda p: p.stat().st_size, reverse=True):
    print(f'GRANDE: {p.stat().st_size/1024**2:.1f} MiB  {p.relative_to(root)}')

# Stage only files below 50MiB. Do not alter staging of unrelated paths.
accepted = [str(p.relative_to(root)) for p in files if p.stat().st_size < 50*1024**2]
accepted.append(str((root/'rl_dqn_project/.gitignore').relative_to(root)))
accepted.append(str((root/'.gitignore').relative_to(root)))
accepted = [p for p in accepted if (root/p).is_file()]

# git add in bounded batches, so large collections do not exceed command length.
for i in range(0, len(accepted), 100):
    subprocess.run(['git','-C',str(root),'add','--',*accepted[i:i+100]],check=True)
print(f'Preparados para Git: {len(accepted)-2 if (root/".gitignore").exists() else len(accepted)-1} elementos como máximo (ver status).')
if large:
    print('AVISO: no se añadieron los archivos >=50 MiB; revisarlos manualmente antes de publicar.')
PY

printf '\n=== Estado final de Git (sin commit ni push) ===\n'
git -C "$ROOT" status --short
printf '\n=== Tamaño de cambios preparados ===\n'
git -C "$ROOT" diff --cached --stat | tail -n 8
printf '\nNO se ejecutó git commit ni git push.\n'
printf 'Cuando verifiques los archivos, publica manualmente con:\n'
printf '  git commit -m "Add complete RL experimental results"\n'
printf '  git push origin main\n'
