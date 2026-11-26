# Arşiv (1 dk): koşu kayıtları, ilk denemeler (turlar ve listeler), liste özetleri, tablolar, öz sınama, kod ve
# manifest tek ZIP'te. Bunu gönderin. (lists.npz dosyaları büyük olduğu için ZIP'e girmez; Drive'da kalır.)
import zipfile


def make_zip(name, include):
    """ZIP önce yerel diskte kurulur, sonra sonuç klasörüne (Drive) kopyalanır."""
    local = WORK_ROOT / name
    count = 0
    with zipfile.ZipFile(local, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            relative = path.relative_to(OUTPUT_ROOT)
            if not path.is_file() or path.suffix in {".zip", ".tmp"} or path.name.endswith(".tmp.npz"):
                continue
            if include(relative):
                z.write(path, str(relative))
                count += 1
    with zipfile.ZipFile(local) as z:
        assert z.testzip() is None
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    shutil.copy2(local, OUTPUT_ROOT / name)
    print(f"{name}: {count} dosya, {local.stat().st_size / 1e6:.1f} MB, SHA-256 {digest}")
    return local


result_zip = make_zip(EXPERIMENT_NAME + "_results.zip",
                      lambda rel: rel.parts[0] in {"runs", "summary", "code", "errors", "selfcheck"}
                      or (rel.parts[0] == "instances" and rel.name in {"first.npz", "lists.json"})
                      or rel.name == "manifest.json")
if SAVE_TO_DRIVE:
    from google.colab import files
    files.download(str(result_zip))      # tarayıcıya iner
print("ZIP şu klasörde:", OUTPUT_ROOT)
