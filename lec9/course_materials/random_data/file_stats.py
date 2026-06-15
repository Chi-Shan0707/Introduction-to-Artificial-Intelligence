import os

def get_file_info(path):
    stat = os.stat(path)
    return {
        "name": os.path.basename(path),
        "size_bytes": stat.st_size,
        "size_kb": round(stat.st_size / 1024, 2),
        "extension": os.path.splitext(path)[1] or "(none)",
    }

def scan_directory(directory):
    files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            files.append(get_file_info(full_path))
    return files

def summarize(files):
    total_size = sum(f["size_bytes"] for f in files)
    by_ext = {}
    for f in files:
        ext = f["extension"]
        by_ext[ext] = by_ext.get(ext, 0) + 1
    return {
        "total_files": len(files),
        "total_size_kb": round(total_size / 1024, 2),
        "by_extension": by_ext,
    }

if __name__ == "__main__":
    target = os.path.dirname(os.path.abspath(__file__))
    files = scan_directory(target)
    summary = summarize(files)
    print(f"Directory: {target}")
    print(f"Files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_kb']} KB")
    print(f"By extension: {summary['by_extension']}")
