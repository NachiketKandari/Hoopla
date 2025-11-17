import pathlib
import sys

def copy_py_files_to_txt(source_dir: pathlib.Path, target_dir: pathlib.Path):
    
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {target_dir}: {e}", file=sys.stderr)
        return

    print(f"\n--- Processing files in: {source_dir} ---")
    
    py_files = list(source_dir.glob("*.py"))
    
    if not py_files:
        print("  No .py files found.")
        return

    for py_file_path in py_files:
        try:
            target_file_name = py_file_path.stem + ".txt"
            target_file_path = target_dir / target_file_name
            
            content = py_file_path.read_text(encoding="utf-8")
            
            target_file_path.write_text(content, encoding="utf-8")
            
            print(f"  Copied: {py_file_path.name}  ->  {target_file_path.name}")
            
        except Exception as e:
            print(f"  Error processing {py_file_path.name}: {e}", file=sys.stderr)

    print(f"--- Finished processing {len(py_files)} files. ---")

def main():
    base_dir = pathlib.Path.cwd()
    
    cli_source = base_dir / "cli"
    lib_source = base_dir / "cli" / "lib"
    
    cli_target = base_dir / "clicpy"
    lib_target = base_dir / "libcpy"
    
    print("Starting copy process...")
    
    copy_py_files_to_txt(cli_source, cli_target)
    copy_py_files_to_txt(lib_source, lib_target)
    
    print("\nAll tasks complete.")

if __name__ == "__main__":
    main()