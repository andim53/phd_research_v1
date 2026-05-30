import os
import re
from typing import List, Optional

def get_struct_paths(
    base_dir: str,
    indices: Optional[List[int]] = None,
    prefix: str = "struct_",
    ext: str = ".xsf"
) -> List[str]:
    """
    Collect .xsf structure files in a directory and optionally filter them
    by numeric indices in the filename.

    Filenames should follow the pattern: <prefix><index><ext>
    Example:
        struct_0.xsf  → index = 0
        struct_3.xsf  → index = 3
        struct_42.xsf → index = 42

    Args:
        base_dir: Directory containing the .xsf files.
        indices: List of indices to include. If None, all files are returned.
        prefix: Filename prefix before numeric index.
        ext: File extension to match.

    Returns:
        List of full paths to matching .xsf files, sorted by numeric index.
    """
    struct_paths: List[str] = []
    pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(ext)}$")

    # Collect matching files
    for fname in os.listdir(base_dir):
        if fname.endswith(ext):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                if indices is None or idx in indices:
                    struct_paths.append(os.path.join(base_dir, fname))

    # Sort paths by numeric index
    struct_paths.sort(key=lambda x: int(re.search(rf"{re.escape(prefix)}(\d+){re.escape(ext)}$", x).group(1)))

    return struct_paths
