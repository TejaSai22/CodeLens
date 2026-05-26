"""
Repository loader module for cloning GitHub repos and traversing local directories.
"""

import os
import tempfile
import shutil
from typing import List, Dict, Optional
from pathlib import Path
import git


class RepoLoader:
    """Loads repositories from GitHub URLs or local paths."""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java',
        '.go', '.rs', '.c', '.cpp', '.h'
    }

    # Directories to skip
    SKIP_DIRS = {
        'node_modules', '.git', '__pycache__', 'pycache',
        'venv', '.venv', 'dist', 'build'
    }

    def __init__(self, repo_path: Optional[str] = None, repo_url: Optional[str] = None):
        """
        Initialize RepoLoader with either a local path or GitHub URL.

        Args:
            repo_path: Path to local repository
            repo_url: GitHub repository URL
        """
        if not repo_path and not repo_url:
            raise ValueError("Either repo_path or repo_url must be provided")

        if repo_path and repo_url:
            raise ValueError("Provide either repo_path or repo_url, not both")

        self.repo_path = repo_path
        self.repo_url = repo_url
        self.temp_dir = None
        self._cloned_path = None

    def _clone_repo(self, url: str) -> str:
        """
        Clone GitHub repository to temporary directory.

        Args:
            url: GitHub repository URL

        Returns:
            Path to cloned repository
        """
        self.temp_dir = tempfile.mkdtemp(prefix="codelens_")

        try:
            # Perform shallow clone with depth=1
            git.Repo.clone_from(url, self.temp_dir, depth=1)
            return self.temp_dir
        except Exception as e:
            # Clean up on failure
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            raise Exception(f"Failed to clone repository: {str(e)}")

    def _detect_language(self, file_path: str) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to file

        Returns:
            Language name
        """
        ext = Path(file_path).suffix.lower()

        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c'
        }

        return language_map.get(ext, 'unknown')

    def _should_skip_dir(self, dir_name: str) -> bool:
        """
        Check if directory should be skipped.

        Args:
            dir_name: Directory name

        Returns:
            True if directory should be skipped
        """
        return dir_name in self.SKIP_DIRS

    def _should_include_file(self, file_path: str) -> bool:
        """
        Check if file should be included based on extension.

        Args:
            file_path: Path to file

        Returns:
            True if file should be included
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def get_files(self) -> List[Dict[str, str]]:
        """
        Traverse repository and return list of file data.

        Returns:
            List of dictionaries with keys: path, content, language, size
        """
        # Determine root path
        if self.repo_url:
            self._cloned_path = self._clone_repo(self.repo_url)
            root_path = self._cloned_path
        else:
            root_path = self.repo_path

        files = []
        root_path_obj = Path(root_path)

        # Walk through directory tree
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Remove skip directories from traversal
            dirnames[:] = [d for d in dirnames if not self._should_skip_dir(d)]

            for filename in filenames:
                file_path = os.path.join(dirpath, filename)

                # Check if file should be included
                if not self._should_include_file(file_path):
                    continue

                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
                        continue
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
                    continue

                # Calculate relative path
                relative_path = str(Path(file_path).relative_to(root_path_obj))

                # Detect language
                language = self._detect_language(file_path)

                files.append({
                    'path': relative_path,
                    'content': content,
                    'language': language,
                    'size': len(content)
                })

        return files

    def cleanup(self):
        """Clean up temporary directory if repository was cloned."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
