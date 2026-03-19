"""
Language Detector - Identify programming languages
Purpose: Route documents to appropriate processors
"""
from typing import Dict, Optional
import os

class LanguageDetector:
    """
    Detect programming language from file extension
    """
    
    LANGUAGE_MAP = {
        # Python
        '.py': 'python',
        '.pyw': 'python',
        
        # JavaScript/TypeScript
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        
        # Java
        '.java': 'java',
        
        # C/C++
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c_header',
        '.hpp': 'cpp_header',
        
        # Go
        '.go': 'go',
        
        # Rust
        '.rs': 'rust',
        
        # Ruby
        '.rb': 'ruby',
        
        # PHP
        '.php': 'php',
        
        # C#
        '.cs': 'csharp',
        
        # Swift
        '.swift': 'swift',
        
        # Kotlin
        '.kt': 'kotlin',
        
        # Scala
        '.scala': 'scala',
        
        # Documentation
        '.md': 'markdown',
        '.rst': 'restructuredtext',
        
        # Config
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.xml': 'xml',
        
        # Shell
        '.sh': 'shell',
        '.bash': 'bash',
        
        # SQL
        '.sql': 'sql',
    }
    
    @classmethod
    def detect(cls, file_path: str) -> Optional[str]:
        """
        Detect language from file path
        
        Args:
            file_path: Path to file
            
        Returns:
            Language name or None
        """
        ext = os.path.splitext(file_path)[1].lower()
        return cls.LANGUAGE_MAP.get(ext)
    
    @classmethod
    def is_code(cls, file_path: str) -> bool:
        """Check if file is code (not config/doc)"""
        language = cls.detect(file_path)
        non_code = {'markdown', 'restructuredtext', 'json', 'yaml', 'toml', 'xml'}
        return language is not None and language not in non_code
    
    @classmethod
    def get_processor_type(cls, file_path: str) -> str:
        """
        Get the type of processor needed
        
        Returns:
            'ast', 'regex', or 'generic'
        """
        language = cls.detect(file_path)
        
        # Languages with AST support
        ast_languages = {'python', 'javascript', 'typescript', 'java', 'go', 'rust'}
        
        if language in ast_languages:
            return 'ast'
        elif language:
            return 'regex'
        else:
            return 'generic'
    
    @classmethod
    def analyze_repository(cls, file_paths: list) -> Dict[str, int]:
        """
        Analyze language distribution in a repository
        
        Args:
            file_paths: List of file paths from the repo
            
        Returns:
            Dictionary of language counts: {'python': 15, 'javascript': 8, ...}
            
        Example:
            >>> files = ['app.py', 'utils.py', 'index.js', 'README.md']
            >>> LanguageDetector.analyze_repository(files)
            {'python': 2, 'javascript': 1, 'markdown': 1}
        """
        language_counts = {}
        
        for path in file_paths:
            lang = cls.detect(path)
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        return language_counts
    
    @classmethod
    def get_dominant_language(cls, file_paths: list) -> Optional[str]:
        """
        Find the most common language in a repository
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Most common language or None
            
        Example:
            >>> files = ['a.py', 'b.py', 'c.py', 'd.js']
            >>> LanguageDetector.get_dominant_language(files)
            'python'
        """
        counts = cls.analyze_repository(file_paths)
        
        if not counts:
            return None
        
        # Return language with highest count
        return max(counts.items(), key=lambda x: x[1])[0]
    
    @classmethod
    def filter_by_language(cls, file_paths: list, language: str) -> list:
        """
        Filter files by specific language
        
        Args:
            file_paths: List of file paths
            language: Language to filter for
            
        Returns:
            List of files matching that language
            
        Example:
            >>> files = ['app.py', 'utils.py', 'index.js']
            >>> LanguageDetector.filter_by_language(files, 'python')
            ['app.py', 'utils.py']
        """
        return [path for path in file_paths if cls.detect(path) == language]
    
    @classmethod
    def get_code_files_only(cls, file_paths: list) -> list:
        """
        Filter out non-code files (configs, docs)
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Only actual code files
            
        Example:
            >>> files = ['app.py', 'README.md', 'config.json']
            >>> LanguageDetector.get_code_files_only(files)
            ['app.py']
        """
        return [path for path in file_paths if cls.is_code(path)]
