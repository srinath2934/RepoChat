"""
GitHubLoader – Production-grade loader for GitHub RAG Chatbot

Features:
- Fetches complete file tree (recursive)
- Filters by extension and size
- Downloads and decodes file content
- Creates metadata-only placeholders for binary files
- Loads commit history for temporal context
- Handles main/master branch fallback
- Rate limit detection and error handling
"""

import os
import base64
import logging
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from langchain_core.documents import Document

# Logging setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s – %(message)s", 
        "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class GitHubLoader:
    """Loads GitHub repository files and commits as LangChain Documents."""
    
    def __init__(
        self,
        repo: str,
        branch: str = "main",
        access_token: Optional[str] = None,
        file_extensions: Tuple[str, ...] = (
            ".py", ".js", ".md", ".yaml", ".yml", 
            ".json", ".toml", ".sql", ".dockerfile", ".css", ".ipynb"
        ),
        max_file_size_bytes: int = 1_048_576,  # 1 MiB
        github_api_url: str = "https://api.github.com",
        commit_history_limit: int = 50,
    ):
        """
        Initialize GitHub loader.
        
        Args:
            repo: Either 'owner/repo' or full URL
            branch: Branch name (fallback to master if not found)
            access_token: GitHub Personal Access Token
            file_extensions: Extensions to fully load
            max_file_size_bytes: Skip files larger than this
            github_api_url: API base URL (for GitHub Enterprise)
            commit_history_limit: Number of recent commits to load
        """
        # Normalize repo to owner/repo format
        if repo.startswith("http"):
            repo = repo.replace("https://", "").replace("http://", "")
            repo = repo.rstrip("/")
            if repo.endswith(".git"):
                repo = repo[:-4]
            repo = repo.split("/", 1)[-1]
        
        self.repo = repo
        self.branch = branch
        self.access_token = access_token
        self.file_extensions = tuple(ext.lower() for ext in file_extensions)
        self.max_file_size = max_file_size_bytes
        self.github_api_url = github_api_url.rstrip("/")
        self.commit_history_limit = commit_history_limit
        
        # Setup headers
        self.headers: Dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if access_token:
            self.headers["Authorization"] = f"token {access_token}"
    
    def _make_request_with_retry(self, url: str, max_retries: int = 3) -> requests.Response:
        """
        Make HTTP request with retry logic for connection errors
        
        Args:
            url: URL to request
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response object
        """
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=self.headers, timeout=60)
                
                # Check for rate limiting
                if resp.status_code == 403 and "X-RateLimit-Remaining" in resp.headers:
                    remaining = resp.headers.get("X-RateLimit-Remaining", "0")
                    if remaining == "0":
                        reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
                        wait_seconds = max(reset_time - int(time.time()), 60)
                        logger.warning(f"⚠️ Rate limit hit. Waiting {wait_seconds}s before retry...")
                        time.sleep(wait_seconds)
                        continue
                
                return resp
                
            except (requests.ConnectionError, requests.Timeout, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"⚠️ Connection error ({type(e).__name__}), retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Failed after {max_retries} attempts: {e}")
                    raise
        
        raise RuntimeError("Request failed after all retries")

    def load(self, progress_callback: Optional[Callable[[int, int], None]] = None, load_commits: bool = False) -> List[Document]:
        """
        Load repository files and optionally commits.
        
        Args:
            progress_callback: Optional callback function(current, total) for progress updates
            load_commits: Whether to load commit history (default: False for speed)
        
        Returns:
            List of LangChain Document objects
        """
        logger.info("🚀 Starting ingestion for %s (branch=%s)", self.repo, self.branch)
        
        # Step 1: Resolve branch (main -> master fallback)
        branch = self._resolve_branch()
        
        # Step 2: Get file tree
        tree_items = self._fetch_tree(branch)
        
        # Step 3: Filter files to load
        files_to_load = []
        placeholders = []
        
        for item in tree_items:
            if item["type"] != "blob":
                continue
            
            path = item["path"]
            size = item.get("size", 0)
            
            if self._should_load_file(path, size):
                files_to_load.append(item)
            else:
                # Binary/large file - create placeholder
                placeholder = Document(
                    page_content=f"[BINARY OR SKIPPED FILE] {os.path.basename(path)}",
                    metadata=self._build_metadata(item, placeholder=True),
                )
                placeholders.append(placeholder)
        
        total_files = len(files_to_load)
        logger.info(f"📁 Found {total_files} files to process")
        
        # Step 4: Load files in parallel
        documents: List[Document] = []
        files_processed = 0
        
        def load_single_file(item):
            """Helper function to load a single file"""
            try:
                doc = self._load_file_content(item)
                time.sleep(0.01)  # Reduced from 0.1s to 0.01s
                return doc
            except Exception as e:
                logger.warning(f"Failed to load {item.get('path', 'unknown')}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all file loading tasks
            future_to_item = {executor.submit(load_single_file, item): item for item in files_to_load}
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_item):
                doc = future.result()
                if doc:
                    documents.append(doc)
                    files_processed += 1
                    
                    # Progress callback for UI updates
                    if progress_callback:
                        progress_callback(files_processed, total_files)
                    
                    # Progress logging
                    if files_processed % 10 == 0:
                        logger.info(f"📄 Processed {files_processed}/{total_files} files...")
        
        # Add placeholders
        documents.extend(placeholders)
        
        # Step 5: Optionally load commit history
        if load_commits:
            logger.info("📜 Loading commit history...")
            documents.extend(self._load_commit_history())
        else:
            logger.info("⏭️  Skipping commit history (load_commits=False)")
        
        logger.info("✅ Ingestion complete – %d documents produced", len(documents))
        return documents
    
    def _resolve_branch(self) -> str:
        """Try requested branch, fallback to master."""
        for candidate in (self.branch, "master"):
            url = f"{self.github_api_url}/repos/{self.repo}/git/trees/{candidate}?recursive=1"
            resp = self._make_request_with_retry(url)
            if resp.status_code == 200:
                logger.info("✅ Using branch '%s'", candidate)
                return candidate
            if resp.status_code == 404:
                logger.warning("Branch '%s' not found – trying fallback", candidate)
                continue
            self._handle_http_error(resp, f"branch resolution ({candidate})")
        raise RuntimeError(f"Unable to resolve branch for repo '{self.repo}'.")
    
    def _fetch_tree(self, branch: str) -> List[Dict[str, Any]]:
        """Get complete file tree."""
        tree_url = f"{self.github_api_url}/repos/{self.repo}/git/trees/{branch}?recursive=1"
        resp = self._make_request_with_retry(tree_url)
        if resp.status_code != 200:
            self._handle_http_error(resp, "fetching repository tree")
        tree = resp.json().get("tree", [])
        logger.info("📂 Retrieved %d items from repository tree", len(tree))
        return tree
    
    def _should_load_file(self, path: str, size: int) -> bool:
        """Check if file should be fully loaded."""
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.file_extensions:
            return False
        if size > self.max_file_size:
            logger.warning(
                "Skipping large file %s (size=%d > %d)", 
                path, size, self.max_file_size
            )
            return False
        return True
    
    def _load_file_content(self, item: Dict[str, Any]) -> Optional[Document]:
        """Download and decode a single file."""
        path = item["path"]
        content_url = (
            f"{self.github_api_url}/repos/{self.repo}/contents/{path}?ref={self.branch}"
        )
        try:
            resp = self._make_request_with_retry(content_url)
            if resp.status_code != 200:
                # SOFTENED: 404s in large repos are common during recursive crawls
                if resp.status_code == 404:
                    logger.debug(f"Skipping {path}: File not found (404)")
                else:
                    logger.warning(f"Failed to fetch {path} (Status {resp.status_code})")
                return None
            
            data = resp.json()
            if data.get("encoding") == "base64":
                raw = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
            else:
                raw = data.get("content", "")
            
            metadata = self._build_metadata(item, placeholder=False)
            return Document(page_content=raw, metadata=metadata)
        except Exception as e:
            logger.error("Exception loading file %s: %s", path, str(e))
            return None
    
    def _build_metadata(self, item: Dict[str, Any], placeholder: bool) -> Dict[str, Any]:
        """Create metadata dictionary."""
        return {
            "source": item["path"],
            "repo": self.repo,
            "branch": self.branch,
            "sha": item.get("sha"),
            "size": item.get("size", 0),
            "type": "placeholder" if placeholder else "code",
            "url": f"https://github.com/{self.repo}/blob/{self.branch}/{item['path']}",
        }
    
    def _load_commit_history(self) -> List[Document]:
        """Load recent commits as Documents."""
        commits_url = (
            f"{self.github_api_url}/repos/{self.repo}/commits?"
            f"per_page={self.commit_history_limit}"
        )
        try:
            resp = self._make_request_with_retry(commits_url)
            if resp.status_code != 200:
                logger.warning("Could not fetch commit history – %s", resp.text)
                return []
            
            commits = resp.json()
            docs: List[Document] = []
            for c in commits:
                sha = c.get("sha")
                author = c.get("commit", {}).get("author", {})
                message = c.get("commit", {}).get("message", "").strip()
                date = author.get("date")
                name = author.get("name")
                email = author.get("email")
                
                content = (
                    f"[COMMIT] SHA: {sha}\n"
                    f"Author: {name} <{email}>\n"
                    f"Date: {date}\n"
                    f"Message: {message}"
                )
                metadata = {
                    "source": "git_history",
                    "repo": self.repo,
                    "branch": self.branch,
                    "sha": sha,
                    "type": "commit",
                    "url": f"https://github.com/{self.repo}/commit/{sha}",
                }
                docs.append(Document(page_content=content, metadata=metadata))
            
            logger.info("🕒 Loaded %d recent commits", len(docs))
            return docs
        except Exception as e:
            logger.warning("Exception loading commit history: %s", str(e))
            return []
    
    @staticmethod
    def _handle_http_error(resp: requests.Response, context: str = ""):
        """Handle HTTP errors with helpful messages."""
        if resp.status_code == 403 and "X-RateLimit-Remaining" in resp.headers:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            raise RuntimeError(
                f"GitHub API rate limit exceeded (remaining={remaining}). "
                f"Reset epoch: {reset}. Context: {context}"
            )
        raise RuntimeError(
            f"GitHub request failed (status {resp.status_code}): "
            f"{resp.text}. Context: {context}"
        )
