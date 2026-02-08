#!/usr/bin/env python3
"""
Collect real-world prompts from public sources for compression benchmarking.

This script collects prompts from various public sources, categorizes them,
and saves them with proper attribution.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Optional imports for GitHub collection
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress bar replacement
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Base directory for prompts
PROMPTS_DIR = Path(__file__).parent / "prompts"
METADATA_FILE = PROMPTS_DIR / "METADATA.md"

# Categories
CATEGORIES = {
    "instructions": "Pure instruction prompts",
    "structured": "JSON/HTML/table prompts",
    "conversations": "Multi-turn conversation prompts",
    "technical": "Technical documentation prompts",
    "mixed": "Mixed content prompts",
}

# Sources to collect from (public repositories and datasets)
SOURCES = [
    {
        "name": "Awesome Prompts",
        "url": "https://github.com/f/awesome-chatgpt-prompts",
        "type": "github",
        "description": "Collection of ChatGPT prompts",
        "path": "",  # Root directory
    },
    {
        "name": "Prompt Engineering Guide",
        "url": "https://github.com/dair-ai/Prompt-Engineering-Guide",
        "type": "github",
        "description": "Comprehensive prompt engineering examples",
        "path": "contents",
    },
    {
        "name": "LangChain Examples",
        "url": "https://github.com/langchain-ai/langchain",
        "type": "github",
        "description": "LangChain prompt examples",
        "path": "libs/langchain/langchain/prompts",
    },
    {
        "name": "OpenAI Cookbook",
        "url": "https://github.com/openai/openai-cookbook",
        "type": "github",
        "description": "OpenAI examples and prompts",
        "path": "examples",
    },
]


def create_directory_structure():
    """Create the directory structure for prompts."""
    for category in CATEGORIES.keys():
        (PROMPTS_DIR / category).mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {PROMPTS_DIR}")


def download_file(url: str, dest: Path, retry_attempt: int = 0) -> bool:
    """Download a file from URL with exponential backoff on rate limits."""
    if not HAS_REQUESTS:
        print("Warning: requests module not available. Skipping download.")
        return False
    
    max_retries = 2
    try:
        response = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (compatible; PromptCollector/1.0)"
        })
        
        # Check for rate limiting
        if response.status_code == 429:
            if retry_attempt >= max_retries:
                print(f"  ✗ Max retries reached for {url}. Skipping...")
                return False
            
            # Get retry-after header
            retry_after = None
            retry_header = response.headers.get('Retry-After')
            if retry_header:
                try:
                    retry_after = int(retry_header)
                except ValueError:
                    pass
            
            # Exponential backoff if no header
            if retry_after is None:
                retry_after = min(10 * (2 ** retry_attempt), 60)  # Cap at 60 seconds (1 minute)
            
            # Always cap retry_after at 60 seconds maximum
            retry_after = min(retry_after, 60)
            
            print(f"  ⚠️  Rate limit on download. Waiting {retry_after}s (attempt {retry_attempt + 1}/{max_retries})...")
            time.sleep(retry_after)
            return download_file(url, dest, retry_attempt + 1)
        
        response.raise_for_status()
        dest.write_text(response.text, encoding='utf-8')
        return True
    except requests.exceptions.RequestException as e:
        # Check for rate limiting in exception
        if (hasattr(e, 'response') and e.response is not None and e.response.status_code == 429):
            if retry_attempt >= max_retries:
                print(f"  ✗ Max retries reached for {url}. Skipping...")
                return False
            
            retry_after = None
            if hasattr(e, 'response') and e.response is not None:
                retry_header = e.response.headers.get('Retry-After')
                if retry_header:
                    try:
                        retry_after = int(retry_header)
                    except ValueError:
                        pass
            
            # Exponential backoff if no header
            if retry_after is None:
                retry_after = min(10 * (2 ** retry_attempt), 60)  # Cap at 60 seconds (1 minute)
            
            # Always cap retry_after at 60 seconds maximum
            retry_after = min(retry_after, 60)
            
            print(f"  ⚠️  Rate limit on download. Waiting {retry_after}s (attempt {retry_attempt + 1}/{max_retries})...")
            time.sleep(retry_after)
            return download_file(url, dest, retry_attempt + 1)
        else:
            print(f"  ⚠️  Error downloading {url}: {e}")
            return False
    except Exception as e:
        print(f"  ⚠️  Unexpected error downloading {url}: {e}")
        return False


def extract_prompts_from_markdown(content: str) -> List[str]:
    """Extract individual prompts from markdown content."""
    import re
    prompts = []
    
    # Look for code blocks that might contain prompts
    code_blocks = re.findall(r'```(?:python|text|markdown|json|yaml)?\n(.*?)```', content, re.DOTALL)
    for block in code_blocks:
        block = block.strip()
        if len(block) > 50 and len(block) < 5000:  # Reasonable prompt size
            prompts.append(block)
    
    # Look for quoted sections
    quoted = re.findall(r'"(.*?)"', content, re.DOTALL)
    for quote in quoted:
        quote = quote.strip()
        if len(quote) > 100 and len(quote) < 3000:
            prompts.append(quote)
    
    return prompts


def collect_from_github_repo(repo_url: str, category: str, subpath: str = "", max_prompts: int = 50) -> Tuple[List[Dict], bool]:
    """Collect prompts from a GitHub repository with recursive traversal."""
    if not HAS_REQUESTS:
        return [], False
    
    from urllib.parse import urlparse
    
    # Extract owner and repo from URL
    parsed = urlparse(repo_url)
    parts = parsed.path.strip('/').split('/')
    if len(parts) < 2:
        return []
    
    owner, repo = parts[0], parts[1]
    base_path = subpath.strip('/') if subpath else ""
    
    prompts = []
    collected_files = set()  # Track collected files to avoid duplicates
    retry_count = {}  # Track retry attempts per path for exponential backoff
    failed_paths = set()  # Track paths that have failed after max retries to avoid retrying
    
    def traverse_directory(path: str, depth: int = 0, retry_attempt: int = 0):
        """Recursively traverse repository directories."""
        if depth > 3 or len(prompts) >= max_prompts:  # Limit recursion depth and total prompts
            return
        
        # Skip paths that have already failed after max retries
        if path in failed_paths:
            return
        
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}" if path else f"https://api.github.com/repos/{owner}/{repo}/contents"
            response = requests.get(url, timeout=30, headers={
                "User-Agent": "Mozilla/5.0 (compatible; PromptCollector/1.0)",
                "Accept": "application/vnd.github.v3+json"
            })
            
            if response.status_code == 404:
                return
            
            response.raise_for_status()
            contents = response.json()
            
            # Handle case where API returns a single file instead of array
            if not isinstance(contents, list):
                contents = [contents]
            
            for item in contents:
                if len(prompts) >= max_prompts:
                    return
                
                if item['type'] == 'file':
                    # Look for prompt files
                    name_lower = item['name'].lower()
                    if (name_lower.endswith('.md') or 
                        name_lower.endswith('.txt') or 
                        name_lower.endswith('.prompt') or
                        'prompt' in name_lower or
                        'example' in name_lower):
                        
                        # Skip very large files
                        if item.get('size', 0) > 100000:  # 100KB limit
                            continue
                        
                        file_key = f"{owner}_{repo}_{item['path']}"
                        if file_key in collected_files:
                            continue
                        collected_files.add(file_key)
                        
                        # Download file content
                        file_url = item.get('download_url')
                        if not file_url:
                            continue
                        
                        file_path = PROMPTS_DIR / category / f"{owner}_{repo}_{item['name'].replace('/', '_')}"
                        
                        if download_file(file_url, file_path):
                            # Try to extract multiple prompts from markdown files
                            if name_lower.endswith('.md'):
                                content = file_path.read_text(encoding='utf-8')
                                extracted = extract_prompts_from_markdown(content)
                                
                                # If we extracted multiple prompts, save them separately
                                if len(extracted) > 1:
                                    for idx, prompt_text in enumerate(extracted[:5]):  # Limit to 5 per file
                                        prompt_file = PROMPTS_DIR / category / f"{owner}_{repo}_{item['name'].replace('/', '_')}_{idx}.txt"
                                        prompt_file.write_text(prompt_text, encoding='utf-8')
                                        prompts.append({
                                            "file": prompt_file.name,
                                            "source": f"{repo_url}/blob/main/{item['path']}",
                                            "category": category,
                                            "license": "Check repository license",
                                        })
                                    file_path.unlink()  # Remove original file
                                    continue
                            
                            prompts.append({
                                "file": file_path.name,
                                "source": f"{repo_url}/blob/main/{item['path']}",
                                "category": category,
                                "license": "Check repository license",
                            })
                            time.sleep(0.3)  # Rate limiting
                
                elif item['type'] == 'dir' and depth < 2:
                    # Skip certain directories
                    dir_name = item['name'].lower()
                    if dir_name in ['.git', 'node_modules', '__pycache__', '.github', 'tests', 'test']:
                        continue
                    
                    # Skip if this path has already failed
                    if item['path'] in failed_paths:
                        continue
                    
                    # Recursively traverse subdirectories
                    traverse_directory(item['path'], depth + 1)
                    time.sleep(0.2)  # Rate limiting between directories
                    
        except requests.exceptions.RequestException as e:
            # Check for rate limiting (429 status code or rate limit in message)
            is_rate_limit = (
                hasattr(e, 'response') and 
                e.response is not None and 
                e.response.status_code == 429
            ) or "rate limit" in str(e).lower()
            
            if is_rate_limit:
                # Exponential backoff: base delay * 2^retry_attempt
                # Cap at 2 retries (max ~2 minutes wait)
                max_retries = 2
                if retry_attempt >= max_retries:
                    print(f"  ✗ Max retries ({max_retries}) reached for {path}. Skipping...")
                    failed_paths.add(path)  # Mark this path as failed
                    # If root path fails, skip entire repository
                    if depth == 0 or path == base_path:
                        print(f"  ✗ Root path failed. Skipping entire repository.")
                        return
                    return
                
                # Get retry-after header if available (most accurate)
                retry_after = None
                if hasattr(e, 'response') and e.response is not None:
                    retry_header = e.response.headers.get('Retry-After')
                    if retry_header:
                        try:
                            retry_after = int(retry_header)
                        except ValueError:
                            pass
                    
                    # Check X-RateLimit-Reset header for exact reset time
                    if retry_after is None:
                        reset_time = e.response.headers.get('X-RateLimit-Reset')
                        if reset_time:
                            try:
                                reset_timestamp = int(reset_time)
                                current_time = int(time.time())
                                retry_after = max(1, min(reset_timestamp - current_time, 60))  # Cap at 60 seconds
                            except (ValueError, TypeError):
                                pass
                
                # If no header info, use exponential backoff
                if retry_after is None:
                    base_delay = 10  # Start with 10 seconds
                    retry_after = min(base_delay * (2 ** retry_attempt), 60)  # Cap at 60 seconds (1 minute)
                
                # Always cap retry_after at 60 seconds maximum
                retry_after = min(retry_after, 60)
                
                print(f"  ⚠️  Rate limit hit (attempt {retry_attempt + 1}/{max_retries}). Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                
                # Retry the request with incremented retry count
                return traverse_directory(path, depth, retry_attempt + 1)
            else:
                print(f"  ⚠️  Error accessing {path}: {e}")
        except Exception as e:
            print(f"  ⚠️  Unexpected error in {path}: {e}")
    
    print(f"  Traversing repository: {repo_url} (path: {base_path or 'root'})")
    traverse_directory(base_path)
    
    # Check if root path failed (indicates rate limit failure)
    root_failed = base_path in failed_paths or (not base_path and "" in failed_paths)
    
    # If root path failed, return early to avoid further attempts
    if root_failed:
        print(f"  ✗ Repository collection failed due to rate limits. Skipping...")
        return prompts, True  # Return (prompts, failed_flag)
    
    return prompts, False  # Return (prompts, failed_flag)


def create_sample_prompts():
    """Create sample real-world-like prompts for testing."""
    samples = {
        "instructions": [
            {
                "name": "data_extraction.txt",
                "content": """Extract the following information from the document:
- Full name of the person
- Date of birth
- Email address
- Phone number

Response Format:
Name::DateOfBirth::Email::Phone

If you are unable to find any field, use "N/A" for that field.
The separator is ALWAYS :: (double colon).
Do NOT output any additional text or explanation.""",
                "source": "Synthetic - Real-world pattern",
                "license": "MIT",
            },
            {
                "name": "code_review.txt",
                "content": """Review the following code and provide feedback:

1. Code Quality: Assess readability, maintainability, and best practices
2. Security: Identify potential security vulnerabilities
3. Performance: Suggest optimizations if applicable
4. Documentation: Evaluate code comments and documentation

Provide your review in the following format:
- Code Quality: [score 1-10] [comments]
- Security: [score 1-10] [comments]
- Performance: [score 1-10] [comments]
- Documentation: [score 1-10] [comments]

Be thorough but concise.""",
                "source": "Synthetic - Real-world pattern",
                "license": "MIT",
            },
        ],
        "structured": [
            {
                "name": "json_schema.txt",
                "content": """Extract data from the following document and return it as JSON.

Required JSON Schema:
{
  "title": "Document Analysis",
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "key_points": {"type": "array", "items": {"type": "string"}},
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "confidence": {"type": "number"}
        }
      }
    }
  },
  "required": ["summary", "key_points"]
}

Document:
[Document content will be provided here]""",
                "source": "Synthetic - Real-world pattern",
                "license": "MIT",
            },
        ],
        "conversations": [
            {
                "name": "multi_turn.txt",
                "content": """You are a helpful assistant for a customer support system.

System: Welcome! How can I help you today?

User: I'm having trouble with my account login.

System: I'm sorry to hear that. Can you provide your account email?

User: [User provides email]

System: I've found your account. What specific issue are you experiencing?

User: [User describes issue]

System: [Provide solution based on issue]

Please continue this conversation following the same pattern, maintaining a helpful and professional tone.""",
                "source": "Synthetic - Real-world pattern",
                "license": "MIT",
            },
        ],
        "technical": [
            {
                "name": "api_documentation.txt",
                "content": """Document the following API endpoint:

Endpoint: POST /api/v1/users
Authentication: Bearer token required
Request Body:
{
  "name": "string (required)",
  "email": "string (required, valid email format)",
  "role": "string (optional, one of: admin, user, guest)"
}

Response Codes:
- 201: User created successfully
- 400: Invalid request data
- 401: Unauthorized
- 409: Email already exists

Response Body (201):
{
  "id": "uuid",
  "name": "string",
  "email": "string",
  "role": "string",
  "created_at": "ISO 8601 timestamp"
}

Provide comprehensive documentation including:
1. Endpoint description
2. Authentication requirements
3. Request/response examples
4. Error handling
5. Rate limiting information""",
                "source": "Synthetic - Real-world pattern",
                "license": "MIT",
            },
        ],
        "mixed": [
            {
                "name": "complex_task.txt",
                "content": """You are an AI assistant helping with document processing.

Task: Analyze the following document and extract structured information.

Instructions:
1. Read the entire document carefully
2. Identify key sections (Introduction, Body, Conclusion)
3. Extract important facts and figures
4. Identify any tables or structured data
5. Note any inconsistencies or errors

Document Format:
The document may contain:
- Plain text paragraphs
- Markdown formatting
- Tables
- Code blocks
- Images (describe content)

Output Format:
Provide your analysis as a JSON object with the following structure:
{
  "sections": ["section1", "section2", ...],
  "key_facts": ["fact1", "fact2", ...],
  "tables": [{"title": "...", "data": [...]}],
  "code_blocks": [{"language": "...", "content": "..."}],
  "issues": ["issue1", "issue2", ...]
}

Critical Requirements:
- DO NOT output any text outside the JSON structure
- Use proper JSON escaping for special characters
- If a field cannot be extracted, use null
- Maintain accuracy - do not guess or infer information""",
                "source": "Synthetic - Real-world pattern",
                "license": "MIT",
            },
        ],
    }
    
    collected = []
    for category, prompt_list in samples.items():
        for prompt_data in prompt_list:
            file_path = PROMPTS_DIR / category / prompt_data["name"]
            file_path.write_text(prompt_data["content"], encoding='utf-8')
            
            collected.append({
                "file": prompt_data["name"],
                "source": prompt_data["source"],
                "category": category,
                "license": prompt_data["license"],
            })
    
    return collected


def generate_metadata(prompts: List[Dict]):
    """Generate METADATA.md file with attribution."""
    metadata_content = """# Compression Benchmark Dataset

This directory contains real-world prompts collected from public sources for testing prompt compression functionality.

## Dataset Information

- **Total Prompts**: {total}
- **Categories**: {categories}
- **Collection Date**: {date}

## Categories

""".format(
        total=len(prompts),
        categories=", ".join(CATEGORIES.keys()),
        date=time.strftime("%Y-%m-%d")
    )
    
    for category, description in CATEGORIES.items():
        category_prompts = [p for p in prompts if p["category"] == category]
        metadata_content += f"### {category.capitalize()}\n"
        metadata_content += f"{description}\n\n"
        metadata_content += f"**Count**: {len(category_prompts)}\n\n"
    
    metadata_content += "## Sources and Attribution\n\n"
    metadata_content += "| File | Source | License |\n"
    metadata_content += "|------|--------|---------|\n"
    
    for prompt in sorted(prompts, key=lambda x: (x["category"], x["file"])):
        metadata_content += f"| {prompt['file']} | {prompt['source']} | {prompt['license']} |\n"
    
    metadata_content += "\n## Usage\n\n"
    metadata_content += "These prompts are used for benchmarking compression performance.\n"
    metadata_content += "All prompts are from public sources or synthetic examples based on real-world patterns.\n"
    
    METADATA_FILE.write_text(metadata_content, encoding='utf-8')
    print(f"Generated metadata file: {METADATA_FILE}")


def main():
    """Main collection function."""
    print("Collecting real-world prompts for compression benchmarking...")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    
    collected = []
    
    # Step 1: Collect sample prompts (synthetic but based on real-world patterns)
    print("\n[Step 1/2] Creating sample prompts based on real-world patterns...")
    sample_prompts = create_sample_prompts()
    collected.extend(sample_prompts)
    print(f"  ✓ Created {len(sample_prompts)} sample prompts")
    
    # Step 2: Collect from GitHub repositories
    if HAS_REQUESTS:
        print("\n[Step 2/2] Collecting from public GitHub repositories...")
        print("  (This may take several minutes due to rate limiting...)\n")
        
        # Category mapping for better organization
        category_mapping = {
            "instructions": ["instructions", "prompts", "examples", "templates"],
            "structured": ["structured", "json", "templates", "schemas"],
            "conversations": ["conversations", "chat", "dialogue", "conversational"],
            "technical": ["technical", "docs", "documentation", "api", "guides"],
            "mixed": ["mixed", "general", "misc", "various"],
        }
        
        target_count = 100
        failed_repos = set()  # Track repositories that have failed to avoid retrying
        
        for source in SOURCES:
            if source["type"] == "github":
                repo_key = source["url"]
                
                # Skip repositories that have already failed
                if repo_key in failed_repos:
                    print(f"\n  ⏭️  Skipping {source['name']} (previously failed due to rate limits)")
                    continue
                
                print(f"\n  📦 Collecting from: {source['name']}")
                print(f"     URL: {source['url']}")
                try:
                    repo_failed = False
                    # Try to collect from different categories
                    for category in CATEGORIES.keys():
                        if len(collected) >= target_count:
                            break
                        
                        # Check if repo path matches category keywords or collect from all
                        repo_path = source.get("path", "")
                        keywords = category_mapping.get(category, [])
                        
                        # Collect from this category
                        max_per_category = min(30, target_count - len(collected))
                        prompts, repo_failed = collect_from_github_repo(
                            source["url"], 
                            category,
                            repo_path,
                            max_prompts=max_per_category
                        )
                        
                        # If repository failed due to rate limits, mark it and break
                        if repo_failed:
                            break
                        
                        if prompts:
                            collected.extend(prompts)
                            print(f"     ✓ Collected {len(prompts)} prompts for '{category}' category")
                        
                        if len(collected) >= target_count:
                            break
                    
                    # Mark repository as failed if collection failed
                    if repo_failed:
                        failed_repos.add(repo_key)
                        print(f"     ✗ Repository collection failed. Moving to next repository...")
                    
                    time.sleep(2)  # Rate limiting between repositories
                    
                    if len(collected) >= target_count:
                        print(f"\n  ✓ Reached target of {target_count}+ prompts!")
                        break
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  Collection interrupted by user")
                    break
                except Exception as e:
                    print(f"     ✗ Error collecting from {source['name']}: {e}")
                    continue
    else:
        print("\n  ⚠️  requests module not available. Install with: pip install requests")
        print("  Skipping GitHub collection. Only sample prompts will be used.")
    
    # Generate metadata
    print(f"\n{'='*60}")
    print("Generating metadata...")
    generate_metadata(collected)
    
    print(f"\n{'='*60}")
    print(f"✓ Collection complete!")
    print(f"✓ Total prompts collected: {len(collected)}")
    print(f"✓ Prompts saved to {PROMPTS_DIR}")
    print(f"✓ Metadata saved to {METADATA_FILE}")
    print(f"{'='*60}")
    
    # Category breakdown
    print("\nCategory breakdown:")
    for category in CATEGORIES.keys():
        count = sum(1 for p in collected if p.get("category") == category)
        print(f"  - {category}: {count} prompts")
    
    if len(collected) < 50:
        print(f"\n⚠️  Warning: Only {len(collected)} prompts collected.")
        print("  Consider:")
        print("  - Adding more repositories to SOURCES list")
        print("  - Expanding create_sample_prompts() with more examples")
        print("  - Manually adding prompts to the prompts/ directories")
    elif len(collected) >= 100:
        print(f"\n✅ Successfully collected {len(collected)} prompts (target: 100+)")


if __name__ == "__main__":
    main()

