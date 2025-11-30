"""
AI-Powered Repository Analyzer with Comprehensive Analysis
Features: Progress saving, rate limiting, large file handling, configuration support
COMPLETE CORRECTED VERSION 
"""

import os
import json
import requests
from datetime import datetime
from pathlib import Path
import html
import time
import anthropic
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

import numpy as np
import zipfile
import shutil

# Load environment variables if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default Configuration
DEFAULT_CONFIG = {
    'max_file_size_mb': 1,
    'max_lines_per_chunk': 800,
    'chunk_overlap_lines': 100,
    'api_delay_seconds': 1.0,
    'max_retries': 3,
    'retry_delay_seconds': 2,
    'skip_directories': ['node_modules', 'venv', 'env', '__pycache__', '.git', 'dist', 'build', 'target', '.pytest_cache'],
    'skip_patterns': ['*.min.js', '*.min.css', 'package-lock.json', 'yarn.lock'],
    'priority_extensions': ['.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.cpp', '.c', '.go', '.rs'],
    'supported_extensions': ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
                            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
                            '.html', '.css', '.scss', '.vue', '.sql']
}

class RepositoryAnalyzer:
    def __init__(self, api_key=None, config_file=None):
        """Initialize the analyzer with configuration"""
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            print("⚠️  No API key provided. Using direct API calls.")
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        
        # Load configuration
        self.config = self.load_config(config_file)
        self.progress_file = '.analyzer_progress.json'
        self.request_count = 0
        self.last_request_time = 0
        
    def load_config(self, config_file):
        """Load configuration from file or use defaults"""
        config = DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    config.update(user_config)
                print(f"✅ Loaded configuration from {config_file}")
            except Exception as e:
                print(f"⚠️  Error loading config: {e}. Using defaults.")
        
        return config
    
    def save_progress(self, analyses, repo_name):
        """Save analysis progress to disk"""
        progress_data = {
            'repo_name': repo_name,
            'timestamp': datetime.now().isoformat(),
            'analyses': analyses
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            print(f"💾 Progress saved ({len(analyses)} files)")
        except Exception as e:
            print(f"⚠️  Could not save progress: {e}")
    
    def load_progress(self, repo_name):
        """Load previous analysis progress"""
        if not os.path.exists(self.progress_file):
            return None
        
        try:
            with open(self.progress_file, 'r') as f:
                progress_data = json.load(f)
            
            if progress_data.get('repo_name') == repo_name:
                print(f"📂 Found previous progress: {len(progress_data['analyses'])} files")
                return progress_data['analyses']
        except Exception as e:
            print(f"⚠️  Could not load progress: {e}")
        
        return None
    
    def rate_limit_wait(self):
        """Implement rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config['api_delay_seconds']:
            wait_time = self.config['api_delay_seconds'] - time_since_last
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def clone_repository(self, repo_url, target_dir='temp_repo'):
        """Clone a git repository"""
        print(f"📥 Cloning repository: {repo_url}")
        
        if os.path.exists(target_dir):
            import shutil
            import stat
            
            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            
            try:
                shutil.rmtree(target_dir, onerror=remove_readonly)
            except:
                pass
        
        os.system(f'git clone {repo_url} {target_dir}')
        
        if not os.path.exists(target_dir):
            raise Exception("Failed to clone repository")
        
        print(f"✅ Repository cloned to {target_dir}")
        return target_dir

    def extract_zip_archive(self, zip_path, target_dir='temp_repo'):
        """Extract a ZIP archive into `target_dir`. Returns the repository root path inside target_dir.

        If the ZIP contains a single top-level folder, return that folder path so analysis
        will operate on the repository root. Otherwise, return `target_dir`.
        """
        print(f"📥 Extracting ZIP archive: {zip_path}")

        if not os.path.exists(zip_path):
            raise Exception(f"ZIP file not found: {zip_path}")

        # Clean existing target directory
        if os.path.exists(target_dir):
            import stat

            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)

            try:
                shutil.rmtree(target_dir, onerror=remove_readonly)
            except Exception:
                pass

        os.makedirs(target_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(target_dir)
        except Exception as e:
            raise Exception(f"Failed to extract ZIP archive: {e}")

        # If the archive extracted to a single top-level directory, use it as repo root
        entries = list(Path(target_dir).iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            repo_root = str(entries[0])
        else:
            repo_root = target_dir

        print(f"✅ ZIP extracted to: {repo_root}")
        return repo_root
    
    def should_skip_file(self, file_path):
        """Check if file should be skipped based on patterns"""
        from fnmatch import fnmatch
        
        file_name = os.path.basename(file_path)
        
        for pattern in self.config['skip_patterns']:
            if fnmatch(file_name, pattern):
                return True
        
        return False
    
    def get_code_files(self, directory):
        """Recursively get all code files with prioritization"""
        code_files = []
        
        print(f"🔍 Scanning for code files...")
        
        for root, dirs, files in os.walk(directory):
            # Skip configured directories
            dirs[:] = [d for d in dirs if d not in self.config['skip_directories']]
            
            for file in files:
                file_path = Path(root) / file
                
                if self.should_skip_file(str(file_path)):
                    continue
                
                if file_path.suffix in self.config['supported_extensions']:
                    file_size = file_path.stat().st_size
                    max_size = self.config['max_file_size_mb'] * 1024 * 1024
                    
                    if file_size <= max_size:
                        relative_path = file_path.relative_to(directory)
                        priority = 1 if file_path.suffix in self.config['priority_extensions'] else 2
                        
                        code_files.append({
                            'path': str(relative_path),
                            'full_path': str(file_path),
                            'extension': file_path.suffix,
                            'size': file_size,
                            'priority': priority
                        })
                    else:
                        print(f"⚠️  Skipping {file_path.name} (too large: {file_size / 1024 / 1024:.2f}MB)")
        
        # Sort by priority (priority files first)
        code_files.sort(key=lambda x: (x['priority'], x['path']))
        
        print(f"✅ Found {len(code_files)} code files")
        return code_files
    
    def read_file_content(self, file_path):
        """Read file content with encoding handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️  Could not read {file_path}: {e}")
                return None
    
    def extract_code_structure(self, content, file_path):
        """Extract key code elements for better context"""
        structure = {
            'imports': [],
            'functions': [],
            'classes': [],
            'exports': [],
            'key_lines': []
        }
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Imports/Dependencies
            if any(keyword in line_stripped for keyword in ['import', 'require', 'include', 'using', 'from']):
                if len(line_stripped) < 200:  # Avoid huge lines
                    structure['imports'].append((i, line_stripped))
            
            # Function definitions
            if any(keyword in line_stripped for keyword in ['def ', 'function ', 'const ', 'async ', 'public ', 'private ']):
                if '(' in line_stripped and len(line_stripped) < 200:
                    structure['functions'].append((i, line_stripped))
            
            # Class definitions
            if any(keyword in line_stripped for keyword in ['class ', 'interface ', 'struct ']):
                if len(line_stripped) < 200:
                    structure['classes'].append((i, line_stripped))
            
            # Exports
            if any(keyword in line_stripped for keyword in ['export', 'module.exports']):
                if len(line_stripped) < 200:
                    structure['exports'].append((i, line_stripped))
        
        return structure
    
    def analyze_repository_context(self, repo_path, code_files):
        """Analyze repository-wide context and relationships"""
        print("\n🔍 Analyzing repository structure and dependencies...")
        
        repo_context = {
            'file_dependencies': {},
            'common_imports': {},
            'file_purposes': {},
            'architecture_pattern': 'Unknown'
        }
        
        # Detect architecture pattern
        has_mvc = any('models' in f['path'] or 'views' in f['path'] or 'controllers' in f['path'] for f in code_files)
        has_components = any('components' in f['path'] for f in code_files)
        has_api = any('api' in f['path'] or 'routes' in f['path'] for f in code_files)
        
        if has_mvc:
            repo_context['architecture_pattern'] = 'MVC (Model-View-Controller)'
        elif has_components:
            repo_context['architecture_pattern'] = 'Component-Based (React/Vue)'
        elif has_api:
            repo_context['architecture_pattern'] = 'API-First Architecture'
        
        # Analyze file relationships
        for file_info in code_files:
            content = self.read_file_content(file_info['full_path'])
            if not content:
                continue
            
            structure = self.extract_code_structure(content, file_info['path'])
            
            # Track imports to find dependencies
            imports = [imp[1] for imp in structure['imports']]
            repo_context['file_dependencies'][file_info['path']] = imports
            
            # Count common imports
            for imp in imports:
                for keyword in ['express', 'react', 'vue', 'django', 'flask', 'mongoose', 'sequelize', 'axios']:
                    if keyword in imp.lower():
                        repo_context['common_imports'][keyword] = repo_context['common_imports'].get(keyword, 0) + 1
        
        print(f"✅ Architecture detected: {repo_context['architecture_pattern']}")
        if repo_context['common_imports']:
            print(f"✅ Key frameworks: {', '.join(repo_context['common_imports'].keys())}")
        
        return repo_context
    
    def smart_code_sampling(self, content, file_path, max_lines=1000):
        """Intelligently sample code sections for better context"""
        lines = content.split('\n')
        total_lines = len(lines)
        
        if total_lines <= max_lines:
            return content, 1, total_lines, False
        
        # Extract structure
        structure = self.extract_code_structure(content, file_path)
        
        # Priority sections to include
        priority_lines = set()
        
        # Always include: imports (top), exports (bottom), and key definitions
        priority_lines.update(range(0, min(50, total_lines)))  # First 50 lines (imports, setup)
        priority_lines.update(range(max(0, total_lines - 30), total_lines))  # Last 30 lines (exports)
        
        # Include function/class definitions with context
        for line_num, _ in structure['functions'] + structure['classes']:
            # Include 5 lines before and 20 after each definition
            start = max(0, line_num - 5)
            end = min(total_lines, line_num + 20)
            priority_lines.update(range(start, end))
        
        # Convert to sorted list
        selected_lines = sorted(list(priority_lines))
        
        # Build sampled content
        sampled_lines = []
        last_line = -1
        
        for line_idx in selected_lines:
            if line_idx > last_line + 1:
                sampled_lines.append(f"... [Lines {last_line + 2}-{line_idx} omitted] ...")
            sampled_lines.append(lines[line_idx])
            last_line = line_idx
        
        sampled_content = '\n'.join(sampled_lines)
        coverage_percent = (len(selected_lines) / total_lines) * 100
        
        print(f"   📄 Smart sampling: {len(selected_lines)}/{total_lines} lines ({coverage_percent:.1f}% coverage)")
        
        return sampled_content, 1, total_lines, True
    
    def analyze_file_with_claude(self, file_path, file_content, chunk_info=None, repo_context=None):
        """Analyze a single file or chunk using Claude API with repository context"""
        
        # Use smart sampling instead of simple truncation
        sampled_content, start_line, total_lines, was_sampled = self.smart_code_sampling(file_content, file_path)
        
        # Prepare content with line numbers
        lines = sampled_content.split('\n')
        numbered_content = '\n'.join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])
        
        # Build context information
        context_info = ""
        if repo_context:
            context_info = f"""

REPOSITORY CONTEXT:
Architecture: {repo_context.get('architecture_pattern', 'Unknown')}
Key Frameworks: {', '.join(repo_context.get('common_imports', {}).keys())}

File Dependencies:
{chr(10).join(repo_context.get('file_dependencies', {}).get(file_path, [])[:5])}
"""
        
        sampling_note = ""
        if was_sampled:
            sampling_note = f"\nNOTE: This is a large file ({total_lines} lines). Showing intelligently sampled sections including imports, key functions, and exports."
        
        chunk_context = ""
        if chunk_info and chunk_info['total_chunks'] > 1:
            chunk_context = f"\nNOTE: This is chunk {chunk_info['chunk_num']}/{chunk_info['total_chunks']} of a large file (lines {chunk_info['start_line']}-{chunk_info['end_line']})."
        
        prompt = f"""Analyze this code file in extreme detail and provide a comprehensive structured evaluation with specific code references.

File: {file_path}{context_info}{sampling_note}{chunk_context}
Code with line numbers:
```
{numbered_content}
```

IMPORTANT CONTEXT CONSIDERATIONS:
- Consider the repository architecture pattern when evaluating design decisions
- Take framework-specific best practices into account
- Evaluate dependencies and how this file fits in the overall system
- For large files, focus on the sampled key sections (imports, main functions, exports)

Provide a DETAILED analysis with SPECIFIC line number references for every point. Respond ONLY with valid JSON (no markdown, no backticks, no preamble).

Return JSON with this EXACT structure:
{{
  "filePurpose": "2-3 sentence description of what this file does, its main responsibility, and key technologies/integrations used",
  "accuracy": {{
    "score": <number 0-100>,
    "explanation": "Detailed explanation",
    "details": ["Point 1 with line references", "Point 2"]
  }},
  "complexity": {{
    "score": <number 0-100>,
    "explanation": "Detailed explanation",
    "details": ["Point 1", "Point 2"]
  }},
  "efficiency": {{
    "score": <number 0-100>,
    "explanation": "Detailed explanation",
    "details": ["Point 1", "Point 2"]
  }},
  "maintainability": {{
    "score": <number 0-100>,
    "explanation": "Detailed explanation",
    "details": ["Point 1", "Point 2"]
  }},
  "documentation": {{
    "score": <number 0-100>,
    "explanation": "Detailed explanation",
    "details": ["Point 1", "Point 2"]
  }},
  "overallScore": <number 0-100>,
  "strengths": [
    {{"point": "Strength", "location": "Lines X-Y", "impact": "Why good"}}
  ],
  "weaknesses": [
    {{"issue": "Problem", "location": "Line X", "code": "Code snippet", "severity": "High/Medium/Low", "reason": "Why problematic", "category": "Frontend/Backend/Database/Security/Performance/CodeQuality"}}
  ],
  "suggestions": [
    {{"suggestion": "What to improve", "location": "Line X", "current": "Current code", "recommended": "Recommended fix", "benefit": "Why this helps", "category": "Frontend/Backend/Database/Security/Performance/CodeQuality"}}
  ],
  "codeQualityIssues": [
    {{"type": "Issue type", "location": "Line X", "description": "Detailed description", "priority": "High/Medium/Low", "category": "Frontend/Backend/Database/Security/Performance/CodeQuality"}}
  ]
}}

IMPORTANT: Categorize each weakness, suggestion, and issue into one of these categories:
- Frontend: UI components, styling, client-side logic, React/Vue/Angular, DOM manipulation
- Backend: Server logic, API endpoints, routing, middleware, business logic
- Database: SQL queries, schema design, migrations, ORM usage, data access
- Security: Authentication, authorization, input validation, encryption, vulnerabilities
- Performance: Optimization, caching, memory usage, blocking operations, efficiency
- CodeQuality: Documentation, naming, structure, patterns, maintainability"""

        # Retry logic
        for attempt in range(self.config['max_retries']):
            try:
                # Rate limiting
                self.rate_limit_wait()
                
                if self.client:
                    message = self.client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=2500,
                        temperature=0,  # Consistent results
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_text = message.content[0].text
                else:
                    response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01"
                        },
                        json={
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 2500,
                            "temperature": 0,
                            "messages": [{"role": "user", "content": prompt}]
                        }
                    )
                    response_text = response.json()['content'][0]['text']
                
                # Enhanced JSON extraction
                clean_text = response_text.strip()
                clean_text = clean_text.replace('```json', '').replace('```', '').strip()
                
                first_brace = clean_text.find('{')
                last_brace = clean_text.rfind('}')
                
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    clean_text = clean_text[first_brace:last_brace + 1]
                
                # Try multiple parsing methods
                analysis = None
                parse_attempts = [
                    lambda: json.loads(clean_text),
                    lambda: json.loads(clean_text.replace('\n', ' ')),
                    lambda: json.loads(clean_text.replace('\\', '\\\\')),
                ]
                
                for parse_attempt in parse_attempts:
                    try:
                        analysis = parse_attempt()
                        break
                    except json.JSONDecodeError:
                        continue
                
                if analysis is None:
                    raise json.JSONDecodeError("All parsing attempts failed", clean_text, 0)
                
                # Validate required fields
                required_fields = ['filePurpose', 'accuracy', 'complexity', 'efficiency', 'maintainability', 
                                 'documentation', 'overallScore']
                
                for field in required_fields:
                    if field not in analysis:
                        if field == 'overallScore':
                            analysis[field] = 0
                        elif field == 'filePurpose':
                            analysis[field] = 'Purpose not analyzed'
                        else:
                            analysis[field] = {'score': 0, 'explanation': 'Not analyzed', 'details': []}
                
                # Add chunk info if applicable
                if chunk_info:
                    analysis['chunk_info'] = chunk_info
                
                return analysis
                
            except json.JSONDecodeError as e:
                if attempt < self.config['max_retries'] - 1:
                    wait_time = self.config['retry_delay_seconds'] * (attempt + 1)
                    print(f"   ⚠️  JSON parse error (attempt {attempt + 1}/{self.config['max_retries']}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️  JSON parsing failed after {self.config['max_retries']} attempts")
                    print(f"   Response preview: {response_text[:200] if 'response_text' in locals() else 'N/A'}...")
                    return None
                    
            except Exception as e:
                if attempt < self.config['max_retries'] - 1:
                    wait_time = self.config['retry_delay_seconds'] * (attempt + 1)
                    print(f"   ⚠️  Error (attempt {attempt + 1}/{self.config['max_retries']}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️  Analysis failed after {self.config['max_retries']} attempts: {e}")
                    return None
        
        return None
    
    def analyze_repository(self, repo_path, repo_name, resume=False):
        """Analyze all files in repository with progress saving and cross-file context"""
        print("\n" + "="*60)
        print("🚀 Starting Comprehensive Repository Analysis")
        print("="*60 + "\n")
        
        # Check for previous progress
        analyses = []
        analyzed_files = set()
        
        if resume:
            previous_analyses = self.load_progress(repo_name)
            if previous_analyses:
                analyses = previous_analyses
                analyzed_files = {a['file'] for a in analyses}
                print(f"▶️  Resuming from {len(analyzed_files)} previously analyzed files")
        
        code_files = self.get_code_files(repo_path)
        
        if not code_files:
            print("❌ No code files found to analyze")
            return None
        
        # PHASE 1: Analyze repository context
        repo_context = self.analyze_repository_context(repo_path, code_files)
        
        skipped_files = []
        total_files = len(code_files)
        
        print(f"\n📊 Analyzing {total_files} files with repository context...\n")
        
        for idx, file_info in enumerate(code_files, 1):
            file_path = file_info['path']
            
            # Skip if already analyzed
            if file_path in analyzed_files:
                print(f"\n⏭️  Skipping [{idx}/{total_files}]: {file_path} (already analyzed)")
                continue
            
            print(f"\n📊 Analyzing [{idx}/{total_files}]: {file_path}")
            
            content = self.read_file_content(file_info['full_path'])
            if not content:
                skipped_files.append((file_path, "Could not read file"))
                continue
            
            # Use smart sampling instead of chunking for large files
            analysis = self.analyze_file_with_claude(file_path, content, None, repo_context)
            
            if analysis:
                analysis['file'] = file_path
                analysis['extension'] = file_info['extension']
                analysis['lines_of_code'] = len(content.split('\n'))
                analyses.append(analysis)
                
                overall = analysis.get('overallScore', 0)
                print(f"   ✅ Overall Score: {overall}/100")
                
                # Save progress after each file
                self.save_progress(analyses, repo_name)
            else:
                skipped_files.append((file_path, "Analysis failed"))
                print(f"   ⏭️  Skipped due to errors")
        
        # Print summary
        if skipped_files:
            print(f"\n⚠️  Skipped {len(skipped_files)} file(s):")
            for file_path, reason in skipped_files:
                print(f"   • {file_path}: {reason}")
        
        print(f"\n📊 API Requests made: {self.request_count}")
        
        return analyses
    
    def create_score_chart(self, aggregate_metrics):
        """Create a bar chart for aggregate metrics"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        metrics = ['Accuracy', 'Complexity', 'Efficiency', 'Maintainability', 'Documentation']
        scores = [
            aggregate_metrics['accuracy'],
            aggregate_metrics['complexity'],
            aggregate_metrics['efficiency'],
            aggregate_metrics['maintainability'],
            aggregate_metrics['documentation']
        ]
        
        # Color coding
        colors = []
        for score in scores:
            if score >= 80:
                colors.append('#4caf50')  # Green
            elif score >= 60:
                colors.append('#ff9800')  # Orange
            else:
                colors.append('#f44336')  # Red
        
        bars = ax.barh(metrics, scores, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 2, i, f'{score:.1f}', va='center', fontweight='bold')
        
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Repository Quality Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, label='Excellent (80+)')
        ax.axvline(x=60, color='orange', linestyle='--', alpha=0.3, label='Good (60+)')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def create_overall_gauge(self, overall_score):
        """Create a gauge/speedometer for overall score"""
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Background segments
        ax.fill_between(theta[:33], 0, 1, color='#f44336', alpha=0.3)  # Red (0-33)
        ax.fill_between(theta[33:66], 0, 1, color='#ff9800', alpha=0.3)  # Orange (33-66)
        ax.fill_between(theta[66:], 0, 1, color='#4caf50', alpha=0.3)  # Green (66-100)
        
        # Needle
        needle_angle = np.pi * (1 - overall_score / 100)
        ax.plot([needle_angle, needle_angle], [0, 0.8], 'k-', linewidth=3)
        ax.plot(needle_angle, 0.8, 'ko', markersize=10)
        
        # Score text
        ax.text(np.pi/2, 0.3, f'{overall_score:.0f}', 
                ha='center', va='center', fontsize=32, fontweight='bold')
        ax.text(np.pi/2, 0.1, 'Overall Score', 
                ha='center', va='center', fontsize=12)
        
        ax.set_ylim(0, 1)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('W')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', transparent=True)
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def create_skills_radar_chart(self, skill_levels):
        """Create a radar chart showing skill levels across different areas"""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        
        skills = list(skill_levels.keys())
        scores = list(skill_levels.values())
        
        # Number of variables
        num_vars = len(skills)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, scores, 'o-', linewidth=2, color='#1a237e', markersize=6)
        ax.fill(angles, scores, alpha=0.25, color='#3949ab')
        
        # Fix axis to go in the right order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(skills, size=9)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        plt.title('Skills Assessment', size=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def create_progress_bars(self, analysis):
        """Create visual progress bars for individual file metrics"""
        fig, axes = plt.subplots(5, 1, figsize=(6, 4))
        
        metrics = ['accuracy', 'complexity', 'efficiency', 'maintainability', 'documentation']
        metric_names = ['Accuracy', 'Complexity', 'Efficiency', 'Maintainability', 'Documentation']
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            if metric in analysis and isinstance(analysis[metric], dict):
                score = analysis[metric].get('score', 0)
            else:
                score = 0
            
            # Color based on score
            if score >= 80:
                color = '#4caf50'
            elif score >= 60:
                color = '#ff9800'
            else:
                color = '#f44336'
            
            # Progress bar
            ax.barh([0], [score], height=0.6, color=color, alpha=0.8, edgecolor='black')
            ax.barh([0], [100-score], left=score, height=0.6, color='#e0e0e0', alpha=0.3)
            
            # Score label
            ax.text(score/2, 0, f'{score}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white' if score > 50 else 'black')
            
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def create_severity_pie_chart(self, analyses):
        """Create pie chart showing severity distribution"""
        high_count = 0
        medium_count = 0
        low_count = 0
        
        for analysis in analyses:
            if 'weaknesses' in analysis:
                for weakness in analysis['weaknesses']:
                    if isinstance(weakness, dict):
                        severity = weakness.get('severity', 'Medium')
                        if severity == 'High':
                            high_count += 1
                        elif severity == 'Medium':
                            medium_count += 1
                        else:
                            low_count += 1
        
        if high_count == 0 and medium_count == 0 and low_count == 0:
            return None
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        sizes = [high_count, medium_count, low_count]
        labels = [f'High ({high_count})', f'Medium ({medium_count})', f'Low ({low_count})']
        colors = ['#f44336', '#ff9800', '#fbc02d']
        explode = (0.1, 0, 0) if high_count > 0 else (0, 0, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title('Issue Severity Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    
    def calculate_aggregate_metrics(self, analyses):
        """Calculate repository-wide metrics"""
        if not analyses:
            return None
        
        metrics = {
            'accuracy': 0,
            'complexity': 0,
            'efficiency': 0,
            'maintainability': 0,
            'documentation': 0,
            'overallScore': 0
        }
        
        for analysis in analyses:
            for key in metrics.keys():
                if key in analysis:
                    if isinstance(analysis[key], dict):
                        metrics[key] += analysis[key].get('score', 0)
                    else:
                        metrics[key] += analysis[key]
        
        for key in metrics.keys():
            metrics[key] = round(metrics[key] / len(analyses), 1)
        
        return metrics

    def analyze_skills_gap(self, analyses, repo_context):
        """Analyze skills gaps using AI based on actual code analysis results.
        
        Uses Claude API to generate intelligent recommendations and learning paths
        tailored to the specific weaknesses found in the code.
        """
        # Aggregate findings from analyses
        skill_areas = {
            'Code Architecture': 0,
            'Documentation': 0,
            'Performance Optimization': 0,
            'Security Practices': 0,
            'Testing & QA': 0,
            'Code Quality': 0,
            'Database Design': 0,
            'API Design': 0,
            'Frontend Development': 0,
            'Backend Development': 0
        }
        
        total_files = len(analyses)
        all_weaknesses = []
        all_strengths = []
        
        for analysis in analyses:
            # Score skill areas
            if 'maintainability' in analysis and isinstance(analysis['maintainability'], dict):
                skill_areas['Code Architecture'] += analysis['maintainability'].get('score', 0) / total_files
            
            if 'documentation' in analysis and isinstance(analysis['documentation'], dict):
                skill_areas['Documentation'] += analysis['documentation'].get('score', 0) / total_files
            
            if 'efficiency' in analysis and isinstance(analysis['efficiency'], dict):
                skill_areas['Performance Optimization'] += analysis['efficiency'].get('score', 0) / total_files
            
            if 'complexity' in analysis and isinstance(analysis['complexity'], dict):
                skill_areas['Code Quality'] += analysis['complexity'].get('score', 0) / total_files
            
            # Collect weaknesses and strengths
            all_weaknesses.extend(analysis.get('weaknesses', []) or [])
            all_strengths.extend(analysis.get('strengths', []) or [])
        
        # Normalize scores
        for key in skill_areas:
            if skill_areas[key] == 0:
                skill_areas[key] = 50
            else:
                skill_areas[key] = min(100, max(0, skill_areas[key]))
        
        # Identify gaps
        gaps = []
        for skill, score in skill_areas.items():
            if score < 70:
                severity = 'Critical' if score < 40 else 'High' if score < 55 else 'Medium'
                gaps.append({
                    'skill': skill,
                    'score': round(score, 1),
                    'severity': severity
                })
        
        gaps.sort(key=lambda x: {'Critical': 0, 'High': 1, 'Medium': 2}.get(x['severity'], 3))
        
        # Prepare summary for AI analysis
        top_weaknesses = sorted(all_weaknesses, key=lambda x: {'High': 0, 'Medium': 1, 'Low': 2}.get(x.get('severity', 'Medium'), 3))[:10]
        top_strengths = all_strengths[:5]
        
        weakness_summary = '\n'.join([f"- {w.get('issue', 'Issue')}: {w.get('reason', '')}" for w in top_weaknesses if isinstance(w, dict)])
        strength_summary = '\n'.join([f"- {s.get('point', 'Strength')}" for s in top_strengths if isinstance(s, dict)])
        
        # Use Claude to generate AI-powered recommendations
        ai_prompt = f"""You are an expert software development mentor. Based on this code analysis, generate personalized recommendations and a learning path.

SKILL ASSESSMENT:
{json.dumps(skill_areas, indent=2)}

TOP ISSUES FOUND:
{weakness_summary or "No critical issues found"}

KEY STRENGTHS:
{strength_summary or "General foundation present"}

Generate a JSON response (ONLY valid JSON, no markdown) with:
1. recommendations: Array of {{gap, severity, issue, recommendation, resources: [], effort}}
2. learning_path: Array of {{order, topic, duration, rationale, subtopics: []}}
3. overall_proficiency: Number 0-100

Focus on:
- Actual gaps from the code analysis above
- Practical, actionable recommendations
- Specific learning resources (books, courses, tools)
- Realistic effort estimates
- Prioritized by severity and impact

Return ONLY valid JSON, no other text."""

        try:
            self.rate_limit_wait()
            
            if self.client:
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0.7,
                    messages=[{"role": "user", "content": ai_prompt}]
                )
                response_text = message.content[0].text
            else:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 2000,
                        "temperature": 0.7,
                        "messages": [{"role": "user", "content": ai_prompt}]
                    }
                )
                response_text = response.json()['content'][0]['text']
            
            # Parse AI response
            clean_text = response_text.strip()
            clean_text = clean_text.replace('```json', '').replace('```', '').strip()
            
            first_brace = clean_text.find('{')
            last_brace = clean_text.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                clean_text = clean_text[first_brace:last_brace + 1]
            
            ai_analysis = json.loads(clean_text)
            
            # Ensure required fields
            recommendations = ai_analysis.get('recommendations', [])
            learning_path = ai_analysis.get('learning_path', [])
            overall_proficiency = ai_analysis.get('overall_proficiency', round(sum(skill_areas.values()) / len(skill_areas), 1))
            
        except Exception as e:
            print(f"⚠️  AI skills analysis failed: {e}. Using fallback with recommendations.")
            # Fallback: generate structured recommendations from actual gaps
            recommendations = []
            
            if any(skill < 70 for skill in skill_areas.values()):
                for skill, score in sorted(skill_areas.items(), key=lambda x: x[1]):
                    if score < 70:
                        severity = 'Critical' if score < 40 else 'High' if score < 55 else 'Medium'
                        recommendations.append({
                            'gap': skill,
                            'severity': severity,
                            'issue': f'{skill} needs improvement (score: {score:.0f}/100)',
                            'recommendation': f'Focus on improving {skill}',
                            'resources': [
                                f'{skill} best practices',
                                f'Learning materials on {skill}',
                                f'Industry standards for {skill}'
                            ],
                            'effort': 'Medium (4-6 weeks)' if severity == 'High' else 'High (6-8 weeks)' if severity == 'Critical' else 'Low (2-4 weeks)'
                        })
            
            # Generate learning path based on gaps
            learning_path = []
            gap_skills = sorted([s for s, score in skill_areas.items() if score < 70], key=lambda s: skill_areas[s])
            
            for idx, skill in enumerate(gap_skills, 1):
                learning_path.append({
                    'order': idx,
                    'topic': f'Master {skill}',
                    'duration': '3-4 weeks',
                    'rationale': f'Gap identified: {skill} score is {skill_areas[skill]:.0f}/100',
                    'subtopics': [
                        f'Core concepts of {skill}',
                        f'Best practices in {skill}',
                        f'Real-world patterns',
                        f'Tools and frameworks'
                    ]
                })
            
            overall_proficiency = round(sum(skill_areas.values()) / len(skill_areas), 1)
        
        return {
            'skill_levels': skill_areas,
            'gaps': gaps,
            'recommendations': recommendations,
            'learning_path': learning_path,
            'overall_proficiency': overall_proficiency
        }
    
    def aggregate_repository_metrics_explanation(self, analyses):
        """Generate aggregate explanation for each metric across all files"""
        explanations = {}
        
        # Collect all metric explanations
        metric_details = {
            'accuracy': [],
            'complexity': [],
            'efficiency': [],
            'maintainability': [],
            'documentation': []
        }
        
        for analysis in analyses:
            for metric in metric_details.keys():
                if metric in analysis and isinstance(analysis[metric], dict):
                    if 'explanation' in analysis[metric]:
                        metric_details[metric].append(analysis[metric]['explanation'])
        
        # Generate synthetic explanations per metric
        for metric, explanations_list in metric_details.items():
            if explanations_list:
                # Count common issues
                common_themes = {}
                for exp in explanations_list:
                    if isinstance(exp, str):
                        # Simple keyword extraction
                        for keyword in ['error', 'bug', 'unused', 'exception', 'performance', 'quality', 'style', 'complexity']:
                            if keyword.lower() in exp.lower():
                                common_themes[keyword] = common_themes.get(keyword, 0) + 1
                
                # Synthesize explanation
                if metric == 'accuracy':
                    explanations[metric] = f"Accuracy issues detected across {len(explanations_list)} files. Common patterns include error handling, edge case coverage, and validation logic."
                elif metric == 'complexity':
                    explanations[metric] = f"Code complexity varies across {len(explanations_list)} files. Areas with high complexity should be refactored for maintainability."
                elif metric == 'efficiency':
                    explanations[metric] = f"Performance analysis across {len(explanations_list)} files shows areas for optimization in algorithms and data structures."
                elif metric == 'maintainability':
                    explanations[metric] = f"Maintainability concerns in {len(explanations_list)} files related to code structure, naming conventions, and modularity."
                elif metric == 'documentation':
                    explanations[metric] = f"Documentation coverage in {len(explanations_list)} files shows gaps in comments and docstrings."
            else:
                explanations[metric] = "Insufficient data for metric analysis."
        
        return explanations

    def aggregate_strengths(self, analyses):
        """Extract and rank top 4 strengths from all files"""
        all_strengths = []
        
        for analysis in analyses:
            if 'strengths' in analysis and analysis['strengths']:
                for strength in analysis['strengths']:
                    if isinstance(strength, dict):
                        all_strengths.append({
                            'point': strength.get('point', 'Unknown'),
                            'file': analysis.get('file', 'Unknown'),
                            'location': strength.get('location', 'N/A'),
                            'impact': strength.get('impact', 'N/A')
                        })
                    else:
                        all_strengths.append({
                            'point': str(strength),
                            'file': analysis.get('file', 'Unknown'),
                            'location': 'N/A',
                            'impact': 'Strong code practice'
                        })
        
        # Sort by relevance/impact (simplified)
        # Take top 4
        return all_strengths[:4]

    def aggregate_weaknesses_and_issues(self, analyses):
        """Extract all High/Medium severity issues from all files"""
        all_issues = []
        
        for analysis in analyses:
            # Collect weaknesses
            if 'weaknesses' in analysis and analysis['weaknesses']:
                for weakness in analysis['weaknesses']:
                    if isinstance(weakness, dict):
                        severity = weakness.get('severity', 'Medium')
                        if severity in ['High', 'Medium']:
                            # Ensure issue is not NA
                            issue = weakness.get('issue', '')
                            if not issue or issue == 'N/A':
                                issue = weakness.get('category', 'Code Quality Issue')
                            
                            # Ensure reason is not NA
                            reason = weakness.get('reason', '')
                            if not reason or reason == 'N/A':
                                reason = weakness.get('explanation', '') or weakness.get('details', '') or f'Issue found in {analysis.get("file", "Unknown")}'
                            
                            all_issues.append({
                                'issue': issue,
                                'severity': severity,
                                'category': weakness.get('category', 'CodeQuality'),
                                'category_label': self.clarify_issue_category(weakness.get('category', 'CodeQuality'), issue),
                                'file': analysis.get('file', 'Unknown'),
                                'location': weakness.get('location', 'N/A'),
                                'reason': reason,
                                'code': weakness.get('code', '')
                            })
            
            # Collect code quality issues
            if 'codeQualityIssues' in analysis and analysis['codeQualityIssues']:
                for issue in analysis['codeQualityIssues']:
                    if isinstance(issue, dict):
                        severity = issue.get('severity', 'Medium')
                        if severity in ['High', 'Medium']:
                            # Ensure issue is not NA
                            issue_text = issue.get('issue', '')
                            if not issue_text or issue_text == 'N/A':
                                issue_text = issue.get('category', 'Code Quality Issue')
                            
                            # Ensure reason is not NA
                            reason = issue.get('reason', '')
                            if not reason or reason == 'N/A':
                                reason = issue.get('explanation', '') or issue.get('details', '') or f'Quality issue in {analysis.get("file", "Unknown")}'
                            
                            all_issues.append({
                                'issue': issue_text,
                                'severity': severity,
                                'category': issue.get('category', 'CodeQuality'),
                                'category_label': self.clarify_issue_category(issue.get('category', 'CodeQuality'), issue_text),
                                'file': analysis.get('file', 'Unknown'),
                                'location': issue.get('location', 'N/A'),
                                'reason': reason,
                                'code': issue.get('code', '')
                            })
        
        # Sort by severity then file
        severity_order = {'High': 0, 'Medium': 1}
        all_issues.sort(key=lambda x: (severity_order.get(x['severity'], 2), x['file']))
        
        return all_issues
    
    def clarify_issue_category(self, category, issue_text=""):
        """Translate generic categories into specific, clear problem types"""
        category_map = {
            'CodeQuality': {
                'keywords': ['complexity', 'cyclomatic', 'nested', 'function too long', 'too many parameters', 'duplication', 'dead code', 'unused'],
                'label': 'Complexity & Maintainability',
                'description': 'Code structure issues (high complexity, long functions, deep nesting, code duplication)'
            },
            'Performance': {
                'keywords': ['loop', 'optimization', 'memory', 'inefficient', 'algorithm', 'cache', 'n+1'],
                'label': 'Performance Optimization',
                'description': 'Efficiency issues (slow algorithms, inefficient loops, memory usage)'
            },
            'Security': {
                'keywords': ['sql injection', 'xss', 'csrf', 'authentication', 'encryption', 'password', 'validation', 'sanitize'],
                'label': 'Security Vulnerabilities',
                'description': 'Security risks (input validation, injection attacks, authentication, data protection)'
            },
            'Documentation': {
                'keywords': ['comment', 'docstring', 'readme', 'documented', 'unclear', 'no documentation'],
                'label': 'Documentation Gaps',
                'description': 'Missing or inadequate documentation (unclear purpose, missing comments, no API docs)'
            },
            'Testing': {
                'keywords': ['test', 'coverage', 'untested', 'no test', 'mock'],
                'label': 'Test Coverage Issues',
                'description': 'Testing gaps (low coverage, missing tests, untested code paths)'
            },
            'Database': {
                'keywords': ['query', 'index', 'database', 'sql', 'schema', 'normalization', 'connection'],
                'label': 'Database Design',
                'description': 'Database issues (inefficient queries, missing indexes, schema problems)'
            },
            'API': {
                'keywords': ['api', 'endpoint', 'request', 'response', 'parameter', 'return', 'interface'],
                'label': 'API Design',
                'description': 'API design issues (unclear contracts, inconsistent interfaces, poor naming)'
            },
            'Architecture': {
                'keywords': ['architecture', 'design pattern', 'coupling', 'separation', 'modularity', 'dependency'],
                'label': 'Architectural Issues',
                'description': 'Design problems (poor modularity, tight coupling, wrong patterns)'
            },
            'Frontend': {
                'keywords': ['ui', 'frontend', 'html', 'css', 'javascript', 'responsive', 'accessibility'],
                'label': 'Frontend Issues',
                'description': 'UI/Frontend problems (responsiveness, accessibility, browser compatibility)'
            },
            'Backend': {
                'keywords': ['backend', 'server', 'handler', 'middleware', 'service', 'route'],
                'label': 'Backend Issues',
                'description': 'Server-side problems (error handling, middleware, service logic)'
            }
        }
        
        # Try to match based on keywords in issue text
        issue_lower = issue_text.lower()
        for cat, info in category_map.items():
            for keyword in info.get('keywords', []):
                if keyword.lower() in issue_lower:
                    return info['label']
        
        # Default mapping
        return category_map.get(category, {}).get('label', category)
    
    def generate_gap_explanations(self, analyses, skill_levels):
        """Generate evidence-based explanations for skill gaps based on actual findings with code evidence"""
        explanations = {}
        evidence = {}  # Store detailed evidence for each skill
        
        # Collect all issues by category with clarification
        issues_by_category = {}
        for analysis in analyses:
            if 'weaknesses' in analysis and analysis['weaknesses']:
                for weakness in analysis['weaknesses']:
                    if isinstance(weakness, dict):
                        cat = weakness.get('category', 'Other')
                        if cat not in issues_by_category:
                            issues_by_category[cat] = []
                        # Add file reference
                        weakness_copy = weakness.copy()
                        weakness_copy['file'] = analysis.get('file', 'Unknown')
                        issues_by_category[cat].append(weakness_copy)
            
            if 'codeQualityIssues' in analysis and analysis['codeQualityIssues']:
                for issue in analysis['codeQualityIssues']:
                    if isinstance(issue, dict):
                        cat = issue.get('category', 'Other')
                        if cat not in issues_by_category:
                            issues_by_category[cat] = []
                        issue_copy = issue.copy()
                        issue_copy['file'] = analysis.get('file', 'Unknown')
                        issues_by_category[cat].append(issue_copy)
        
        # Map skill areas to issue categories
        skill_category_map = {
            'Code Architecture': ['Architecture', 'Design', 'Structure'],
            'Documentation': ['Documentation'],
            'Performance Optimization': ['Performance', 'Efficiency'],
            'Security Practices': ['Security'],
            'Testing & QA': ['Testing', 'Test'],
            'Code Quality': ['CodeQuality', 'Quality'],
            'Database Design': ['Database', 'Query'],
            'API Design': ['API'],
            'Frontend Development': ['Frontend', 'UI'],
            'Backend Development': ['Backend', 'Server']
        }
        
        for skill, score in skill_levels.items():
            # Find relevant issues for this skill
            relevant_issues = []
            for category in skill_category_map.get(skill, []):
                if category in issues_by_category:
                    relevant_issues.extend(issues_by_category[category])
            
            # Generate evidence-based explanation with score justification
            if not relevant_issues:
                # High/acceptable score with no issues
                if score >= 80:
                    explanations[skill] = f"Score: {score}/100 ✅ - Excellent practices demonstrated. No significant issues detected."
                    evidence[skill] = "No issues found in code analysis"
                elif score >= 60:
                    explanations[skill] = f"Score: {score}/100 ⚠️ - No critical issues found, but potential for improvement exists in this area."
                    evidence[skill] = "Limited issues detected"
                else:
                    explanations[skill] = f"Score: {score}/100 - Limited evaluation data. Consider review and enhancement."
                    evidence[skill] = "Insufficient data for detailed analysis"
            else:
                # Has issues - explain the score based on issue severity with evidence
                high_count = len([i for i in relevant_issues if i.get('severity') == 'High'])
                medium_count = len([i for i in relevant_issues if i.get('severity') == 'Medium'])
                total_issues = len(relevant_issues)
                
                # Build evidence string with file locations and line numbers
                evidence_lines = []
                for issue in relevant_issues[:3]:  # Show top 3 issues
                    file_name = issue.get('file', 'Unknown')
                    location = issue.get('location', 'N/A')
                    issue_text = issue.get('issue', 'Issue')
                    code = issue.get('code', '')
                    
                    if code:
                        code_preview = code[:50].replace('\n', ' ')
                        evidence_lines.append(f"{file_name} (Line {location}): {code_preview}...")
                    else:
                        evidence_lines.append(f"{file_name} (Line {location}): {issue_text}")
                
                evidence[skill] = " | ".join(evidence_lines) if evidence_lines else f"{total_issues} issues found"
                
                if score >= 80:
                    explanations[skill] = f"Score: {score}/100 ✅ - Strong foundation with minor issues: {medium_count} medium concern(s)."
                elif score >= 60:
                    if high_count > 0:
                        explanations[skill] = f"Score: {score}/100 ⚠️ - {high_count} High severity + {medium_count} Medium issues found. Requires targeted improvements."
                    else:
                        explanations[skill] = f"Score: {score}/100 ⚠️ - {medium_count} quality issues detected. Improvement needed in practices."
                else:
                    explanations[skill] = f"Score: {score}/100 ❌ - {high_count} Critical + {medium_count} Medium issues. Immediate attention required: {total_issues} total issues found."
        
        return explanations, evidence

    def generate_immediate_fixes(self, analyses):
        """Generate prioritized immediate fix list from High severity issues"""
        fixes = []
        
        for analysis in analyses:
            if 'weaknesses' in analysis and analysis['weaknesses']:
                for weakness in analysis['weaknesses']:
                    if isinstance(weakness, dict) and weakness.get('severity') == 'High':
                        # Try to find matching suggestion for this weakness
                        matching_suggestion = None
                        if 'suggestions' in analysis:
                            for suggestion in analysis['suggestions']:
                                if isinstance(suggestion, dict):
                                    # Simple heuristic: suggestions near the weakness location
                                    if suggestion.get('location') == weakness.get('location'):
                                        matching_suggestion = suggestion
                                        break
                        
                        fixes.append({
                            'what': weakness.get('issue', 'Issue'),
                            'file': analysis.get('file', 'Unknown'),
                            'location': weakness.get('location', 'N/A'),
                            'current': weakness.get('code', ''),
                            'reason': weakness.get('reason', ''),
                            'recommended': matching_suggestion.get('suggestion', 'Review and refactor') if matching_suggestion else 'See suggestions section',
                            'benefit': matching_suggestion.get('benefit', 'Improved code quality') if matching_suggestion else 'Resolves critical issue',
                            'category': weakness.get('category', 'CodeQuality')
                        })
        
        # Limit to top 15 most critical
        return fixes[:15]

    def aggregate_suggestions(self, analyses):
        """Consolidate all improvement suggestions grouped by category"""
        suggestions_by_category = {}
        
        for analysis in analyses:
            if 'suggestions' in analysis and analysis['suggestions']:
                for suggestion in analysis['suggestions']:
                    if isinstance(suggestion, dict):
                        category = suggestion.get('category', 'CodeQuality')
                        if category not in suggestions_by_category:
                            suggestions_by_category[category] = []
                        
                        suggestions_by_category[category].append({
                            'suggestion': suggestion.get('suggestion', 'Unknown'),
                            'file': analysis.get('file', 'Unknown'),
                            'location': suggestion.get('location', 'N/A'),
                            'current': suggestion.get('current', ''),
                            'recommended': suggestion.get('recommended', ''),
                            'benefit': suggestion.get('benefit', '')
                        })
        
        return suggestions_by_category

    def save_json_report(self, analyses, repo_name, output_file='analysis_report.json', skills_analysis=None, aggregate_metrics=None):
        """Save analysis results as JSON to json_output folder including skill gap analysis"""
        import json
        
        # Ensure json_output folder exists
        json_output_dir = 'json_output'
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)
        
        # Create JSON output file path
        json_file_path = os.path.join(json_output_dir, output_file)
        
        # Calculate aggregate metrics if not provided
        if aggregate_metrics is None:
            aggregate_metrics = self.calculate_aggregate_metrics(analyses)
        
        # Calculate skills analysis if not provided
        if skills_analysis is None:
            skills_analysis = self.analyze_skills_gap(analyses, {'architecture_pattern': 'Unknown'})
        
        # Extract strengths, weaknesses, and suggestions
        top_strengths = self.aggregate_strengths(analyses)
        all_issues = self.aggregate_weaknesses_and_issues(analyses)
        suggestions_by_category = self.aggregate_suggestions(analyses)
        immediate_fixes = self.generate_immediate_fixes(analyses)
        
        # Convert analyses to JSON-serializable format
        json_data = {
            'repository': repo_name,
            'analysis_date': datetime.now().isoformat(),
            'total_files_analyzed': len(analyses) if analyses else 0,
            'aggregate_metrics': {
                'overall_score': aggregate_metrics.get('overallScore', 0),
                'accuracy': aggregate_metrics.get('accuracy', 0),
                'complexity': aggregate_metrics.get('complexity', 0),
                'efficiency': aggregate_metrics.get('efficiency', 0),
                'maintainability': aggregate_metrics.get('maintainability', 0),
                'documentation': aggregate_metrics.get('documentation', 0)
            },
            'file_analyses': analyses if analyses else [],
            'repository_strengths': top_strengths if top_strengths else [],
            'critical_issues': all_issues if all_issues else [],
            'improvement_suggestions': suggestions_by_category if suggestions_by_category else {},
            'immediate_fixes': immediate_fixes if immediate_fixes else [],
            'skills_gap_analysis': {
                'overall_proficiency': skills_analysis.get('overall_proficiency', 0),
                'skill_levels': skills_analysis.get('skill_levels', {}),
                'identified_gaps': skills_analysis.get('gaps', []),
                'recommendations': skills_analysis.get('recommendations', []),
                'learning_path': skills_analysis.get('learning_path', [])
            }
        }
        
        # Save to JSON file
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON analysis data saved to: json_output/{output_file}")
            return json_file_path
        except Exception as e:
            print(f"❌ Error saving JSON report: {e}")
            return None

    def generate_pdf_report(self, analyses, repo_name, output_file='analysis_report.pdf'):
        """Generate comprehensive PDF report"""
        # Ensure reports folder exists
        reports_dir = 'reports'
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        # Save to reports folder
        output_file_path = os.path.join(reports_dir, output_file)
        print(f"\n📄 Generating comprehensive PDF report: {output_file}")
        
        doc = SimpleDocTemplate(output_file_path, pagesize=letter,
                                rightMargin=0.75*inch, leftMargin=0.75*inch,
                                topMargin=1*inch, bottomMargin=0.75*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Reduce font sizes for compact display (single file - max 5 pages)
        styles['Normal'].fontSize = 10
        styles['Normal'].spaceAfter = 3
        styles['Normal'].leading = 11
        styles['Italic'].fontSize = 10
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=HexColor('#1a237e'),
            spaceAfter=14,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=HexColor('#283593'),
            spaceAfter=9,
            spaceBefore=7,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=11,
            textColor=HexColor('#3949ab'),
            spaceAfter=5,
            spaceBefore=5,
            fontName='Helvetica-Bold'
        )
        
        small_heading_style = ParagraphStyle(
            'SmallHeading',
            parent=styles['Heading4'],
            fontSize=10,
            textColor=HexColor('#5c6bc0'),
            spaceAfter=4,
            fontName='Helvetica'
        )
        
        # Title page
        story.append(Paragraph(f"🤖 Comprehensive AI Repository Analysis", title_style))
        story.append(Paragraph(f"<b>Repository:</b> {repo_name}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Files Analyzed:</b> {len(analyses)}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Lines of Code:</b> {sum(a.get('lines_of_code', 0) for a in analyses)}", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Aggregate metrics
        aggregate = self.calculate_aggregate_metrics(analyses)
        
        story.append(Paragraph("📊 Overall Repository Metrics", heading_style))
        
        metrics_data = [
            ['Metric', 'Score', 'Grade', 'Status'],
            ['Overall Score', f"{aggregate['overallScore']}/100", 
             self.get_grade(aggregate['overallScore']), 
             self.get_status(aggregate['overallScore'])],
            ['Accuracy', f"{aggregate['accuracy']}/100", 
             self.get_grade(aggregate['accuracy']),
             self.get_status(aggregate['accuracy'])],
            ['Complexity', f"{aggregate['complexity']}/100", 
             self.get_grade(aggregate['complexity']),
             self.get_status(aggregate['complexity'])],
            ['Efficiency', f"{aggregate['efficiency']}/100", 
             self.get_grade(aggregate['efficiency']),
             self.get_status(aggregate['efficiency'])],
            ['Maintainability', f"{aggregate['maintainability']}/100", 
             self.get_grade(aggregate['maintainability']),
             self.get_status(aggregate['maintainability'])],
            ['Documentation', f"{aggregate['documentation']}/100", 
             self.get_grade(aggregate['documentation']),
             self.get_status(aggregate['documentation'])],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.3*inch, 1*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#e8eaf6')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#3949ab')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#e8eaf6'), white]),
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.08*inch))
        
        # NEW SECTION 3: Detailed Repository-Wide Metric Analysis
        story.append(PageBreak())
        story.append(Paragraph("📊 Detailed Repository-Wide Metric Analysis", heading_style))
        story.append(Spacer(1, 0.04*inch))
        
        metric_explanations = self.aggregate_repository_metrics_explanation(analyses)
        
        for metric_name in ['Accuracy', 'Complexity', 'Efficiency', 'Maintainability', 'Documentation']:
            key = metric_name.lower()
            score = aggregate.get(key, 50)
            grade = self.get_grade(score)
            
            explanation = metric_explanations.get(key, "No data available")
            
            story.append(Paragraph(f"<b>{metric_name}: {score}/100 ({grade})</b>", small_heading_style))
            story.append(Paragraph(f"{html.escape(str(explanation))}", styles['Normal']))
            
            # Add 3-5 key findings (compact)
            story.append(Paragraph("<i>Key Findings:</i>", styles['Normal']))
            if key == 'accuracy':
                story.append(Paragraph("• Error handling consistency across codebase", styles['Normal']))
                story.append(Paragraph("• Edge case coverage in critical functions", styles['Normal']))
                story.append(Paragraph("• Input validation thoroughness", styles['Normal']))
            elif key == 'complexity':
                story.append(Paragraph("• Cyclomatic complexity levels in core modules", styles['Normal']))
                story.append(Paragraph("• Nesting depth and code branching patterns", styles['Normal']))
                story.append(Paragraph("• Opportunities for code refactoring", styles['Normal']))
            elif key == 'efficiency':
                story.append(Paragraph("• Algorithm performance optimization potential", styles['Normal']))
                story.append(Paragraph("• Memory usage and resource allocation", styles['Normal']))
                story.append(Paragraph("• Query optimization opportunities", styles['Normal']))
            elif key == 'maintainability':
                story.append(Paragraph("• Code structure and modularity ratings", styles['Normal']))
                story.append(Paragraph("• Naming conventions and code readability", styles['Normal']))
                story.append(Paragraph("• Technical debt assessment", styles['Normal']))
            elif key == 'documentation':
                story.append(Paragraph("• Comment coverage and quality", styles['Normal']))
                story.append(Paragraph("• Docstring completeness", styles['Normal']))
                story.append(Paragraph("• README and inline documentation", styles['Normal']))
            
            story.append(Spacer(1, 0.04*inch))
        
        story.append(PageBreak())
        
        # NEW SECTION 4: Top 4 Repository Strengths
        story.append(Paragraph("⭐ Top 4 Repository Strengths", heading_style))
        story.append(Spacer(1, 0.04*inch))
        
        top_strengths = self.aggregate_strengths(analyses)
        
        if top_strengths:
            for idx, strength in enumerate(top_strengths, 1):
                story.append(Paragraph(f"<b>{idx}. {html.escape(str(strength['point']))}</b>", small_heading_style))
                story.append(Paragraph(f"<i>File:</i> {html.escape(str(strength['file']))}", styles['Normal']))
                story.append(Paragraph(f"<i>Location:</i> {html.escape(str(strength['location']))}", styles['Normal']))
                story.append(Paragraph(f"<i>Impact:</i> {html.escape(str(strength['impact']))}", styles['Normal']))
                story.append(Spacer(1, 0.04*inch))
        else:
            story.append(Paragraph("No significant strengths identified.", styles['Normal']))
        
        story.append(Spacer(1, 0.08*inch))
        
        # NEW SECTION 5: Critical Weaknesses & Issues (ENHANCED)
        story.append(Paragraph("🔴 Critical Weaknesses & Issues Analysis", heading_style))
        story.append(Spacer(1, 0.04*inch))
        
        issues = self.aggregate_weaknesses_and_issues(analyses)
        
        if issues:
            # Statistics section
            high_count = len([i for i in issues if i['severity'] == 'High'])
            medium_count = len([i for i in issues if i['severity'] == 'Medium'])
            
            # Group by category
            issues_by_category = {}
            for issue in issues:
                cat = issue.get('category', 'Other')
                if cat not in issues_by_category:
                    issues_by_category[cat] = []
                issues_by_category[cat].append(issue)
            
            # Summary table
            summary_data = [
                ['Severity', 'Count', 'Percentage'],
                ['🔴 High', str(high_count), f"{round(100*high_count/len(issues), 1)}%"],
                ['🟠 Medium', str(medium_count), f"{round(100*medium_count/len(issues), 1)}%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 1*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3949ab')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, 1), HexColor('#ffebee')),
                ('BACKGROUND', (0, 2), (-1, 2), HexColor('#fff3e0')),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdbdbd'))
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.08*inch))
            
            story.append(Paragraph(f"<i>Total High/Medium severity issues: <b>{len(issues)}</b></i>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            # Issues by category
            for category in sorted(issues_by_category.keys()):
                cat_issues = issues_by_category[category]
                cat_high = len([i for i in cat_issues if i['severity'] == 'High'])
                cat_medium = len([i for i in cat_issues if i['severity'] == 'Medium'])
                
                # Use clarified category label if available
                category_label = cat_issues[0].get('category_label', category) if cat_issues else category
                
                story.append(Paragraph(
                    f"<b>{category_label}</b> ({cat_high} High, {cat_medium} Medium)",
                    small_heading_style
                ))
                story.append(Spacer(1, 0.08*inch))
                
                for idx, issue in enumerate(cat_issues, 1):
                    severity_color = {'High': '#d32f2f', 'Medium': '#f57c00'}.get(issue['severity'], '#f57c00')
                    severity_bg = {'High': '#ffcdd2', 'Medium': '#ffe0b2'}.get(issue['severity'], '#fff3e0')
                    
                    issue_text = html.escape(str(issue['issue']))
                    file_text = html.escape(str(issue['file']))
                    location_text = html.escape(str(issue['location']))
                    reason_text = html.escape(str(issue['reason']))
                    category_label = html.escape(str(issue.get('category_label', issue.get('category', 'Other'))))
                    
                    # Enhanced header with box and clarified category
                    story.append(Paragraph(
                        f"<font color='{severity_color}'><b>[{issue['severity']}]</b></font> {issue_text} <font color='#666' size='8'>[{category_label}]</font>",
                        styles['Normal']
                    ))
                    
                    # Details table
                    details_data = [
                        ['File', file_text],
                        ['Location', location_text],
                        ['Issue', reason_text]
                    ]
                    
                    if issue.get('code'):
                        # Truncate code significantly and create Paragraph for proper wrapping
                        code_snippet = html.escape(str(issue['code'][:60]))
                        # Use Paragraph to enable text wrapping within the cell
                        code_para = Paragraph(f"<font face='Courier' size='6'>{code_snippet}</font>", styles['Normal'])
                        details_data.append(['Code', code_para])
                    
                    # Use smaller column to force wrapping - content column will auto-expand
                    details_table = Table(details_data, colWidths=[0.8*inch, 4.7*inch])
                    details_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), HexColor('#f5f5f5')),
                        ('BACKGROUND', (1, 0), (1, -1), HexColor(severity_bg)),
                        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 7),
                        ('LEFTPADDING', (0, 0), (-1, -1), 2),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                        ('TOPPADDING', (0, 0), (-1, -1), 1),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e0e0e0')),
                        ('WORDWRAP', (1, 0), (1, -1), True),  # Enable wrapping for all content cells
                    ]))
                    story.append(details_table)
                    story.append(Spacer(1, 0.04*inch))
                
                story.append(Spacer(1, 0.02*inch))
        else:
            story.append(Paragraph("No High or Medium severity issues found. ✅", styles['Normal']))
        
        story.append(Spacer(1, 0.1*inch))
        story.append(PageBreak())
        
        # NEW SECTION 6: Immediate Fixes
        story.append(Paragraph("🔧 Immediate Fixes - High Priority Actions", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        fixes = self.generate_immediate_fixes(analyses)
        
        if fixes:
            story.append(Paragraph(f"<i>Top {len(fixes)} critical fixes requiring immediate attention:</i>", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
            
            for idx, fix in enumerate(fixes, 1):
                what = html.escape(str(fix['what']))
                file_text = html.escape(str(fix['file']))
                location = html.escape(str(fix['location']))
                reason = html.escape(str(fix['reason']))
                recommended = html.escape(str(fix['recommended']))
                benefit = html.escape(str(fix['benefit']))
                category = html.escape(str(fix['category']))
                
                story.append(Paragraph(f"<b>{idx}. {what}</b> <font color='#f57c00'>[{category}]</font>", small_heading_style))
                story.append(Paragraph(f"  <i>File:</i> {file_text}", styles['Normal']))
                story.append(Paragraph(f"  <i>Location:</i> {location}", styles['Normal']))
                story.append(Paragraph(f"  <i>Why:</i> {reason}", styles['Normal']))
                
                if fix.get('current'):
                    story.append(Paragraph(f"  <i>Current:</i> <font face='Courier' size='8'>{html.escape(str(fix['current'][:80]))}</font>", styles['Normal']))
                
                story.append(Paragraph(f"  <i>Recommended:</i> {recommended}", styles['Normal']))
                story.append(Paragraph(f"  <i>Benefit:</i> {benefit}", styles['Normal']))
                story.append(Spacer(1, 0.06*inch))
        else:
            story.append(Paragraph("No High severity issues requiring immediate fixes. ✅", styles['Normal']))
        
        story.append(Spacer(1, 0.12*inch))
        
        # NEW SECTION 7: Improvement Suggestions
        story.append(Paragraph("💡 Improvement Suggestions", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        suggestions_by_cat = self.aggregate_suggestions(analyses)
        
        if suggestions_by_cat:
            for category in sorted(suggestions_by_cat.keys()):
                suggestions = suggestions_by_cat[category]
                story.append(Paragraph(f"<b>{category}</b>", small_heading_style))
                
                for sug in suggestions[:5]:  # Top 5 per category
                    sug_text = html.escape(str(sug['suggestion']))
                    file_text = html.escape(str(sug['file']))
                    location = html.escape(str(sug['location']))
                    benefit = html.escape(str(sug['benefit']))
                    
                    story.append(Paragraph(f"  • {sug_text}", styles['Normal']))
                    story.append(Paragraph(f"    <i>File:</i> {file_text}", styles['Normal']))
                    story.append(Paragraph(f"    <i>Location:</i> {location}", styles['Normal']))
                    story.append(Paragraph(f"    <i>Benefit:</i> {benefit}", styles['Normal']))
                    story.append(Spacer(1, 0.08*inch))
        else:
            story.append(Paragraph("No improvement suggestions at this time.", styles['Normal']))
        
        story.append(Spacer(1, 0.12*inch))
        story.append(PageBreak())
        
        # Skills Gap Analysis Section (moved to end)
        story.append(Paragraph("🎓 Skills Gap Analysis & Learning Path", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Analyze skills gap using just the analyses
        skills_analysis = self.analyze_skills_gap(analyses, {'architecture_pattern': 'Unknown'})
        
        # Overall proficiency
        overall_prof = skills_analysis['overall_proficiency']
        prof_grade = self.get_grade(overall_prof)
        story.append(Paragraph(f"<b>Overall Proficiency: {overall_prof}/100 ({prof_grade})</b>", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        # Skills radar chart
        try:
            radar_img = self.create_skills_radar_chart(skills_analysis['skill_levels'])
            radar = Image(radar_img, width=5.5*inch, height=5.5*inch)
            story.append(Paragraph("📊 Your Skills Assessment", subheading_style))
            story.append(radar)
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            print(f"⚠️  Could not generate skills radar: {e}")
        
        # Skill gaps section
        gaps = skills_analysis['gaps']
        if gaps:
            story.append(Paragraph("⚠️ Skill Gaps Identified", subheading_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Generate evidence-based explanations with code evidence
            gap_explanations, gap_evidence = self.generate_gap_explanations(analyses, skills_analysis['skill_levels'])
            
            for gap in gaps[:15]:
                gap_skill = html.escape(str(gap['skill']))
                gap_score = gap['score']
                gap_sev = gap['severity']
                sev_color = {'Critical': '#d32f2f', 'High': '#f57c00', 'Medium': '#fbc02d'}.get(gap_sev, '#f57c00')
                
                # Get evidence-based explanation and code evidence
                explanation = gap_explanations.get(gap_skill, "Evidence collected from code analysis")
                code_evidence = gap_evidence.get(gap_skill, "")
                
                story.append(Paragraph(
                    f"• <b>{gap_skill}</b>: {gap_score}/100 <font color='{sev_color}'>[{gap_sev}]</font>",
                    styles['Normal']
                ))
                story.append(Paragraph(
                    f"  <i style='font-size:8'>{explanation}</i>",
                    styles['Normal']
                ))
                
                # Add code evidence if available
                if code_evidence:
                    evidence_safe = html.escape(str(code_evidence))
                    story.append(Paragraph(
                        f"  <i style='font-size:7' color='#666'>Evidence: {evidence_safe}</i>",
                        styles['Normal']
                    ))
            
            story.append(Spacer(1, 0.12*inch))
        
        # Recommendations section
        recs = skills_analysis['recommendations']
        if recs:
            story.append(Paragraph("💡 Personalized Recommendations", subheading_style))
            story.append(Spacer(1, 0.1*inch))
            
            for rec in recs[:10]:
                rec_gap = html.escape(str(rec['gap']))
                rec_issue = html.escape(str(rec['issue']))
                rec_rec = html.escape(str(rec['recommendation']))
                rec_effort = html.escape(str(rec['effort']))
                
                story.append(Paragraph(f"<b>{rec_gap}</b>", small_heading_style))
                story.append(Paragraph(f"<i>Issue:</i> {rec_issue}", styles['Normal']))
                story.append(Paragraph(f"<i>Recommendation:</i> {rec_rec}", styles['Normal']))
                
                if 'resources' in rec and rec['resources']:
                    story.append(Paragraph(f"<i>Resources:</i>", styles['Normal']))
                    for resource in rec['resources'][:3]:
                        safe_res = html.escape(str(resource))
                        story.append(Paragraph(f"  • {safe_res}", styles['Normal']))
                
                story.append(Paragraph(f"<i>Effort:</i> {rec_effort}", styles['Normal']))
                story.append(Spacer(1, 0.06*inch))
            
            story.append(PageBreak())
        
        # Learning path section
        lp = skills_analysis['learning_path']
        if lp:
            story.append(Paragraph("📚 Your Personalized Learning Path", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            for item in lp:
                order = item['order']
                topic = html.escape(str(item['topic']))
                duration = html.escape(str(item['duration']))
                rationale = html.escape(str(item['rationale']))
                subtopics = item.get('subtopics', [])
                
                story.append(Paragraph(f"<b>Phase {order}: {topic}</b>", small_heading_style))
                story.append(Paragraph(f"<i>Duration:</i> {duration}", styles['Normal']))
                story.append(Paragraph(f"<i>Why:</i> {rationale}", styles['Normal']))
                
                if subtopics:
                    story.append(Paragraph(f"<i>Learn:</i>", styles['Normal']))
                    for subtopic in subtopics[:6]:
                        safe_sub = html.escape(str(subtopic))
                        story.append(Paragraph(f"  • {safe_sub}", styles['Normal']))
                
                story.append(Spacer(1, 0.1*inch))
        
        # Skills gap analysis removed per user request
        
        # Build PDF
        doc.build(story)
        print(f"✅ Comprehensive PDF report generated: reports/{output_file}")
    
    def get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90: return 'A+'
        elif score >= 80: return 'A'
        elif score >= 70: return 'B'
        elif score >= 60: return 'C'
        elif score >= 50: return 'D'
        else: return 'F'
    
    def get_status(self, score):
        """Get status indicator for score"""
        if score >= 80: return '✓ Excellent'
        elif score >= 60: return '~ Good'
        elif score >= 40: return '⚠ Needs Work'
        else: return '✗ Critical'

# Main execution
if __name__ == "__main__":
    import sys
    import argparse
    
    print("\n" + "="*60)
    print("[AI] Powered Comprehensive Repository Analyzer")
    print("   Enhanced with Progress Saving & Rate Limiting")
    print("="*60 + "\n")
    
    # Argument parser
    parser = argparse.ArgumentParser(description='Analyze code repositories with AI')
    parser.add_argument('repo_url', nargs='?', help='GitHub repository URL')
    parser.add_argument('--zip', '-z', help='Path to a local ZIP archive to analyze')
    parser.add_argument('--config', '-c', help='Configuration file path (YAML)')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume previous analysis')
    parser.add_argument('--max-files', type=int, help='Maximum number of files to analyze')
    
    args = parser.parse_args()
    
    # Determine input source: ZIP archive or GitHub URL
    zip_path = args.zip
    repo_url = args.repo_url

    if not zip_path and not repo_url:
        repo_url = input("Enter GitHub repository URL: ").strip()
        if not repo_url:
            print("❌ No repository URL or ZIP provided")
            sys.exit(1)

    # If the user passed a positional path that is a ZIP file (common on Windows),
    # treat it as if they had passed --zip. This avoids trying to `git clone` a .zip file.
    try:
        if repo_url and (repo_url.lower().endswith('.zip') or (os.path.isfile(repo_url) and zipfile.is_zipfile(repo_url))):
            zip_path = repo_url
            repo_url = None
    except Exception:
        # If any filesystem check fails, just continue and let clone/validation handle it
        pass
    
    # Initialize analyzer with config
    analyzer = RepositoryAnalyzer(config_file=args.config)
    
    try:
        # Prepare repository source: either extract ZIP or clone repo
        if zip_path:
            if repo_url:
                print("⚠️  Both --zip and a repo URL were provided; using the ZIP archive.")
            repo_dir = analyzer.extract_zip_archive(zip_path)
            repo_name = Path(repo_dir).name
        else:
            # Clone repository from URL
            repo_dir = analyzer.clone_repository(repo_url)
            repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Analyze repository
        analyses = analyzer.analyze_repository(repo_dir, repo_name, resume=args.resume)
        
        if analyses:
            # Limit files if specified
            if args.max_files and len(analyses) > args.max_files:
                print(f"\n⚠️  Limiting to first {args.max_files} files (use --max-files to change)")
                analyses = analyses[:args.max_files]
            
            # Generate timestamp for consistent naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Calculate aggregate metrics and skills analysis
            aggregate_metrics = analyzer.calculate_aggregate_metrics(analyses)
            skills_analysis = analyzer.analyze_skills_gap(analyses, {'architecture_pattern': 'Unknown'})
            
            # Generate comprehensive PDF report
            pdf_output = f"{repo_name}_comprehensive_analysis_{timestamp}.pdf"
            analyzer.generate_pdf_report(analyses, repo_name, pdf_output)
            
            # Generate JSON analysis report with skill gap data
            json_output = f"{repo_name}_analysis_data_{timestamp}.json"
            analyzer.save_json_report(analyses, repo_name, json_output, skills_analysis, aggregate_metrics)
            
            print("\n" + "="*60)
            print("✅ Comprehensive Analysis Complete!")
            print(f"📊 Total files analyzed: {len(analyses)}")
            print(f"📄 Detailed report saved to: reports/{pdf_output}")
            print(f"📋 Analysis data saved to: json_output/{json_output}")
            print(f"🔧 API requests made: {analyzer.request_count}")
            print("="*60 + "\n")
            
            # Clean up progress file on success
            if os.path.exists(analyzer.progress_file):
                os.remove(analyzer.progress_file)
                print("✅ Progress file cleaned up")
        else:
            print("\n❌ No analysis results generated")
        
        print("\n📝 Temporary repository kept at: temp_repo")
        print("   (You can delete it manually or it will be overwritten next run)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        print("💾 Progress saved! Use --resume to continue from where you left off")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💾 Progress may have been saved. Use --resume to continue")
        sys.exit(1) 