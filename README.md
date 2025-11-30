# 🤖 AI-Powered Repository Analyzer

A comprehensive code analysis tool that uses Claude AI to perform deep analysis of entire code repositories, generating detailed reports with actionable insights, skill gap analysis, and personalized learning paths.

## ✨ Features

### 📊 **Comprehensive Code Analysis**
- **Multi-file analysis** with intelligent code sampling for large files
- **5 key metrics**: Accuracy, Complexity, Efficiency, Maintainability, Documentation
- **Smart prioritization** of high-priority file types (.py, .js, .ts, etc.)
- **Repository-wide context** awareness for better insights

### 🎯 **Intelligent Issue Detection**
- **Severity-based categorization**: High, Medium, Low
- **Category classification**: Frontend, Backend, Database, Security, Performance, Code Quality
- **Specific line references** for every issue found
- **Code snippets** showing problematic patterns

### 🎓 **Skills Gap Analysis**
- **10 skill areas** evaluated: Architecture, Documentation, Performance, Security, Testing, Code Quality, Database Design, API Design, Frontend, Backend
- **AI-powered recommendations** tailored to your actual code issues
- **Personalized learning paths** with specific resources and timelines
- **Visual radar charts** showing skill distribution

### 📄 **Professional Reports**
- **PDF reports** with charts, tables, and visual metrics (5-page comprehensive format)
- **JSON exports** with full analysis data for programmatic use
- **Progress saving** to resume interrupted analyses
- **Rate limiting** to respect API quotas

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Anthropic API key ([get one here](https://console.anthropic.com/))
- Git (for cloning repositories)

### Setup

1. **Clone or download this script**
```bash
git clone <your-repo-url>
cd repository-analyzer
```

2. **Install dependencies**
```bash
pip install anthropic requests matplotlib reportlab numpy pyyaml python-dotenv
```

3. **Set up your API key**

Create a `.env` file:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

Or export it directly:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## 📖 Usage

### Basic Usage

**Analyze a GitHub repository:**
```bash
python analyzer.py https://github.com/username/repo-name
```

**Analyze a local ZIP archive:**
```bash
python analyzer.py --zip path/to/repository.zip
```

### Advanced Options

```bash
# Use custom configuration file
python analyzer.py https://github.com/user/repo --config my_config.yaml

# Resume interrupted analysis
python analyzer.py https://github.com/user/repo --resume

# Limit number of files analyzed
python analyzer.py https://github.com/user/repo --max-files 20
```

### Configuration File

Create a `config.yaml` file to customize behavior:

```yaml
# File size limits
max_file_size_mb: 1

# Chunking settings
max_lines_per_chunk: 800
chunk_overlap_lines: 100

# API settings
api_delay_seconds: 1.0
max_retries: 3
retry_delay_seconds: 2

# Directory exclusions
skip_directories:
  - node_modules
  - venv
  - .git
  - dist
  - build

# File pattern exclusions
skip_patterns:
  - "*.min.js"
  - "*.min.css"
  - package-lock.json

# Priority file extensions (analyzed first)
priority_extensions:
  - .py
  - .js
  - .ts
  - .tsx

# Supported file extensions
supported_extensions:
  - .py
  - .js
  - .jsx
  - .ts
  - .tsx
  - .java
  - .cpp
  - .c
  - .go
  - .rs
  - .html
  - .css
```

## 📊 Output

The analyzer generates two types of output:

### 1. PDF Report (`reports/`)
A comprehensive 5-page report including:
- **Overall metrics** with scores and grades
- **Detailed metric analysis** for each quality dimension
- **Top 4 strengths** from the entire codebase
- **Critical issues** categorized by severity and type
- **Immediate fixes** for high-priority problems
- **Improvement suggestions** grouped by category
- **Skills gap analysis** with visual radar chart
- **Personalized learning path** with phases and resources

### 2. JSON Data (`json_output/`)
Machine-readable analysis including:
- File-by-file analysis results
- Aggregate repository metrics
- All weaknesses and suggestions
- Skills assessment data
- Complete learning path recommendations

## 🏗️ How It Works

### Analysis Pipeline

1. **Repository Acquisition**
   - Clones Git repository or extracts ZIP archive
   - Scans for code files with supported extensions

2. **Context Building**
   - Analyzes repository structure (MVC, Component-based, etc.)
   - Detects frameworks and dependencies
   - Maps file relationships

3. **Intelligent Sampling**
   - For large files, extracts key sections (imports, functions, exports)
   - Maintains context while staying within token limits
   - Includes line numbers for precise references

4. **AI Analysis**
   - Uses Claude Sonnet 4 for deep code analysis
   - Evaluates 5 quality dimensions
   - Identifies specific issues with severity levels
   - Generates improvement suggestions

5. **Skills Assessment**
   - Aggregates findings across all files
   - Uses AI to identify skill gaps
   - Generates personalized recommendations
   - Creates structured learning paths

6. **Report Generation**
   - Compiles comprehensive PDF with visualizations
   - Exports structured JSON for integration
   - Saves progress for resumability

## 🎨 Sample Output

### Metrics Scores
```
Accuracy:        85/100 (A)  ✓ Excellent
Complexity:      72/100 (B)  ~ Good
Efficiency:      68/100 (C)  ~ Good
Maintainability: 78/100 (B)  ✓ Excellent
Documentation:   55/100 (D)  ⚠ Needs Work
```

### Issue Categories
- **Security**: Input validation, SQL injection risks
- **Performance**: N+1 queries, inefficient loops
- **Code Quality**: High complexity, code duplication
- **Documentation**: Missing docstrings, unclear comments

### Skills Gap Example
```
Code Architecture:     85/100 ✅
Performance Optimization: 62/100 ⚠️ (Critical issues in 3 files)
Security Practices:    45/100 ❌ (Input validation gaps)
```

## 🔧 Troubleshooting

### Common Issues

**"No API key provided"**
- Set `ANTHROPIC_API_KEY` environment variable
- Or create a `.env` file with your key

**"Failed to clone repository"**
- Ensure Git is installed and accessible
- Check repository URL is correct
- Try using `--zip` with a downloaded archive instead

**"JSON parsing failed"**
- The tool retries automatically up to 3 times
- Check your API key is valid and has credits
- Large files may need configuration adjustments

**Analysis interrupted**
- Use `--resume` flag to continue from where you left off
- Progress is automatically saved after each file

### Rate Limiting

The tool implements automatic rate limiting:
- 1 second delay between API calls (configurable)
- Automatic retries with exponential backoff
- Progress saving to prevent data loss

## 📝 Notes

- **Cost Estimation**: Each file analysis costs approximately 0.01-0.05 USD depending on file size
- **Time**: Analysis takes about 1-3 seconds per file
- **Large Repositories**: Use `--max-files` to limit scope for initial testing
- **Resume Feature**: Interrupted analyses can be resumed without losing progress

## 🤝 Contributing

Suggestions for improvements:
- Additional language support
- More analysis dimensions
- Integration with CI/CD pipelines
- Custom report templates
- Team collaboration features

## 📄 License

This tool uses the Anthropic Claude API. Ensure you comply with Anthropic's terms of service.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review your configuration file
3. Ensure your API key has sufficient credits
4. Check the console output for specific error messages

---

**Built with Claude Sonnet 4** - Powered by Anthropic AI 🚀
