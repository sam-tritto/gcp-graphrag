from utils.markdown_parser import render_markdown_to_html

def test_render_markdown_to_html_empty():
    assert render_markdown_to_html("") == ""
    assert render_markdown_to_html(None) == ""

def test_render_markdown_to_html_bold_and_italic():
    text = "This is **bold** and *italic* text."
    html_output = render_markdown_to_html(text)
    assert "<strong>bold</strong>" in html_output
    assert "<em>italic</em>" in html_output
    assert "This is " in html_output

def test_render_markdown_to_html_inline_code():
    text = "Run the command `uv run app.py` to start."
    html_output = render_markdown_to_html(text)
    assert "<code style=" in html_output
    assert "uv run app.py" in html_output

def test_render_markdown_to_html_code_block():
    text = "Here is a code block:\n```python\ndef hello():\n    return 'world'\n```"
    html_output = render_markdown_to_html(text)
    assert "<pre style=" in html_output
    assert "<code style=" in html_output
    assert "def hello():" in html_output
    assert "    return &#x27;world&#x27;" in html_output
    # Verify newlines are preserved inside code block and not converted to <br>
    assert "def hello():<br>" not in html_output

def test_render_markdown_to_html_unordered_list():
    text = "List of topics:\n* Topic A\n* Topic B\n- Topic C"
    html_output = render_markdown_to_html(text)
    assert '<ul style="margin-top: 4px; margin-bottom: 4px; padding-left: 20px;">' in html_output
    assert "<li>Topic A</li>" in html_output
    assert "<li>Topic B</li>" in html_output
    assert "<li>Topic C</li>" in html_output
    assert "</ul>" in html_output

def test_render_markdown_to_html_ordered_list():
    text = "Steps:\n1. Step One\n2. Step Two"
    html_output = render_markdown_to_html(text)
    assert '<ol style="margin-top: 4px; margin-bottom: 4px; padding-left: 20px;">' in html_output
    assert "<li>Step One</li>" in html_output
    assert "<li>Step Two</li>" in html_output
    assert "</ol>" in html_output

def test_render_markdown_to_html_escaping():
    text = "Click <a href='javascript:alert(1)'>here</a> and run **bold**"
    html_output = render_markdown_to_html(text)
    assert "&lt;a href=" in html_output
    assert "<strong>bold</strong>" in html_output
    assert "<a href" not in html_output

def test_render_markdown_to_html_newlines():
    text = "Line 1\nLine 2\n\nLine 3"
    html_output = render_markdown_to_html(text)
    assert "Line 1<br>Line 2<br><br>Line 3" in html_output

def test_render_markdown_to_html_headers():
    text = "# Header 1\n## Header 2\n### Header 3\n#### Header 4"
    html_output = render_markdown_to_html(text)
    assert "<h1 style=" in html_output
    assert ">Header 1</h1>" in html_output
    assert "<h2 style=" in html_output
    assert ">Header 2</h2>" in html_output
    assert "<h3 style=" in html_output
    assert ">Header 3</h3>" in html_output
    assert "<h4 style=" in html_output
    assert ">Header 4</h4>" in html_output
    assert "</h1><br>" not in html_output
