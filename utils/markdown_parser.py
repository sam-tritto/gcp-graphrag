import re
import html

def render_markdown_to_html(text: str) -> str:
    """
    Transforms basic markdown constructs (bold, italic, inline code, code blocks,
    lists, and headers) into safe, style-compatible HTML for rendering inside
    custom styled containers (like st.markdown(..., unsafe_allow_html=True)).
    
    Args:
        text (str): The raw markdown string.
        
    Returns:
        str: Styled and sanitized HTML string.
    """
    if not text:
        return ""
    
    # 1. HTML Escape to prevent raw HTML rendering or injection issues
    escaped = html.escape(text)
    
    # 2. Code blocks: ```[lang]\n[code]\n```
    def replace_code_block(match):
        code_content = match.group(2)
        # Preserve newlines and exact formatting inside <pre><code>
        return f'<pre style="background-color: #000000; padding: 12px; border-radius: 6px; overflow-x: auto; border: 1px solid #3c4043; margin: 8px 0;"><code style="font-family: \'Roboto Mono\', monospace; color: #e3e3e3; font-size: 0.85rem;">{code_content}</code></pre>'
    
    escaped = re.sub(r'```(\w*)\n([\s\S]*?)\n```', replace_code_block, escaped)
    
    # 3. Inline code: `code`
    escaped = re.sub(r'`([^`\n]+)`', r'<code style="font-family: \'Roboto Mono\', monospace; background-color: #000000; padding: 2px 6px; border-radius: 4px; color: #e3e3e3; font-size: 0.85rem;">\1</code>', escaped)
    
    # 4. Bold: **text** or __text__ (non-newline matching to avoid spanning lines)
    escaped = re.sub(r'\*\*([^\*\n]+)\*\*|__([^\_\n]+)__', lambda m: f'<strong>{m.group(1) or m.group(2)}</strong>', escaped)
    
    # 5. Italic: *text* or _text_ (non-newline matching, excluding double asterisks/underscores)
    escaped = re.sub(r'\*([^\*`\n]+)\*|_([^\_\n]+)_', lambda m: f'<em>{m.group(1) or m.group(2)}</em>', escaped)
    
    # 6. Parse lists (unordered and ordered) and headers line by line
    lines = escaped.split('\n')
    new_lines = []
    list_type = None  # None, 'ul', 'ol'
    
    for line in lines:
        stripped = line.strip()
        is_ul_item = stripped.startswith('* ') or stripped.startswith('- ')
        is_ol_item = bool(re.match(r'^\d+\.\s', stripped))
        
        # Automatically close lists if the current line does not belong to the list type
        if list_type == 'ul' and not is_ul_item:
            new_lines.append('</ul>')
            list_type = None
        elif list_type == 'ol' and not is_ol_item:
            new_lines.append('</ol>')
            list_type = None
            
        if is_ul_item:
            item_content = re.sub(r'^\s*[-*]\s+', '', line)
            if list_type is None:
                new_lines.append('<ul style="margin-top: 4px; margin-bottom: 4px; padding-left: 20px;">')
                list_type = 'ul'
            new_lines.append(f'<li>{item_content}</li>')
        elif is_ol_item:
            item_content = re.sub(r'^\s*\d+\.\s+', '', line)
            if list_type is None:
                new_lines.append('<ol style="margin-top: 4px; margin-bottom: 4px; padding-left: 20px;">')
                list_type = 'ol'
            new_lines.append(f'<li>{item_content}</li>')
        elif stripped.startswith('#'):
            # Header check: match up to 6 hashes at the beginning of the line
            header_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if header_match:
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                # Style headers with nice Google Console font sizes
                style = ""
                if level == 1:
                    style = 'style="font-size: 1.35rem; font-weight: 600; color: #ffffff; margin-top: 14px; margin-bottom: 8px; border-bottom: 1px solid #3c4043; padding-bottom: 4px;"'
                elif level == 2:
                    style = 'style="font-size: 1.2rem; font-weight: 600; color: #ffffff; margin-top: 12px; margin-bottom: 6px;"'
                elif level == 3:
                    style = 'style="font-size: 1.1rem; font-weight: 500; color: #ffffff; margin-top: 10px; margin-bottom: 4px;"'
                else:
                    style = 'style="font-size: 1.0rem; font-weight: 500; color: #ffffff; margin-top: 8px; margin-bottom: 4px;"'
                new_lines.append(f'<h{level} {style}>{header_text}</h{level}>')
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    if list_type == 'ul':
        new_lines.append('</ul>')
    elif list_type == 'ol':
        new_lines.append('</ol>')
        
    escaped = '\n'.join(new_lines)
    
    # 7. Convert line breaks (excluding code blocks, lists, and header tags)
    parts = re.split(r'(<pre[\s\S]*?</pre>)', escaped)
    for i in range(len(parts)):
        if not parts[i].startswith('<pre'):
            sub_lines = parts[i].split('\n')
            new_sub_lines = []
            for idx, sl in enumerate(sub_lines):
                sl_strip = sl.strip()
                if sl_strip.startswith('<ul') or sl_strip.startswith('</ul>') or \
                   sl_strip.startswith('<ol') or sl_strip.startswith('</ol>') or \
                   sl_strip.startswith('<li') or sl_strip.startswith('</li>') or \
                   re.match(r'^</?h\d', sl_strip):
                    new_sub_lines.append(sl)
                else:
                    if idx == len(sub_lines) - 1 and not sl:
                        new_sub_lines.append(sl)
                    else:
                        new_sub_lines.append(sl + '<br>')
            parts[i] = "".join(new_sub_lines)
            
    res_html = "".join(parts)
    while res_html.endswith('<br>'):
        res_html = res_html[:-4]
        
    return res_html
