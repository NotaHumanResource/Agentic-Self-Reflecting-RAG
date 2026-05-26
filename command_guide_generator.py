# command_guide_generator.py
"""Generate comprehensive command guide HTML."""

import os
from typing import List, Dict
from command_reference import COMMANDS, COMMAND_CATEGORIES

def generate_command_guide_html() -> str:
    """Generate a comprehensive HTML command guide."""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> AI Assistant - Command Reference</title>
        <style>
            {get_css_styles()}
        </style>
    </head>
    <body>
        <div class="container">
            {generate_header()}
            {generate_thinking_mode_section()}
            {generate_search_section()}
            {generate_categories_nav()}
            {generate_command_sections()}
            {generate_footer()}
        </div>
        <script>
            {get_javascript()}
        </script>
    </body>
    </html>
    """
    
    return html_content

def get_css_styles() -> str:
    """Return CSS styles for the command guide."""
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            margin-top: 20px;
            margin-bottom: 20px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 10px;
        }
        
        .thinking-mode-info {
            background: white;
            color: #1e3a8a;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
            border: 2px solid #3b82f6;
        }
        
        .thinking-mode-info h3 {
            margin-bottom: 10px;
            font-size: 20px;
            color: #1e40af;
        }
        
        .thinking-mode-info code {
            background: #dbeafe;
            color: #1e40af;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        
        .search-section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .search-box {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .categories-nav {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 30px;
            justify-content: center;
        }
        
        .category-button {
            padding: 10px 20px;
            background: #e9ecef;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .category-button:hover,
        .category-button.active {
            background: #007bff;
            color: white;
            transform: translateY(-2px);
        }
        
        .command-section {
            margin-bottom: 40px;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        
        .section-icon {
            font-size: 24px;
            margin-right: 15px;
        }
        
        .command-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .command-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        
        .command-header {
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #ddd;
        }
        
        .command-syntax {
            font-family: 'Courier New', monospace;
            background: #2d3748;
            color: #e2e8f0;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            display: inline-block;
            margin-bottom: 10px;
        }
        
        .command-body {
            padding: 15px;
        }
        
        .examples {
            background: #f1f3f4;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
        }
        
        .example-code {
            font-family: 'Courier New', monospace;
            background: #2d3748;
            color: #68d391;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 5px 0;
            display: block;
            font-size: 13px;
        }
        
        .parameters {
            margin-top: 15px;
        }
        
        .parameter {
            margin: 8px 0;
            padding-left: 20px;
        }
        
        .hidden {
            display: none;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 10px;
                padding: 15px;
            }
            
            .categories-nav {
                justify-content: flex-start;
            }
            
            .category-button {
                padding: 8px 15px;
                font-size: 14px;
            }
        }
    """

def generate_header() -> str:
    """Generate the header section."""
    return """
        <div class="header">
            <h1>🤖 AI Assistant</h1>
            <h2>Complete Command Reference Guide</h2>
            <p>Your comprehensive guide to memory commands, search operations, and AI interactions</p>
        </div>
    """

def generate_thinking_mode_section() -> str:
    """Generate the thinking mode controls section."""
    return """
        <div class="thinking-mode-info">
            <h3>💭 Thinking Mode Controls</h3>
            <p>
                <strong>Turn OFF thinking mode:</strong> Type <code>/no_think</code> at the start of your first message<br>
                <strong>Turn ON thinking mode:</strong> Type <code>/think</code> at the start of your message
            </p>
        </div>
    """

def generate_search_section() -> str:
    """Generate the search section."""
    return """
        <div class="search-section">
            <input type="text" class="search-box" id="commandSearch" 
                   placeholder="🔍 Search commands... (e.g. 'store', 'search', 'reminder')" 
                   onkeyup="filterCommands()">
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <button class="category-button" onclick="showAllCommands()">Show All</button>
                <button class="category-button" onclick="clearSearch()">Clear Search</button>
            </div>
        </div>
    """

def generate_categories_nav() -> str:
    """Generate the categories navigation."""
    nav_html = '<div class="categories-nav">'
    
    for category_id, category_info in COMMAND_CATEGORIES.items():
        nav_html += f'''
            <button class="category-button" onclick="filterByCategory('{category_id}')">
                {category_info['icon']} {category_info['name']}
            </button>
        '''
    
    nav_html += '</div>'
    return nav_html

def generate_command_sections() -> str:
    """Generate all command sections organized by category."""
    sections_html = ""
    
    for category_id, category_info in COMMAND_CATEGORIES.items():
        # Get commands for this category
        category_commands = [cmd for cmd in COMMANDS if cmd['category'] == category_id]
        
        if not category_commands:
            continue
            
        sections_html += f'''
            <div class="command-section" data-category="{category_id}">
                <div class="section-header">
                    <span class="section-icon">{category_info['icon']}</span>
                    <div>
                        <h2>{category_info['name']}</h2>
                        <p>{category_info['description']}</p>
                    </div>
                </div>
        '''
        
        for command in category_commands:
            sections_html += generate_command_card(command)
            
        sections_html += '</div>'
    
    return sections_html

def generate_command_card(command: Dict) -> str:
    """Generate a single command card."""
    card_html = f'''
        <div class="command-card" data-category="{command['category']}" data-searchable="{command['syntax']} {command['description']}">
            <div class="command-header">
                <div class="command-syntax">{command['syntax']}</div>
                <div class="command-description">{command['description']}</div>
            </div>
            <div class="command-body">
    '''
    
    # Add examples if they exist
    if command.get('examples'):
        card_html += '''
            <div class="examples">
                <h4>📋 Examples:</h4>
        '''
        for example in command['examples']:
            card_html += f'<code class="example-code">{example}</code>'
        card_html += '</div>'
    
    # Add parameters if they exist
    if command.get('parameters'):
        card_html += '''
            <div class="parameters">
                <h4>⚙️ Parameters:</h4>
        '''
        for param, description in command['parameters'].items():
            card_html += f'<div class="parameter"><strong>{param}:</strong> {description}</div>'
        card_html += '</div>'
    
    # Add filters if they exist (for search commands)
    if command.get('filters'):
        card_html += '''
            <div class="parameters">
                <h4>🔎 Available Filters:</h4>
        '''
        for filter_name, description in command['filters'].items():
            card_html += f'<div class="parameter"><strong>{filter_name}:</strong> {description}</div>'
        card_html += '</div>'
    
    # Add tips if they exist
    if command.get('tips'):
        card_html += f'''
            <div class="examples">
                <h4>💡 Tips:</h4>
                <p>{command['tips']}</p>
            </div>
        '''
    
    card_html += '''
            </div>
        </div>
    '''
    
    return card_html

def generate_footer() -> str:
    """Generate the footer section."""
    return """
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <p><strong>🚀 AI Assistant</strong> - Built with ❤️ for enhanced human-AI collaboration</p>
            <p style="margin-top: 10px; color: #666;">
                For more information or support contact <a href="mailto:nonbiologicalintelligence@gmail.com">nonbiologicalintelligence@gmail.com</a>
            </p>
        </div>
    """

def get_javascript() -> str:
    """Return JavaScript for interactive functionality."""
    return """
        function filterCommands() {
            const searchTerm = document.getElementById('commandSearch').value.toLowerCase();
            const commands = document.querySelectorAll('.command-card');
            
            commands.forEach(command => {
                const searchableText = command.getAttribute('data-searchable').toLowerCase();
                if (searchableText.includes(searchTerm)) {
                    command.style.display = 'block';
                } else {
                    command.style.display = 'none';
                }
            });
            
            // Show/hide sections based on visible commands
            const sections = document.querySelectorAll('.command-section');
            sections.forEach(section => {
                const visibleCommands = section.querySelectorAll('.command-card[style*="block"], .command-card:not([style*="none"])');
                section.style.display = visibleCommands.length > 0 ? 'block' : 'none';
            });
        }
        
        function filterByCategory(categoryId) {
            // Reset search box
            document.getElementById('commandSearch').value = '';
            
            const sections = document.querySelectorAll('.command-section');
            const commands = document.querySelectorAll('.command-card');
            
            // Show all commands first
            commands.forEach(command => {
                command.style.display = 'block';
            });
            
            // Show all sections first
            sections.forEach(section => {
                section.style.display = 'block';
            });
            
            // If specific category selected, hide others
            if (categoryId !== 'all') {
                sections.forEach(section => {
                    if (section.getAttribute('data-category') !== categoryId) {
                        section.style.display = 'none';
                    }
                });
            }
            
            // Update active button
            const buttons = document.querySelectorAll('.category-button');
            buttons.forEach(button => {
                button.classList.remove('active');
            });
            event.target.classList.add('active');
        }
        
        function showAllCommands() {
            const sections = document.querySelectorAll('.command-section');
            const commands = document.querySelectorAll('.command-card');
            
            sections.forEach(section => {
                section.style.display = 'block';
            });
            
            commands.forEach(command => {
                command.style.display = 'block';
            });
            
            // Clear active buttons
            const buttons = document.querySelectorAll('.category-button');
            buttons.forEach(button => {
                button.classList.remove('active');
            });
            
            // Clear search
            document.getElementById('commandSearch').value = '';
        }
        
        function clearSearch() {
            document.getElementById('commandSearch').value = '';
            filterCommands();
        }
    """

def save_command_guide_html():
    """Save the command guide as an HTML file."""
    html_content = generate_command_guide_html()
    
    # Create guides directory if it doesn't exist
    guides_dir = os.path.join(os.path.dirname(__file__), "guides")
    os.makedirs(guides_dir, exist_ok=True)
    
    file_path = os.path.join(guides_dir, "command_reference.html")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return file_path
