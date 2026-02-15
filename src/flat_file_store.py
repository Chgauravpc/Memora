"""
Flat File Storage - Core Memory Layer
Handles reading/writing human-editable Markdown files
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from .config import MEMORY_DIR, CORE_MEMORY_FILES

logger = logging.getLogger(__name__)


class FlatFileStore:
    """Manages flat file storage for Core Memory (always-injected memories)"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_dir = MEMORY_DIR / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_files()

    def _initialize_files(self):
        """Create core memory files if they don't exist"""
        # Define templates with proper section headers for promotion
        templates = {
            "CORE.md": '''# Core Identity

**Last Updated:** {date}

## Identity

''',
            "PREFERENCES.md": '''# User Preferences

**Last Updated:** {date}

## General Preferences

''',
            "INSTRUCTIONS.md": '''# Instructions & Guidelines

**Last Updated:** {date}

## Communication Style

''',
            "CONSTRAINTS.md": '''# Constraints & Limitations

**Last Updated:** {date}

## Hard Constraints

''',
        }
        
        for filename in CORE_MEMORY_FILES:
            filepath = self.user_dir / filename
            if not filepath.exists():
                logger.info(f"Initializing {filename} for user {self.user_id}")
                template = templates.get(filename, "")
                if template:
                    content = template.format(date=datetime.now().strftime("%Y-%m-%d"))
                    filepath.write_text(content, encoding='utf-8')
                else:
                    filepath.touch()

    def read_core_memory(self) -> str:
        """
        Read all core memory files and concatenate them.
        This is ALWAYS injected into the prompt.
        
        Returns:
            Combined text from all core memory files
        """
        core_content = []
        
        for filename in CORE_MEMORY_FILES:
            filepath = self.user_dir / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(encoding='utf-8')
                    if content.strip():  # Only include non-empty files
                        core_content.append(f"=== {filename} ===\n{content}\n")
                except Exception as e:
                    logger.error(f"Error reading {filename}: {e}")
        
        return "\n".join(core_content)

    def update_core_field(self, file: str, section: str, field: str, value: str):
        """
        Update a specific field in a core memory file.
        Used when high-confidence updates to core identity are needed.
        
        Args:
            file: Which core file (e.g., "CORE.md")
            section: Section header (e.g., "Identity")
            field: Field name (e.g., "Name")
            value: New value
        """
        filepath = self.user_dir / file
        if not filepath.exists():
            logger.warning(f"File {file} does not exist")
            return

        try:
            content = filepath.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Find the section and field
            in_section = False
            updated = False
            
            for i, line in enumerate(lines):
                # Check if we're entering the target section
                if line.strip().startswith('##') and section in line:
                    in_section = True
                    continue
                
                # Check if we've left the section
                if in_section and line.strip().startswith('##'):
                    in_section = False
                
                # Update the field if we're in the right section
                if in_section and line.strip().startswith(f'- **{field}:**'):
                    lines[i] = f'- **{field}:** {value}'
                    updated = True
                    break
            
            if updated:
                # Update the "Last Updated" timestamp
                for i, line in enumerate(lines):
                    if line.strip().startswith('**Last Updated:**'):
                        lines[i] = f'**Last Updated:** {datetime.now().strftime("%Y-%m-%d")}'
                        break
                
                filepath.write_text('\n'.join(lines), encoding='utf-8')
                logger.info(f"Updated {file} -> {section} -> {field} = {value}")
            else:
                logger.warning(f"Field {field} not found in {section} of {file}")
                
        except Exception as e:
            logger.error(f"Error updating {file}: {e}")

    def append_to_section(self, file: str, section: str, content: str):
        """
        Append content to a section in a core memory file.
        
        Args:
            file: Which core file
            section: Section header
            content: Content to append
        """
        filepath = self.user_dir / file
        if not filepath.exists():
            logger.warning(f"File {file} does not exist")
            return

        try:
            file_content = filepath.read_text(encoding='utf-8')
            lines = file_content.split('\n')
            
            in_section = False
            insert_index = None
            
            for i, line in enumerate(lines):
                if line.strip().startswith('##') and section in line:
                    in_section = True
                    insert_index = i + 1
                    continue
                
                if in_section and line.strip().startswith('##'):
                    insert_index = i
                    break
            
            if insert_index is not None:
                lines.insert(insert_index, f'- {content}')
                filepath.write_text('\n'.join(lines), encoding='utf-8')
                logger.info(f"Appended to {file} -> {section}: {content}")
            else:
                logger.warning(f"Section {section} not found in {file}")
                
        except Exception as e:
            logger.error(f"Error appending to {file}: {e}")

    def get_file_path(self, filename: str) -> Path:
        """Get the full path to a memory file"""
        return self.user_dir / filename
