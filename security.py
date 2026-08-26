import logging
# this file is to handle security related questions, such as sanitizing agents' inputs, and preventing malicious code execution.
# the script's input is the strings/files read by the agents, and the output is the sanitized strings/files that are safe to use in the script.
# this is a prevention against malicious code execution, and it is not a replacement for proper security practices in the script itself.
# this prevention includes string sanitization, xss prevention, but it does not include sql injection prevention, as the script does not use a database, and this is not a full covered prevention.

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
logger.propagate = False

def sanitize_string(input_string: str) -> str:
    """
    Sanitizes the input string to prevent malicious code execution.
    This function removes potentially dangerous characters and patterns.
    """
    sanitized = input_string.replace("<script>", "").replace("</script>", "")
    sanitized = sanitized.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return sanitized

def sanitize_file_content(file_content: str) -> str:
    """
    Sanitizes the content of a file to prevent malicious code execution.
    This function removes potentially dangerous characters and patterns.
    """
    sanitized = file_content.replace("<script>", "").replace("</script>", "")
    sanitized = sanitized.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return sanitized

def cover_external_links_with_nofollow(file_content: str) -> str:
    """
    Covers external links in the file content with rel="nofollow" to prevent SEO manipulation.
    This function adds rel="nofollow" to all external links.
    """
    import re
    pattern = r'<a\s+(?!.*?rel=["\']nofollow["\'])([^>]*?)href=["\'](http[s]?://[^"\']+)["\']([^>]*)>'
    replacement = r'<a \1href="\2" rel="nofollow"\3>'
    sanitized = re.sub(pattern, replacement, file_content, flags=re.IGNORECASE)
    return sanitized

def cover_external_input_with_tags(input_string: str) -> str:
    """
    Covers external input with <external> tags to prevent malicious code execution.
    This function wraps the input string with <external> tags.
    """
    return f"<external>{input_string}</external>"

def sanitize_xss(input_string: str) -> str:
    """
    Sanitizes the input string to prevent XSS attacks.
    This function removes potentially dangerous characters and patterns.
    """
    sanitized = input_string.replace("<", "&lt;").replace(">", "&gt;")
    sanitized = sanitized.replace('"', "&quot;").replace("'", "&#x27;")
    sanitized = sanitized.replace("/", "&#x2F;")
    return sanitized

def sanitize_shell_and_powershell_commands(input_string: str) -> str:
    """
    Sanitizes the input string to prevent shell and PowerShell command injection.
    This function removes potentially dangerous characters and patterns.
    """
    sanitized = input_string.replace(";", "").replace("&", "").replace("|", "")
    sanitized = sanitized.replace("`", "").replace("$", "").replace(">", "")
    sanitized = sanitized.replace("<", "").replace("(", "").replace(")", "")
    return sanitized

def filter_out_prompt_injection(input_string: str) -> str:
    """
    Filters out prompt injection attempts from the input string.
    This function removes potentially dangerous patterns that could be used for prompt injection.
    """
    import re
    # Remove common prompt injection patterns
    patterns = [
        r'(?i)system\(',  # system() calls
        r'(?i)exec\(',    # exec() calls
        r'(?i)eval\(',    # eval() calls
        r'(?i)import\s+os',  # import os
        r'(?i)import\s+subprocess',  # import subprocess
        r'(?i)from\s+os',  # from os
        r'(?i)from\s+subprocess',  # from subprocess
        r'(?i)open\(',     # open() calls
        r'(?i)input\(',    # input() calls
        r'(?i)exit\(',     # exit() calls
        r'(?i)quit\(',     # quit() calls
        r'(?i)import\s+sys',  # import sys
        r'(?i)ignore\s+previous\s+instructions'  # ignore previous instructions
    ]
    for pattern in patterns:
        input_string = re.sub(pattern, "", input_string)
    return input_string

def main_security_sanitization(input_string: str) -> str:
    """
    Main function to sanitize the input string for security.
    This function applies all sanitization methods to the input string.
    """
    sanitized = cover_external_links_with_nofollow(input_string)
    sanitized = filter_out_prompt_injection(sanitized)
    sanitized = sanitize_string(sanitized)
    sanitized = sanitize_xss(sanitized)
    sanitized = sanitize_shell_and_powershell_commands(sanitized)
    sanitized = cover_external_input_with_tags(sanitized)
    
    logger.info("[main_security_sanitization] sanitized content: %s", sanitized)
    return sanitized