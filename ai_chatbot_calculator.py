from transformers import pipeline
import re
from word2number import w2n
import sympy as sp
import math
import string
from pint import UnitRegistry
from datetime import datetime, timedelta
try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None 
import difflib
from textblob import TextBlob
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Add this mapping near the top of your file
display_func_names = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "sin⁻¹",
    "acos": "cos⁻¹",
    "atan": "tan⁻¹",
    # Add others as needed
}

# Place this near the top of your file, after imports
temp_unit_symbols = {
    "degC": "°C",
    "celsius": "°C",
    "centigrade": "°C",
    "degF": "°F",
    "fahrenheit": "°F",
    "kelvin": "K",
    "k": "K"
}

def pretty_temp_unit(unit):
    """
    Normalize and format temperature unit names for output with correct capitalization and symbols.
    For Celsius: always output '°C' (degree sign and capital C).
    For Fahrenheit: always output '°F' (degree sign and capital F).
    For Kelvin: always output 'K'.
    """
    unit_lc = unit.strip().lower().replace("degree ", "").replace("degrees ", "")
    celsius_aliases = {"c", "celsius", "centigrade", "degc"}
    fahrenheit_aliases = {"f", "fahrenheit", "degf"}
    kelvin_aliases = {"k", "kelvin"}
    if unit_lc in celsius_aliases:
        return "°C"
    elif unit_lc in fahrenheit_aliases:
        return "°F"
    elif unit_lc in kelvin_aliases:
        return "K"
    else:
        # Default fallback (existing behavior)
        return temp_unit_symbols.get(unit_lc, temp_unit_symbols.get(unit_lc.strip(), unit))

def normalize_power_phrases(expr, var=None):
    """
    Normalizes phrases like 'x squared', 'x cubed', '2x squared', etc. to 'x**2', 'x**3', '2*x**2', etc.
    Optionally, restricts normalization to a specific variable.
    """
    # Handle e.g. '2x squared' or '2 x squared' -> '2*x**2'
    if var:
        expr = re.sub(rf"([\-]?\d*\.?\d*)\s*{var}\s*squared", rf"\1*{var}**2", expr, flags=re.IGNORECASE)
        expr = re.sub(rf"([\-]?\d*\.?\d*)\s*{var}\s*cubed", rf"\1*{var}**3", expr, flags=re.IGNORECASE)
        expr = re.sub(rf"\b{var}\s*squared\b", f"{var}**2", expr, flags=re.IGNORECASE)
        expr = re.sub(rf"\b{var}\s*cubed\b", f"{var}**3", expr, flags=re.IGNORECASE)
    else:
        # Generic: any variable
        expr = re.sub(r"([a-zA-Z]\w*)\s*squared", r"\1**2", expr, flags=re.IGNORECASE)
        expr = re.sub(r"([a-zA-Z]\w*)\s*cubed", r"\1**3", expr, flags=re.IGNORECASE)
        expr = re.sub(r"([\-]?\d*\.?\d*)\s*([a-zA-Z]\w*)\s*squared", r"\1*\2**2", expr, flags=re.IGNORECASE)
        expr = re.sub(r"([\-]?\d*\.?\d*)\s*([a-zA-Z]\w*)\s*cubed", r"\1*\2**3", expr, flags=re.IGNORECASE)
    # Handle 'to the power of'
    expr = re.sub(r"([a-zA-Z]\w*)\s*to the power of\s*([\-]?\d+)", r"\1**\2", expr, flags=re.IGNORECASE)
    expr = re.sub(r"([\-]?\d*\.?\d*)\s*([a-zA-Z]\w*)\s*to the power of\s*([\-]?\d+)", r"\1*\2**\3", expr, flags=re.IGNORECASE)
    return expr

VALID_FUNCTIONS = [
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "log", "ln", "sqrt", "exp", "abs", "factorial"
]

def preprocess(expression):
    # --- PATCH: Preprocess NL math phrases before tokenization ---
    expression = strip_leading_phrases(expression)
    expression = preprocess_math_phrases(expression)  # <-- Add this line early!
    token_pattern = r'([a-zA-Z_][a-zA-Z_0-9]*|\d+\.\d+|\d+|[^\s])'
    tokens = re.findall(token_pattern, expression)
    corrected_tokens = []
    for token in tokens:
        # Only correct tokens that are not variables or numbers
        if re.match(r'^[a-zA-Z_][a-zA-Z_0-9]*$', token) or re.match(r'^\d+(\.\d+)?$', token):
            # Do not correct variables or numbers
            corrected_tokens.append(token)
        elif token.lower() in VALID_FUNCTIONS:
            corrected_tokens.append(token)
        else:
            corrected, _ = correct_math_typos(token)
            corrected_tokens.append(str(corrected))
    expression = ''.join(corrected_tokens)
    expression = strip_math_command_words(expression)
    expression = extract_math_expression(expression)
    return expression.strip()

# --- Improved replace_log_with_base to handle nested parentheses and multiple logs ---
def replace_log_with_base(text):
    """
    Replaces log with subscript base notation (e.g., log₁₀(100), log₂(8)) with SymPy-compatible log(x, base).
    Handles Unicode subscripts and also log_b(x) notation.
    Supports nested parentheses and multiple log expressions.
    """
    # Unicode subscript mapping
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    # Helper to find matching parenthesis
    def find_matching_paren(s, start):
        stack = 0
        for i in range(start, len(s)):
            if s[i] == '(':
                stack += 1
            elif s[i] == ')':
                stack -= 1
                if stack == 0:
                    return i
        return -1

    # Replace all logₓ(...) with log(..., x)
    i = 0
    out = ""
    while i < len(text):
        m = re.match(r"log([₀₁₂₃₄₅₆₇₈₉]+)\(", text[i:])
        if m:
            base = m.group(1).translate(subscript_map)
            start = i + m.end() - 1  # position of '('
            end = find_matching_paren(text, start)
            if end == -1:
                # No matching parenthesis, treat as normal text
                out += text[i]
                i += 1
                continue
            arg = text[start+1:end]
            out += f"log({arg}, {base})"
            i = end + 1
        else:
            out += text[i]
            i += 1

    # Replace log_b(...) with log(..., b)
    # This regex is safe because _ is not a valid operator
    out = re.sub(r"log[_]([0-9]+)\(([^()]+)\)", r"log(\2, \1)", out)
    return out

# PATCH: In correct_math_typos, do not correct 'subtracted from' or 'subtract from' as a phrase
def correct_math_typos(text):
    """
    Corrects common math command typos in user input using fuzzy matching.
    Returns the corrected text and a flag indicating if a correction was made.
    PATCH: Do not correct 'subtracted from' or 'subtract from' as a phrase.
    """
    # List of math commands to correct
    math_commands = [
        "multiply", "add", "subtract", "divide", "plus", "minus", "times", "factorial",
        "simplify", "expand", "factor", "evaluate", "calculate", "find", "compute", "solve"
    ]
    # Words that should never be corrected (they are already valid math operators or units)
    skip_correction = {
        "plus", "minus", "add", "subtract", "multiply", "divide", "times", "and", "or",
        # Add unit words to skip list
        "minute", "minutes", "hour", "hours", "second", "seconds",
        "day", "days", "week", "weeks", "month", "months", "year", "years"
    }

    # PATCH: Do not correct 'subtracted from' or 'subtract from' as a phrase
    text = re.sub(r'\b(subtracted from|subtract from)\b', lambda m: m.group(1), text, flags=re.IGNORECASE)

    words = text.split()
    corrected_words = []
    was_corrected = False
    for word in words:
        # Only try to correct words that are not numbers or operators or units
        if not word.isalpha():
            corrected_words.append(word)
            continue
        # If the word is already correct or should be skipped, keep it
        if word.lower() in math_commands or word.lower() in skip_correction:
            corrected_words.append(word)
            continue
        # Fuzzy match to math commands
        matches = difflib.get_close_matches(word.lower(), math_commands, n=1, cutoff=0.8)
        if matches:
            corrected_words.append(matches[0])
            was_corrected = True
        else:
            corrected_words.append(word)
    corrected_text = " ".join(corrected_words)
    return corrected_text, was_corrected

# --- PATCH: Add this helper near the top ---
def is_function_name(expr):
    """Check if the input is just a function name (e.g., 'sin', 'log'), not a constant like 'pi'."""
    expr = expr.strip()
    func = allowed_functions.get(expr)
    return callable(func) 

# Add this helper function near the top of your file (after imports)
def normalize_math_constants(text):
    """
    Replace Unicode math constants with their SymPy equivalents.
    E.g., π -> pi, ℯ -> E, etc.
    """
    replacements = {
        "π": "pi",
        "ℯ": "E",
        "𝑒": "E",
        "𝜋": "pi",
        "𝛑": "pi",
        # Add more as needed
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# --- PATCH: Add digit strings to NUM_WORDS ---
NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000
}
# Add digit strings to NUM_WORDS
for i in range(0, 10001):
    NUM_WORDS[str(i)] = i

# Add ordinal words to NUM_WORDS
ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
    "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50, "sixtieth": 60,
    "seventieth": 70, "eightieth": 80, "ninetieth": 90, "hundredth": 100, "thousandth": 1000
}
NUM_WORDS.update(ORDINAL_WORDS)

def words_to_number(text):
    """
    Converts a string of number words (e.g., 'ten thousand', '2 thousand') to its integer value (e.g., 10000, 2000).
    Handles numbers up to millions and digit+word combos.
    """
    text = text.lower().replace('-', ' ')
    current = result = 0
    for word in text.split():
        # Accept digit strings as numbers
        if word.isdigit():
            scale = int(word)
        elif word in NUM_WORDS:
            scale = NUM_WORDS[word]
        else:
            return None
        if scale in (100, 1000, 1000000):
            if current == 0:
                current = 1
            current *= scale
            result += current
            current = 0
        else:
            current += scale
    return result + current

def words_phrase_to_number(text):
    """
    Converts a phrase like 'five and a half', 'seven point two', 'three and three quarters'
    to its float value (e.g., 5.5, 7.2, 3.75).
    Handles:
      - 'and a half', 'and a quarter', 'and three quarters'
      - 'point two', 'point five six'
      - 'and X/Y' (e.g., 'and 3/4')
    """
    text = text.lower().replace('-', ' ')
    # Handle 'point' for decimals
    if 'point' in text:
        parts = text.split('point')
        int_part = words_to_number(parts[0].strip())
        if int_part is None:
            return None
        dec_words = parts[1].strip().split()
        dec_digits = []
        for w in dec_words:
            if w in NUM_WORDS:
                dec_digits.append(str(NUM_WORDS[w]))
            elif w.isdigit():
                dec_digits.append(w)
            else:
                # Unknown word, fallback
                return None
        dec_str = ''.join(dec_digits)
        try:
            return float(f"{int_part}.{dec_str}")
        except Exception:
            return None

    # Handle 'and a half', 'and a quarter', 'and three quarters', 'and X/Y'
    m = re.match(r'(.+?)\s+and\s+(a|an|\d+|[a-z]+)?\s*(half|quarter|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|[0-9]+/[0-9]+|[a-z]+)?s?$', text)
    if m:
        int_part = words_to_number(m.group(1).strip())
        if int_part is None:
            return None
        num_word = m.group(2)
        frac_word = m.group(3)
        # Map common fractions
        frac_map = {
            "half": 0.5,
            "quarter": 0.25,
            "third": 1/3,
            "fourth": 0.25,
            "fifth": 0.2,
            "sixth": 1/6,
            "seventh": 1/7,
            "eighth": 0.125,
            "ninth": 1/9,
            "tenth": 0.1,
        }
        if frac_word in frac_map:
            frac_val = frac_map[frac_word]
            if num_word in ["a", "an", None]:
                return int_part + frac_val
            elif num_word is not None and num_word.isdigit():
                return int_part + int(num_word) * frac_val
            elif num_word in NUM_WORDS:
                return int_part + NUM_WORDS[num_word] * frac_val
        elif frac_word and '/' in frac_word:
            # e.g., 'and 3/4'
            try:
                num, denom = frac_word.split('/')
                return int_part + float(num) / float(denom)
            except Exception:
                return None
        elif frac_word in NUM_WORDS:
            # e.g., 'and three'
            return int_part + NUM_WORDS[frac_word]
        else:
            return None

    # Fallback to normal number word parsing
    return words_to_number(text)

# --- PATCH: Fix replace_number_words to not consume operator words ---
def replace_number_words(text):
    """
    Replaces all number word phrases in the text with their digit equivalents.
    Handles multi-word numbers greedily (e.g., 'ten thousand' -> '10000', '2 thousand' -> '2000').
    Handles 'five and a half', 'seven point two', etc.
    Handles hyphenated number words like 'twenty-two'.
    Uses a greedy sliding window to match the longest possible number word phrase.
    PATCH: Do not treat operator words as part of number phrases.
    """
    # PATCH: Do not treat "subtracted from" or "subtract from" as a number phrase
    if re.search(r'\bsubtracted from\b', text) or re.search(r'\bsubtract from\b', text):
        return text

    # Replace hyphens between number words with spaces for easier parsing
    text = re.sub(r'([a-zA-Z]+)-([a-zA-Z]+)', r'\1 \2', text)
    words = text.split()
    n = len(words)
    i = 0
    out = []
    operator_words = {"plus", "minus", "add", "subtract", "times", "multiplied", "divided", "by", "over", "and"}
    while i < n:
        # Try to match the longest possible number phrase starting at position i
        max_j = None
        max_val = None
        # Only consider up to 7 words ahead (arbitrary, but covers most cases)
        for j in range(min(n, i+7), i, -1):
            phrase = ' '.join(words[i:j])
            # PATCH: If phrase contains operator words, break early
            if any(op in phrase for op in operator_words):
                continue
            val = words_phrase_to_number(phrase)
            if val is not None:
                max_j = j
                max_val = val
                break  # Greedy: take the longest match
        if max_j is not None:
            # Remove trailing .0 for integers
            if isinstance(max_val, float) and max_val.is_integer():
                out.append(str(int(max_val)))
            else:
                out.append(str(max_val))
            i = max_j
        else:
            out.append(words[i])
            i += 1
    return ' '.join(out)

def enhanced_nl_math_parser(text):
    """
    Handles a wide variety of NL math queries, including:
    - Add ten and twenty-three
    - Subtract X from Y
    - X subtracted from Y
    Returns a tuple (expression, result) or (None, None) if not matched.
    PATCH: Ensure subtraction is handled as subtraction, not addition, and only parse if the match covers the *entire* input and there are no extra operations (to preserve operator precedence).
    PATCH (2024-06-09): Only match these patterns if they consume the *entire* input; otherwise fall through.
    PATCH (2024-06-14): If input contains multiple operators (e.g. 'plus' and 'multiplied by'), skip shortcut matching so that precedence is handled correctly.
    """
    orig_text = text
    text = text.lower().strip().replace('?', '')
    # Replace number words with digits for robust matching
    text_numwords = replace_number_words(text)
    orig_text_numwords = replace_number_words(orig_text.lower())

    # If input contains more than one operator word: skip shortcut patterns
    operator_words = ["plus", "minus", "add", "subtract", "times", "multiplied", "divided", "by", "over", "and"]
    operator_count = sum(text.count(op) for op in operator_words)
    # 'and' is ambiguous; only count if between numbers
    op_hit = 0
    for op in operator_words:
        if op == "and":
            continue
        if re.search(rf"\b{op}\b", text):
            op_hit += 1
    if any(op in text for op in ["plus", "minus", "add", "subtract", "times", "multiplied", "divided", "over"]):
        ops_present = [op for op in operator_words if op in text]
        if len(ops_present) > 1:
            return (None, None)

    # --- PATCH: Only match if the pattern matches the *entire* input ---
    patterns = [
        (r'^([a-zA-Z0-9\s\.\-]+)\s*subtracted from\s*([a-zA-Z0-9\s\.\-]+)$', lambda x, y: f"{y} - {x}"),
        (r'^subtract\s*([a-zA-Z0-9\s\.\-]+)\s*from\s*([a-zA-Z0-9\s\.\-]+)$', lambda x, y: f"{y} - {x}"),
        (r'^add\s*([a-zA-Z0-9\s\.\-]+)\s*to\s*([a-zA-Z0-9\s\.\-]+)$', lambda x, y: f"{y} + {x}"),
        (r'^multiply\s*([a-zA-Z0-9\s\.\-]+)\s*by\s*([a-zA-Z0-9\s\.\-]+)$', lambda x, y: f"{x} * {y}"),
        (r'^divide\s*([a-zA-Z0-9\s\.\-]+)\s*by\s*([a-zA-Z0-9\s\.\-]+)$', lambda x, y: f"{x} / {y}"),
    ]
    # PATCH: Only attempt these if there's just one operator and the input does not contain other operator words (for precedence safety)
    op_pattern = re.compile(r'\b(plus|minus|add|subtract|times|multiplied|divided|over)\b')
    if len(op_pattern.findall(text)) > 1:
        return (None, None)

    for pat, expr_builder in patterns:
        m = re.fullmatch(pat, text)
        if m:
            x = replace_number_words(m.group(1).strip())
            y = replace_number_words(m.group(2).strip())
            try:
                x_val = float(x)
                y_val = float(y)
                expr = expr_builder(x_val, y_val)
                try:
                    result = eval(expr)
                except Exception:
                    result = None
                return expr, result
            except Exception:
                expr = expr_builder(x, y)
                # PATCH START: re-run replace_number_words on built expression
                expr_for_eval = replace_number_words(expr)
                try:
                    result = eval(expr_for_eval)
                    # For output clarity, show the final numeric expr if translated
                    return expr_for_eval, result
                except Exception:
                    result = None
                    return expr, result
                # PATCH END
                try:
                    result = eval(expr)
                except Exception:
                    result = None
                return expr, result

    # Existing patterns...
    # --- PATCH: Handle "subtract X by Y" robustly for number words and digits (floats supported) ---
    m = re.match(r'subtract\s+([\d\.]+)\s+by\s+([\d\.]+)', text_numwords)
    if m:
        left = float(m.group(1))
        right = float(m.group(2))
        expr = f"{left} - {right}"
        return expr, eval(expr)
    m = re.match(r'subtract\s+([\d\.]+)\s+by\s+([\d\.]+)', orig_text_numwords)
    if m:
        left = float(m.group(1))
        right = float(m.group(2))
        expr = f"{left} - {right}"
        return expr, eval(expr)

    # --- PATCH: Handle "subtract X from Y" robustly (floats supported) ---
    m = re.match(r'(subtract|minus)\s*([\d\.]+)\s*from\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(3)} - {m.group(2)}"
        return expr, eval(expr)
    m = re.match(r'what do i get if i subtract\s*([\d\.]+)\s*from\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(2)} - {m.group(1)}"
        return expr, eval(expr)

    # Addition: "add X and Y" or "X plus Y" (floats supported)
    m = re.match(r'(add|sum|plus)\s*([\d\.]+)\s*(and|to)?\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(2)} + {m.group(4)}"
        return expr, eval(expr)
    m = re.match(r'([\d\.]+)\s*(plus|add|added to)\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(1)} + {m.group(3)}"
        return expr, eval(expr)

    # Subtraction: "subtract X and Y" (X - Y), "X minus Y" (floats supported)
    m = re.match(r'(subtract|minus)\s*([\d\.]+)\s*(and)?\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(2)} - {m.group(4)}"
        return expr, eval(expr)
    m = re.match(r'([\d\.]+)\s*(minus|subtract|subtracted by|less)\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(1)} - {m.group(3)}"
        return expr, eval(expr)

    # Multiplication: "multiply X and Y", "multiply X by Y", "X times Y" (floats supported)
    m = re.match(r'(multiply|times|product)\s*([\d\.]+)\s*(and|by)?\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(2)} * {m.group(4)}"
        return expr, eval(expr)
    m = re.match(r'([\d\.]+)\s*(times|multiply|multiplied by)\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(1)} * {m.group(3)}"
        return expr, eval(expr)

    # Division: "divide X and Y", "divide X by Y", "X divided by Y" (floats supported)
    m = re.match(r'(divide|over|quotient)\s*([\d\.]+)\s*(and|by)?\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(2)} / {m.group(4)}"
        return expr, eval(expr)
    m = re.match(r'([\d\.]+)\s*(divided by|divide|over)\s*([\d\.]+)', text_numwords)
    if m:
        expr = f"{m.group(1)} / {m.group(3)}"
        return expr, eval(expr)

    # Percentage: "what's X percent of Y"
    m = re.match(r'(what(\'s| is)? )?([\d\.]+) percent of ([\d\.]+)', text_numwords)
    if m:
        percent = float(m.group(3))
        base = float(m.group(4))
        result = (percent / 100) * base
        expr = f"{percent}% of {base}"
        return expr, result

    # Double/Half
    m = re.match(r'(double|twice) ([\d\.]+)', text_numwords)
    if m:
        val = float(m.group(2))
        return f"2 * {val}", 2 * val
    m = re.match(r'half of ([\d\.]+)', text_numwords)
    if m:
        val = float(m.group(1))
        return f"{val} / 2", val / 2

    # Leap year
    m = re.match(r'is ([\d]{4}) a leap year', text_numwords)
    if m:
        year = int(m.group(1))
        is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        return f"Leap year check for {year}", "Yes" if is_leap else "No"

    # General arithmetic: "X plus Y", "X minus Y", etc. (floats supported)
    m = re.match(r'([\d\.]+)\s*(plus|minus|times|multiplied by|divided by|over)\s*([\d\.]+)', text_numwords)
    if m:
        op = m.group(2)
        op_map = {
            "plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "divided by": "/", "over": "/"
        }
        expr = f"{m.group(1)} {op_map[op]} {m.group(3)}"
        return expr, eval(expr)

    # Fallback: try to extract a simple math expression
    if re.match(r'^[\d\s\+\-\*/\.]+$', text_numwords):
        try:
            return text_numwords, eval(text_numwords)
        except Exception:
            pass

    return None, None

def handle_date_difference(query):
    """
    Handles queries like:
    - 'How many days are between Jan 1, 2022 and March 1, 2022?'
    - 'Days between 2022-01-01 and 2022-03-01'
    - 'Difference between 1/1/2022 and 3/1/2022'
    Returns a string with the result, or None if not matched.
    """
    q = query.lower().strip()
    # Look for "days between ... and ..." or "difference between ... and ..."
    pat = r'(?:how many\s+)?days\s+(?:are\s+)?(?:between|from)\s+([a-zA-Z0-9,/\-\s]+?)\s+(?:and|to)\s+([a-zA-Z0-9,/\-\s]+)\??'
    m = re.search(pat, q)
    if not m:
        # Try "difference between ... and ..."
        pat2 = r'difference\s+(?:between|from)\s+([a-zA-Z0-9,/\-\s]+?)\s+(?:and|to)\s+([a-zA-Z0-9,/\-\s]+)\??'
        m = re.search(pat2, q)
    if not m:
        return None
    date1_str, date2_str = m.group(1).strip(), m.group(2).strip()
    # Try to parse dates
    def parse_date(s):
        if date_parser:
            try:
                return date_parser.parse(s, fuzzy=True)
            except Exception:
                pass
        # Fallback: try common formats
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None
    d1 = parse_date(date1_str)
    d2 = parse_date(date2_str)
    if not d1 or not d2:
        return f"Sorry, I couldn't parse one of the dates: '{date1_str}' or '{date2_str}'."
    delta = abs((d2 - d1).days)
    return f"There are {delta} days between {d1.strftime('%Y-%m-%d')} and {d2.strftime('%Y-%m-%d')}."

# Initialize pint's unit registry (add at the top-level, not inside a function)
ureg = UnitRegistry()
Q_ = ureg.Quantity

def pluralize_unit(unit: str, value: float):
    """
    Pluralize the unit for correct English output.
    - E.g., pluralize_unit("minute", 1) -> "minute"
    - pluralize_unit("minute", 2) -> "minutes"
    - Doesn't pluralize abbreviations or units that shouldn't be pluralized.
    PATCH: Fixes 'inch' to 'inches' not 'inchs'.
    """
    # Don't pluralize these
    dont_pluralize = {
        "celsius", "fahrenheit", "kelvin", "k", "hz", "kg", "g", "lb", "lbs",
        "in", "ft", "m", "cm", "mm", "l", "L", "ml", "s", "min", "h", "A", "V",
        "N", "J", "W", "Pa", "mol", "cd", "sr", "rad", "K", "Ω", "ohm", "degC", "degF"
    }
    cleaned = unit.lower().strip()
    if cleaned in dont_pluralize or len(unit) <= 2:
        return unit
    # Heuristic: Temperature/degree units not pluralized, even if value != 1
    if cleaned.startswith('degree') or cleaned in {'°c', '°f', 'degc', 'degf', 'celsius', 'fahrenheit', 'kelvin'}:
        return unit
    try:
        absval = abs(float(value))
        if absval == 1:
            return unit
        # --- PATCH: Pluralize with 'es' for units ending in 'ch', 'sh', 'x', 's', 'z', or 'o'
        if unit.endswith("inch"):
            return "inches"
        # Standard rules:
        if unit.endswith(("ch", "sh", "x", "s", "z", "o")):
            return unit + "es"
        # Ends with 'y' preceded by a consonant: 'y' -> 'ies'
        elif unit.endswith('y') and len(unit) > 1 and unit[-2] not in 'aeiou':
            return unit[:-1] + 'ies'
        elif unit.endswith('f'):
            return unit[:-1] + 'ves'
        elif unit.endswith('fe'):
            return unit[:-2] + 'ves'
        else:
            return unit + 's'
    except Exception:
        return unit

def handle_unit_conversion(query):
    """
    Handles queries asking for unit conversions (e.g., "how many inches in 8 feet").
    Returns a conversion string or None if input is not a unit conversion.
    PATCH: Fixes false positives on non-unit expressions like 'added to', 'plus', 'minus'.
    PATCH: Outputs use pluralize_unit for target units where applicable in all formatted results.
    """
    q = normalize_nl_input(query)
    q = q.replace('?', '').strip()
    q_numwords = replace_number_words(q)

    # Patch: Add operator_words with broader matching, including 'added', 'added to'
    operator_words = {
        "plus", "minus", "add", "added", "added to", "subtract", "times",
        "multiplied", "divide", "over", "and", "to", "from", "by", "subtracted", "subtracted from"
    }

    # Helper to check if a string contains math operator words
    def contains_math_operators(s):
        s = s.lower()
        for w in operator_words:
            if re.search(rf"\b{re.escape(w)}\b", s):
                return True
        return False

    unit_synonyms = {
        "c": "degC", "celsius": "degC", "centigrade": "degC",
        "f": "degF", "fahrenheit": "degF",
        "k": "kelvin", "kelvin": "kelvin",
        "degrees celsius": "degC", "degrees fahrenheit": "degF",
        "degree celsius": "degC", "degree fahrenheit": "degF",
        "degrees": None, "degree": None,
        "hrs": "hour", "hr": "hour", "hours": "hour",
        "mins": "minute", "min": "minute", "minutes": "minute",
        "seconds": "second", "secs": "second", "sec": "second",
        "kgs": "kilogram", "kg": "kilogram", "g": "gram", "grams": "gram",
        "lbs": "pound", "lb": "pound", "pounds": "pound",
        "oz": "ounce", "ounces": "ounce",
        "miles": "mile", "kms": "kilometer", "km": "kilometer",
        "kilometers": "kilometer", "meters": "meter", "metres": "meter",
        "m": "meter", "feet": "foot", "ft": "foot",
        "inches": "inch", "inch": "inch",
        "yards": "yard", "yard": "yard",
        "liters": "liter", "litres": "liter", "l": "liter",
        "ml": "milliliter", "milliliters": "milliliter", "millilitres": "milliliter",
        "gallons": "gallon", "gallon": "gallon", "cups": "cup", "cup": "cup",
    }

    def normalize_unit(unit, other_unit=None):
        unit = unit.strip().lower()
        if unit.endswith("s") and unit[:-1] in unit_synonyms:
            unit = unit[:-1]
        unit = re.sub(r'\bare there\b|\bare\b|there\b', '', unit).strip()
        if unit in unit_synonyms and unit_synonyms[unit]:
            return unit_synonyms[unit]
        if unit in ["degrees", "degree"]:
            if other_unit:
                if other_unit in ["degf", "f", "fahrenheit"]:
                    return "degC"
                elif other_unit in ["degc", "c", "celsius", "centigrade"]:
                    return "degF"
            return "degC"
        return unit

    def add_period(s):
        s = s.rstrip()
        return s if s.endswith('.') else s + '.'

    # --- PATCH: All user output returns will pass through add_period ---

    # -- PATCH: improved regex for "how many [to_unit] are there in [value] [from_unit]"
    special_pat = re.compile(
        r'how many ([a-zA-Z]+)\s+(?:are there|is|are|)?\s*in\s*([0-9\.\-and\/ ]+)\s*([a-zA-Z]+)',
        re.IGNORECASE
    )
    m = special_pat.search(q_numwords)
    if m:
        to_unit_raw = m.group(1).strip()
        val_phrase = m.group(2).strip()
        from_unit_raw = m.group(3).strip()
        if contains_math_operators(q) or contains_math_operators(val_phrase):
            return None  # NOT a unit conversion if value phrase has math operator words
        value = words_phrase_to_number(val_phrase)
        if value is None:
            try:
                value = float(val_phrase)
            except Exception:
                return add_period(f"Sorry, I couldn't parse the value '{val_phrase}'")
        from_unit = normalize_unit(from_unit_raw, to_unit_raw)
        to_unit = normalize_unit(to_unit_raw, from_unit_raw)
        try:
            quantity = Q_(value, from_unit)
            converted = quantity.to(to_unit)
            magnitude = converted.magnitude
            if from_unit in ["degC", "degF", "kelvin"] or to_unit in ["degC", "degF", "kelvin"]:
                from_unit_symbol = pretty_temp_unit(from_unit_raw)
                to_unit_symbol = pretty_temp_unit(to_unit_raw)
                magnitude_str = f"{magnitude:.2f}" if abs(magnitude) >= 1e-8 else "0.00"
                return add_period(f"{value:.2f} {from_unit_symbol} is equal to {magnitude_str} {to_unit_symbol}")
            else:
                if isinstance(magnitude, float) and magnitude.is_integer():
                    magnitude_str = str(int(magnitude))
                else:
                    magnitude_str = f"{magnitude:.4g}".replace(",", "")
                pluralized = pluralize_unit(str(converted.units), magnitude)
                return add_period(f"{value} {from_unit_raw} is {magnitude_str} {pluralized}")
        except Exception as e:
            return add_period(f"Sorry, I couldn't convert {value} {from_unit_raw} to {to_unit_raw}: {e}")

    # PATCH: Handle "[How many] [to_unit] is [value] [from_unit]" (as in 'how many kilometers is 10 miles')
    direct_pat = re.compile(
        r'how many (\w+) is ([a-zA-Z0-9. \-and\/]+?) (\w+)',
        re.IGNORECASE
    )
    m = direct_pat.search(q_numwords)
    if m:
        to_unit_raw = m.group(1).strip()
        val_phrase = m.group(2).strip()
        from_unit_raw = m.group(3).strip()
        if contains_math_operators(q) or contains_math_operators(val_phrase):
            return None
        value = words_phrase_to_number(val_phrase)
        if value is None:
            try:
                value = float(val_phrase)
            except Exception:
                return add_period(f"Sorry, I couldn't parse the value '{val_phrase}'")
        from_unit = normalize_unit(from_unit_raw, to_unit_raw)
        to_unit = normalize_unit(to_unit_raw, from_unit_raw)
        try:
            quantity = Q_(value, from_unit)
            converted = quantity.to(to_unit)
            magnitude = converted.magnitude
            if from_unit in ["degC", "degF", "kelvin"] or to_unit in ["degC", "degF", "kelvin"]:
                from_unit_symbol = pretty_temp_unit(from_unit_raw)
                to_unit_symbol = pretty_temp_unit(to_unit_raw)
                magnitude_str = f"{magnitude:.2f}" if abs(magnitude) >= 1e-8 else "0.00"
                return add_period(f"{value:.2f} {from_unit_symbol} is equal to {magnitude_str} {to_unit_symbol}")
            else:
                if isinstance(magnitude, float) and magnitude.is_integer():
                    magnitude_str = str(int(magnitude))
                else:
                    magnitude_str = f"{magnitude:.4g}".replace(",", "")
                pluralized = pluralize_unit(str(converted.units), magnitude)
                return add_period(f"{value} {from_unit_raw} is {magnitude_str} {pluralized}")
        except Exception as e:
            return add_period(f"Sorry, I couldn't convert {value} {from_unit_raw} to {to_unit_raw}: {e}")
        
    # -- PATCH: Add support for "[value] [from_unit] to [to_unit]" style e.g. "5 feet to meters"
    # -- PATCH: Improved pattern to allow multi-word/phrase numbers like 'three and a half'
    to_pat = re.compile(
        r'([a-zA-Z \d\.\-\/]+?)\s*([a-zA-Z]+)\s+to\s+([a-zA-Z]+)', re.IGNORECASE
    )
    m = to_pat.search(q_numwords)
    if m:
        val_phrase = m.group(1).strip()
        from_unit_raw = m.group(2).strip()
        to_unit_raw = m.group(3).strip()
        value = words_phrase_to_number(val_phrase)
        if value is None:
            try:
                value = float(val_phrase)
            except Exception:
                return add_period(f"Sorry, I couldn't parse the value '{val_phrase}'")
        from_unit = normalize_unit(from_unit_raw, to_unit_raw)
        to_unit = normalize_unit(to_unit_raw, from_unit_raw)
        try:
            quantity = Q_(value, from_unit)
            converted = quantity.to(to_unit)
            magnitude = converted.magnitude
            if (from_unit in ["foot", "feet"] and to_unit in ["meter"]) or (from_unit in ["meter"] and to_unit in ["foot", "feet"]):
                pluralized = pluralize_unit(str(converted.units), magnitude)
                if isinstance(magnitude, float) and magnitude.is_integer():
                    magnitude_str = str(int(magnitude))
                else:
                    magnitude_str = f"{magnitude:.4g}".replace(",", "")
                return add_period(f"{value} {from_unit_raw} is {magnitude_str} {pluralized}")
            if from_unit in ["degC", "degF", "kelvin"] or to_unit in ["degC", "degF", "kelvin"]:
                from_unit_symbol = pretty_temp_unit(from_unit_raw)
                to_unit_symbol = pretty_temp_unit(to_unit_raw)
                magnitude_str = f"{magnitude:.2f}" if abs(magnitude) >= 1e-8 else "0.00"
                return add_period(f"{value:.2f}{from_unit_symbol} is equal to {magnitude_str}{to_unit_symbol}")
            else:
                if isinstance(magnitude, float) and magnitude.is_integer():
                    magnitude_str = str(int(magnitude))
                else:
                    magnitude_str = f"{magnitude:.4g}".replace(",", "")
                pluralized = pluralize_unit(str(converted.units), magnitude)
                return add_period(f"{value} {from_unit_raw} is {magnitude_str} {pluralized}")
        except Exception as e:
            return add_period(f"Sorry, I couldn't convert {value} {from_unit_raw} to {to_unit_raw}: {e}")

    # --- PATCH: Greedy pattern to extract [number phrase] [unit] (to/in/into) [unit] ---
    phrase_pattern = re.compile(
        r'''(?P<value_phrase>(?:\d+(\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|[0-9\s\-\.\/and]+)+)\s+(?P<from_unit>[a-zA-Z]+)\s*(?:to|in|into)\s*(?P<to_unit>[a-zA-Z\s]+)''',
        re.IGNORECASE
    )
    m = phrase_pattern.search(q_numwords)
    if m:
        raw_value_phrase = m.group("value_phrase").strip()
        from_unit_raw = m.group("from_unit").strip()
        to_unit_raw = m.group("to_unit").strip()
        if contains_math_operators(q) or contains_math_operators(raw_value_phrase):
            return None
        raw_value_phrase = re.sub(
            r'^(what is|convert|please|can you|could you|show me|give me|tell me|how much is|whats|solve|determine|work out|i need|i want|i\'d like|id like|help me|how many)\s+',
            '', raw_value_phrase, flags=re.IGNORECASE
        )
        value = words_phrase_to_number(raw_value_phrase)
        if value is None:
            try:
                value = float(raw_value_phrase)
            except Exception:
                return add_period(f"Sorry, I couldn't parse the value '{raw_value_phrase}'")
        from_unit = normalize_unit(from_unit_raw, to_unit_raw)
        to_unit = normalize_unit(to_unit_raw, from_unit_raw)
        try:
            quantity = Q_(value, from_unit)
            converted = quantity.to(to_unit)
            magnitude = converted.magnitude
            if from_unit in ["degC", "degF", "kelvin"] or to_unit in ["degC", "degF", "kelvin"]:
                from_unit_symbol = pretty_temp_unit(from_unit_raw)
                to_unit_symbol = pretty_temp_unit(to_unit_raw)
                magnitude_str = f"{magnitude:.2f}" if abs(magnitude) >= 1e-8 else "0.00"
                return add_period(f"{value:.2f} {from_unit_symbol} is equal to {magnitude_str} {to_unit_symbol}")
            else:
                if isinstance(magnitude, float) and magnitude.is_integer():
                    magnitude_str = str(int(magnitude))
                else:
                    magnitude_str = f"{magnitude:.4g}".replace(",", "")
                pluralized = pluralize_unit(str(converted.units), magnitude)
                return add_period(f"{value} {from_unit_raw} is {magnitude_str} {pluralized}")
        except Exception as e:
            if (from_unit, to_unit) in [("degC", "degF"), ("degF", "degC")]:
                from_unit_symbol = pretty_temp_unit(from_unit_raw)
                to_unit_symbol = pretty_temp_unit(to_unit_raw)
                if from_unit == "degC":
                    fahrenheit = (value * 9/5) + 32
                    fahrenheit = 0.0 if abs(fahrenheit) < 1e-8 else round(fahrenheit, 2)
                    return add_period(f"{value:.2f} {from_unit_symbol} is equal to {fahrenheit:.2f} {to_unit_symbol}")
                else:
                    celsius = (value - 32) * 5/9
                    celsius = 0.0 if abs(celsius) < 1e-8 else round(celsius, 2)
                    return add_period(f"{value:.2f} {from_unit_symbol} is equal to {celsius:.2f} {to_unit_symbol}")
            return add_period(f"Sorry, I couldn't convert {value} {from_unit_raw} to {to_unit_raw}: {e}")

    # ...rest of original method unchanged...

    return None

# PATCHED handle_nl_multi_step_math's multi-step regex to accept 'divide by', 'multiply by', 'add', 'subtract', etc.

def handle_nl_multi_step_math(user_input):
    """
    Handles multi-step NL math queries like:
    ...
    PATCH: Formats answer as 'The result of [expression] is [value].'
    PATCH (2024-07): Accepts 'divide by', 'multiply by' style ops for multi-step patterns.
    """
    q = user_input.lower().strip().replace('?', '')

    # Helper to convert number words to float
    def to_number(s):
        val = words_phrase_to_number(s)
        if val is not None:
            return float(val)
        try:
            return float(s)
        except Exception:
            return None

    # PATCH: Allow multi-word operators
    op_map = {
        "double": lambda x: x * 2,
        "triple": lambda x: x * 3,
        "halve": lambda x: x / 2,
        "half": lambda x: x / 2,
        "add": lambda x, y: x + y,
        "plus": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "minus": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "times": lambda x, y: x * y,
        "multiply by": lambda x, y: x * y,
        "multiplied by": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
        "divide by": lambda x, y: x / y,
        "divided by": lambda x, y: x / y,
        "over": lambda x, y: x / y,
    }

    and_pat = r"(?:and(?:\s+then)?)"
    # Operators as alternation (longest first!)
    op_alternation = "multiplied by|multiply by|divided by|divide by|add|plus|subtract|minus|multiply|times|divide|over"

    # General pattern: [op1] [num1] and/and then [op2] [num2]
    m = re.match(
        rf"(double|triple|halve|half)\s+([-\w\s\.]+)\s+{and_pat}\s+({op_alternation})\s+([-\w\s\.]+)",
        q)
    if m:
        op1, num1_raw, op2, num2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        if num1 is None or num2 is None or op2 not in op_map:
            return None
        val = op_map[op1](num1)
        expr_map = {
            "add": "+", "plus": "+", "subtract": "-", "minus": "-",
            "multiply": "*", "times": "*", "multiply by": "*", "multiplied by": "*",
            "divide": "/", "divided by": "/", "divide by": "/", "over": "/",
        }
        symbol2 = expr_map[op2]
        left_expr = f"({num1}) * 2" if op1 == "double" else f"({num1}) * 3" if op1 == "triple" else f"({num1}) / 2"
        expr = f"{left_expr} {symbol2} {num2}"
        result = op_map[op2](val, num2)
        return f"The result of {expr} is {result}."

    # Step 1: Handle "If I <op1> <num1> and/and then <op2> <num2>, what do I get"
    m = re.match(
        rf"if i (double|triple|halve|half)\s+([-\w\s\.]+)\s+{and_pat}\s+({op_alternation})\s+([-\w\s\.]+),?\s*(what do i get|what is the result|what's the result|what will i get)?",
        q)
    if m:
        op1, num1_raw, op2, num2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        if num1 is None or num2 is None or op2 not in op_map:
            return None
        val = op_map[op1](num1)
        expr_map = {
            "add": "+", "plus": "+", "subtract": "-", "minus": "-",
            "multiply": "*", "times": "*", "multiply by": "*", "multiplied by": "*",
            "divide": "/", "divided by": "/", "divide by": "/", "over": "/",
        }
        symbol2 = expr_map[op2]
        left_expr = f"({num1}) * 2" if op1 == "double" else f"({num1}) * 3" if op1 == "triple" else f"({num1}) / 2"
        expr = f"{left_expr} {symbol2} {num2}"
        result = op_map[op2](val, num2)
        return f"The result of {expr} is {result}."

    # Step 2: "If I <op1> <num1> and then <op2> <num2>, ..."
    m = re.match(
        rf"if i (double|triple|halve|half)\s+([-\w\s\.]+)\s+and then\s+({op_alternation})\s+([-\w\s\.]+),?\s*(what do i get|what is the result|what's the result|what will i get)?",
        q)
    if m:
        op1, num1_raw, op2, num2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        if num1 is None or num2 is None or op2 not in op_map:
            return None
        val = op_map[op1](num1)
        expr_map = {
            "add": "+", "plus": "+", "subtract": "-", "minus": "-",
            "multiply": "*", "times": "*", "multiply by": "*", "multiplied by": "*",
            "divide": "/", "divided by": "/", "divide by": "/", "over": "/",
        }
        symbol2 = expr_map[op2]
        left_expr = f"({num1}) * 2" if op1 == "double" else f"({num1}) * 3" if op1 == "triple" else f"({num1}) / 2"
        expr = f"{left_expr} {symbol2} {num2}"
        result = op_map[op2](val, num2)
        return f"The result of {expr} is {result}."

    # (remaining code unchanged...)
    # [other cases for simpler multi-step/single-step or single op patterns]
    # ... rest of handle_nl_multi_step_math remains as before ...

    # The remaining existing multi-step and single-step patterns are unchanged...
    # Step 0: Handle "<op1> <num1> and <op2> <num2>" (without "if I")
    m = re.match(
        r"(double|triple|halve|half)\s+([-\w\s\.]+)\s+and\s+(add|plus|subtract|minus|multiply|times|divide|over)\s+([-\w\s\.]+)",
        q)
    if m:
        op1, num1_raw, op2, num2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        if num1 is None or num2 is None:
            return None
        val = op_map[op1](num1)
        expr_map = {
            "add": "+", "plus": "+", "subtract": "-", "minus": "-",
            "multiply": "*", "times": "*", "divide": "/", "over": "/",
        }
        symbol2 = expr_map[op2]
        left_expr = f"({num1}) * 2" if op1 == "double" else f"({num1}) * 3" if op1 == "triple" else f"({num1}) / 2"
        expr = f"{left_expr} {symbol2} {num2}"
        result = op_map[op2](val, num2)
        return f"The result of {expr} is {result}."

    # Step 3+: unchanged...
    # Step 3: Handle "If I <op1> <num1>, what do I get?"
    m = re.match(
        r"if i (double|triple|halve|half)\s+([-\w\s\.]+),?\s*(what do i get|what is the result|what's the result|what will i get)?",
        q)
    if m:
        op1, num1_raw = m.group(1), m.group(2)
        num1 = to_number(num1_raw)
        if num1 is None:
            return None
        if op1 in op_map:
            result = op_map[op1](num1)
            left_expr = f"({num1}) * 2" if op1 == "double" else f"({num1}) * 3" if op1 == "triple" else f"({num1}) / 2"
            return f"The result of {left_expr} is {result}."

    # Step 4: Handle "If I <op1> <num1> and <op2> <num2> and <op3> <num3>, what do I get?"
    m = re.match(
        r"if i (double|triple|halve|half)\s+([-\w\s\.]+)\s+and\s+(add|plus|subtract|minus|multiply|times|divide|over)\s+([-\w\s\.]+)\s+and\s+(add|plus|subtract|minus|multiply|times|divide|over)\s+([-\w\s\.]+),?\s*(what do i get|what is the result|what's the result|what will i get)?",
        q)
    if m:
        op1, num1_raw, op2, num2_raw, op3, num3_raw = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        num3 = to_number(num3_raw)
        if num1 is None or num2 is None or num3 is None:
            return None
        if op1 in op_map:
            val = op_map[op1](num1)
        else:
            return None
        if op2 in op_map:
            val2 = op_map[op2](val, num2)
        else:
            return None
        if op3 in op_map:
            result = op_map[op3](val2, num3)
        else:
            return None
        # Compose expression string
        op_map_str = {
            "add": "+", "plus": "+", "subtract": "-", "minus": "-",
            "multiply": "*", "times": "*", "divide": "/", "over": "/",
        }
        left_expr = f"({num1}) * 2" if op1 == "double" else f"({num1}) * 3" if op1 == "triple" else f"({num1}) / 2"
        expr = f"{left_expr} {op_map_str[op2]} {num2} {op_map_str[op3]} {num3}"
        return f"The result of {expr} is {result}."

    # Step 5: Handle "If I add 5 to 10 and double the result, what do I get?"
    m = re.match(
        r"if i (add|plus|subtract|minus|multiply|times|divide|over)\s+([-\w\s\.]+)\s+(to|from|by)\s+([-\w\s\.]+)\s+and\s+(double|triple|halve|half)\s+the result,?\s*(what do i get|what is the result|what's the result|what will i get)?",
        q)
    if m:
        op1, num1_raw, prep, num2_raw, op2 = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        if num1 is None or num2 is None:
            return None
        if op1 in ["add", "plus"]:
            val = num2 + num1
            expr1 = f"{num2} + {num1}"
        elif op1 in ["subtract", "minus"]:
            val = num2 - num1
            expr1 = f"{num2} - {num1}"
        elif op1 in ["multiply", "times"]:
            val = num2 * num1
            expr1 = f"{num2} * {num1}"
        elif op1 in ["divide", "over"]:
            val = num2 / num1
            expr1 = f"{num2} / {num1}"
        else:
            return None
        left_expr = f"({expr1}) * 2" if op2 == "double" else f"({expr1}) * 3" if op2 == "triple" else f"({expr1}) / 2"
        if op2 in op_map:
            result = op_map[op2](val)
            return f"The result of {left_expr} is {result}."

    # Step 6: Handle "If I add 5 to 10, what do I get?"
    m = re.match(
        r"if i (add|plus|subtract|minus|multiply|times|divide|over)\s+([-\w\s\.]+)\s+(to|from|by)\s+([-\w\s\.]+),?\s*(what do i get|what is the result|what's the result|what will i get)?",
        q)
    if m:
        op1, num1_raw, prep, num2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        if num1 is None or num2 is None:
            return None
        if op1 in ["add", "plus"]:
            val = num2 + num1
            expr = f"{num2} + {num1}"
        elif op1 in ["subtract", "minus"]:
            val = num2 - num1
            expr = f"{num2} - {num1}"
        elif op1 in ["multiply", "times"]:
            val = num2 * num1
            expr = f"{num2} * {num1}"
        elif op1 in ["divide", "over"]:
            val = num2 / num1
            expr = f"{num2} / {num1}"
        else:
            return None
        return f"The result of {expr} is {val}."

    # Step 7: Handle "If I add 5 and 10, what do I get?"
    m = re.match(
        r"if i (add|plus|subtract|minus|multiply|times|divide|over)\s+([-\w\s\.]+)\s+and\s+([-\w\s\.]+),?\s*(what do i get|what is the result|what's the result|what will i get)?",
        q)
    if m:
        op1, num1_raw, num2_raw = m.group(1), m.group(2), m.group(3)
        num1 = to_number(num1_raw)
        num2 = to_number(num2_raw)
        if num1 is None or num2 is None:
            return None
        op_map_str = {
            "add": "+", "plus": "+", "subtract": "-", "minus": "-",
            "multiply": "*", "times": "*", "divide": "/", "over": "/",
        }
        if op1 in ["add", "plus"]:
            val = num1 + num2
        elif op1 in ["subtract", "minus"]:
            val = num1 - num2
        elif op1 in ["multiply", "times"]:
            val = num1 * num2
        elif op1 in ["divide", "over"]:
            val = num1 / num2
        else:
            return None
        expr = f"{num1} {op_map_str[op1]} {num2}"
        return f"The result of {expr} is {val}."

    return None

def handle_comparison_queries(user_input):
    """
    Handles queries like:
    - 'Which is bigger: e^2 or pi^2?'
    - 'Which is greater, 2^10 or 1000?'
    - 'Which is smaller: sqrt(2) or 1.5?'
    Returns a string with the comparison result, or None if not matched.
    """
    import re
    import sympy as sp

    # Normalize input
    q = user_input.lower().strip().rstrip("?.!,;:")

    # Patterns for comparison
    patterns = [
        r'which is (bigger|greater|larger|smaller|less)[\s:,-]*(.+?)\s*(?:or|vs|versus|,)\s*(.+)',
        r'(?:is|which is)\s*(.+?)\s*(bigger|greater|larger|smaller|less)\s*than\s*(.+)',
    ]
    for pat in patterns:
        m = re.match(pat, q)
        if m:
            if pat.startswith('which is'):
                comp_word = m.group(1)
                expr1 = m.group(2).strip()
                expr2 = m.group(3).strip()
            else:
                expr1 = m.group(1).strip()
                comp_word = m.group(2)
                expr2 = m.group(3).strip()
            # Remove any trailing question marks or punctuation
            expr1 = expr1.rstrip("?.!,;:")
            expr2 = expr2.rstrip("?.!,;:")

            # --- PATCH: Normalize exponentiation operator ---
            expr1_norm = extract_math_expression(expr1.replace('^', '**'))
            expr2_norm = extract_math_expression(expr2.replace('^', '**'))

            # --- PATCH: Add pi and e to locals for sympify ---
            sympy_locals = dict(allowed_functions)
            sympy_locals.update({
                "pi": sp.pi,
                "e": sp.E,
                "E": sp.E,
            })

            try:
                val1 = float(sp.sympify(expr1_norm, locals=sympy_locals).evalf())
                val2 = float(sp.sympify(expr2_norm, locals=sympy_locals).evalf())
            except Exception as e:
                return f"Sorry, I couldn't evaluate one of the expressions: '{expr1}' or '{expr2}'."

            # Determine comparison
            if comp_word in ['bigger', 'greater', 'larger']:
                if val1 > val2:
                    return (f"{expr1} is bigger than {expr2}.\n"
                            f"{expr1} ≈ {val1:.4g}, {expr2} ≈ {val2:.4g}")
                elif val2 > val1:
                    return (f"{expr2} is bigger than {expr1}.\n"
                            f"{expr2} ≈ {val2:.4g}, {expr1} ≈ {val1:.4g}")
                else:
                    return (f"{expr1} and {expr2} are equal.\n"
                            f"Both ≈ {val1:.4g}")
            elif comp_word in ['smaller', 'less']:
                if val1 < val2:
                    return (f"{expr1} is smaller than {expr2}.\n"
                            f"{expr1} ≈ {val1:.4g}, {expr2} ≈ {val2:.4g}")
                elif val2 < val1:
                    return (f"{expr2} is smaller than {expr1}.\n"
                            f"{expr2} ≈ {val2:.4g}, {expr1} ≈ {val1:.4g}")
                else:
                    return (f"{expr1} and {expr2} are equal.\n"
                            f"Both ≈ {val1:.4g}")
            else:
                return f"Comparison word '{comp_word}' not recognized."
    return None

def handle_equality_comparison_query(user_input):
    """
    Handles queries like:
    - 'Is 3^2 + 4^2 equal to 5^2?'
    - 'Is 2 + 2 = 4?'
    Returns a string with the comparison result, or None if not matched.
    """
    # Remove trailing punctuation and lowercase
    q = user_input.strip().rstrip("?.!,;:").lower()

    # Pattern 1: "is ... equal to ..."
    m = re.match(r'is\s+(.+?)\s+equal\s+to\s+(.+)', q)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
    else:
        # Pattern 2: "is ... = ..."
        m = re.match(r'is\s+(.+?)\s*=\s*(.+)', q)
        if m:
            left = m.group(1).strip()
            right = m.group(2).strip()
        else:
            return None

    # Extract and normalize both sides
    left_expr = extract_math_expression(left)
    right_expr = extract_math_expression(right)
    try:
        left_val = float(calculate(left_expr))
        right_val = float(calculate(right_expr))
        if abs(left_val - right_val) < 1e-8:
            return (f"Yes, {left_expr} = {right_expr}.\n"
                    f"Both sides evaluate to {left_val:.4g}.")
        else:
            return (f"No, {left_expr} ≠ {right_expr}.\n"
                    f"Left: {left_val:.4g}, Right: {right_val:.4g}.")
    except Exception:
        # If calculation fails, show both expressions and their attempted results
        left_result = calculate(left_expr)
        right_result = calculate(right_expr)
        return (f"Could not compare numerically.\n"
                f"Left: {left_expr} → {left_result}\n"
                f"Right: {right_expr} → {right_result}")

def handle_date_offset_query(query):
    """
    Handles queries like:
    - 'What date is 30 days after January 15, 2023?'
    - 'What date is 2 weeks before June 5, 2024?'
    - 'What date is 4 months after March 1, 2020?'
    Returns a string with the resulting date, or None if not matched.
    """
    
    try:
        from dateutil import parser as date_parser
        from dateutil.relativedelta import relativedelta
    except ImportError:
        date_parser = None
        relativedelta = None

    q = query.lower().strip().replace("?", "")
    # Patterns for various units
    offset_patterns = [
        # [number] [unit] after/before [date]
        (r'(\d+)\s*day(?:s)?\s*(after|before)\s+([a-zA-Z0-9,/\-\s]+)', 'days'),
        (r'(\d+)\s*week(?:s)?\s*(after|before)\s+([a-zA-Z0-9,/\-\s]+)', 'weeks'),
        (r'(\d+)\s*month(?:s)?\s*(after|before)\s+([a-zA-Z0-9,/\-\s]+)', 'months'),
        # after/before [number] [unit] from [date]
        (r'(after|before)\s*(\d+)\s*day(?:s)?\s*(?:from|since|starting on)?\s+([a-zA-Z0-9,/\-\s]+)', 'days-rev'),
        (r'(after|before)\s*(\d+)\s*week(?:s)?\s*(?:from|since|starting on)?\s+([a-zA-Z0-9,/\-\s]+)', 'weeks-rev'),
        (r'(after|before)\s*(\d+)\s*month(?:s)?\s*(?:from|since|starting on)?\s+([a-zA-Z0-9,/\-\s]+)', 'months-rev'),
    ]

    # Date parsing helper
    def parse_date(s):
        if date_parser:
            try:
                return date_parser.parse(s, fuzzy=True)
            except Exception:
                pass
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        return None

    for pat, unit in offset_patterns:
        m = re.search(pat, q)
        if m:
            # Forward form: "[number] [unit] after/before [date]"
            if unit in ["days", "weeks", "months"]:
                n = int(m.group(1))
                after = m.group(2) == "after"
                date_str = m.group(3).strip()
            # Reverse form: "after/before [number] [unit] from [date]"
            else:
                after = m.group(1) == "after"
                n = int(m.group(2))
                date_str = m.group(3).strip()
                unit = unit.replace("-rev", "")

            dt = parse_date(date_str)
            if not dt:
                return f"Sorry, I couldn't parse the date '{date_str}'."
            # Compute new date
            if unit == "days":
                delta = timedelta(days=n)
                new_date = dt + delta if after else dt - delta
                phrase = f"{n} day{'s' if n != 1 else ''} {'after' if after else 'before'} {dt.strftime('%Y-%m-%d')}"
            elif unit == "weeks":
                delta = timedelta(weeks=n)
                new_date = dt + delta if after else dt - delta
                phrase = f"{n} week{'s' if n != 1 else ''} {'after' if after else 'before'} {dt.strftime('%Y-%m-%d')}"
            elif unit == "months":
                # Use relativedelta for months if available
                if relativedelta:
                    delta = relativedelta(months=n)
                    new_date = dt + delta if after else dt - delta
                else:
                    # Approximate a month as 30 days if dateutil is not available
                    delta = timedelta(days=30 * n)
                    new_date = dt + delta if after else dt - delta
                phrase = f"{n} month{'s' if n != 1 else ''} {'after' if after else 'before'} {dt.strftime('%Y-%m-%d')}"
            else:
                return f"Sorry, this offset unit is not supported."

            return f"{phrase} is {new_date.strftime('%Y-%m-%d')}."
    return None
    
def difference_between_expressions(expr1: str, expr2: str):
    """
    Given two math expressions as strings, returns a detailed comparison
    - Shows calculation for expr1 and expr2 step by step
    - Shows their difference
    - If difference is zero, adds clarifying message about their equivalence
    """
    expr1_norm = extract_math_expression(expr1)
    expr2_norm = extract_math_expression(expr2)

    try:
        val1 = float(calculate(expr1_norm))
    except Exception:
        val1 = f"Could not evaluate '{expr1}'"
    try:
        val2 = float(calculate(expr2_norm))
    except Exception:
        val2 = f"Could not evaluate '{expr2}'"

    if isinstance(val1, float) and isinstance(val2, float):
        diff = val1 - val2
        explanation = (
            f"`{expr1}` = {val1}\n"
            f"`{expr2}` = {val2}\n"
            f"Difference: {val1} - {val2} = {diff}"
        )
        if abs(diff) < 1e-8:
            explanation += (
                "\n\nNote: Both expressions are mathematically equivalent, "
                f"so their difference is zero."
            )
        return explanation
    else:
        return f"Could not compute difference:\n{expr1_norm}: {val1}\n{expr2_norm}: {val2}"

def strip_leading_phrases(text):
    """
    Removes leading question/command phrases from the input string.
    Robust to common chatty variants and extra spaces. Now strips phrases like 'can u', 'calc', etc.
    Also normalizes curly apostrophes/apostrophe-s.
    """
    # Normalize curly apostrophes and lowercase
    text = text.replace("’", "'").replace("‘", "'").strip()
    text = text.lower()

    # Extended leading phrase patterns
    leading_phrases = [
        r"how much is",   # <-- Added!
        r"what do i get with", r"what do i get by",
        r"can u", r"can you", r"could you", r"please", r"pls", r"show me", r"give me", r"tell me", r"help me",
        r"what is", r"whats", r"what's", r"calculate", r"calc", r"find", r"compute", r"evaluate", r"solve", r"work out", r"determine",
        r"convert", r"conversion", r"the value of", r"value of", r"the result of", r"result of", r"i need", r"i want", r"i'd like", r"id like",
        r"the"  # <--- ADDED 'the' here
    ]
    # REGEX: match any leading phrase, greedy, optional whitespace after each
    lp_pattern = r'^((?:' + '|'.join(leading_phrases) + r')\s*)+'
    text = re.sub(lp_pattern, '', text, flags=re.IGNORECASE)

    # Remove leading apostrophe-s (e.g. "what's ..." leaves "'s ...")
    text = re.sub(r"^'s\s+", '', text, flags=re.IGNORECASE)

    return text.strip()

def strip_trailing_phrases(text):
    """
    Remove trailing chatty phrases (e.g., 'pls', 'please', 'thanks', 'thank you') robustly.
    """
    trailing_phrases = [r"pls", r"please", r"thanks", r"thank you", r"show me"]
    tp_pattern = r'((?:\s*' + '|'.join(trailing_phrases) + r'))+$'
    text = re.sub(tp_pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def normalize_nl_input(text):
    """
    Perform full normalization: strip both leading and trailing phrases.
    """
    text = strip_leading_phrases(text)
    text = strip_trailing_phrases(text)
    # Optional: Remove unwanted trailing punctuation
    text = text.rstrip("?.!,;: \t\n\r")
    return text

# Example usage:
# s = "can u calc 10 + 4 pls"
# print(normalize_nl_input(s))  # Output: "10 + 4"

# Replace your current `strip_math_command_words` with this at an appropriate place:
def strip_math_command_words(text):
    """
    Removes leading math command words like 'expand', 'simplify', 'factor', etc., as well as the word 'this' and any colons, from the input text.
    Handles cases like 'simplify this:', 'simplify:', 'expand this:', and 'this: ...', even if 'this:' appears at the start.
    """
    text = text.strip()
    # List of math command words
    commands = [
        r'expand',
        r'simplify',
        r'factor',
        r'evaluate',
        r'calculate',
        r'find',
        r'compute',
        r'solve'
    ]
    # Word boundary to prevent partial word matches
    # Remove pattern: [any command] [optional 'this'] : [spaces]
    pattern = r'^(?:' + '|'.join(commands) + r')\b\s*(?:this\b)?\s*:?\s*'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    # Also remove just "this:" at string start or after spaces
    text = re.sub(r'^(this\b\s*:?\s*)', '', text, flags=re.IGNORECASE).strip()
    return text

# Load a pre-trained NLP model for intent recognition
nlp = pipeline("text-classification", model="distilbert-base-uncased")

# New basic mapping for NL phrases to math
natural_language_operations = {
    "add": "+",
    "plus": "+",
    "sum": "+",
    "subtract": "-",
    "minus": "-",
    "difference": "-",
    "multiply": "*",
    "times": "*",
    "product": "*",
    "divide": "/",
    "over": "/",
    "modulus": "%",
    "mod": "%",
    "remainder": "%",
    "power": "**",
    "square": "**2",
    "cube": "**3",
    "square root": "sqrt",
    "cube root": "cbrt",
    "root": "sqrt",
    "log": "log",
    "sine": "sin",
    "cosine": "cos",
    "tangent": "tan"
}

def preprocess_math_phrases(text):
    """
    Preprocesses input text to handle:
    ...
    """

    # --- PATCH: Replace number words with digits FIRST ---
    text = replace_number_words(text)

    # --- PATCH: Handle phrases like "base-10 logarithm of 1000" or "base 2 log of 16" ---
    # This must run before typo correction and below number-word replacement!
    def log_base_replacer(match):
        base = match.group(1)
        number = match.group(2)
        return f"log({number}, {base})"
    # Accept "base-10 logarithm of", "base 10 logarithm of", "base 2 log of", etc.
    text = re.sub(
        r'\bbase[\s\-]*([0-9]+)\s*log(?:arithm)?\s*of\s*([-\w\.\+\*/\(\)]+)',
        log_base_replacer,
        text,
        flags=re.IGNORECASE
    )
    # Optionally, also allow "logarithm base 10 of X"
    text = re.sub(
        r'\blog(?:arithm)?\s*base\s*([0-9]+)\s*of\s*([-\w\.\+\*/\(\)]+)',
        log_base_replacer,
        text,
        flags=re.IGNORECASE
    )
    # Optionally, "log base-10 of X"
    text = re.sub(
        r'\blog\s*base[\s\-]*([0-9]+)\s*of\s*([-\w\.\+\*/\(\)]+)',
        log_base_replacer,
        text,
        flags=re.IGNORECASE
    )

    # PATCH: Handle explicit NL phrases 'inverse tan of X', 'inverse sine of X', etc.; must be before any generic function NL/degree pattern
    text = re.sub(
        r"\binverse\s*(sine|sin|cosine|cos|tangent|tan)\s*of\s*([\-a-zA-Z0-9_\.\+\*/\(\) ]+)",
        lambda m: {
            'sine': f"asin({m.group(2).strip()})",
            'sin':  f"asin({m.group(2).strip()})",
            'cosine': f"acos({m.group(2).strip()})",
            'cos': f"acos({m.group(2).strip()})",
            'tangent': f"atan({m.group(2).strip()})",
            'tan': f"atan({m.group(2).strip()})",
        }[m.group(1).strip().lower()],
        text,
        flags=re.IGNORECASE
    )
    # Also handle "[trig] inverse of X"
    text = re.sub(
        r"\b(sine|sin|cosine|cos|tangent|tan)\s*inverse\s*of\s*([\-a-zA-Z0-9_\.\+\*/\(\) ]+)",
        lambda m: {
            'sine': f"asin({m.group(2).strip()})",
            'sin':  f"asin({m.group(2).strip()})",
            'cosine': f"acos({m.group(2).strip()})",
            'cos': f"acos({m.group(2).strip()})",
            'tangent': f"atan({m.group(2).strip()})",
            'tan': f"atan({m.group(2).strip()})",
        }[m.group(1).strip().lower()],
        text,
        flags=re.IGNORECASE
    )

    # --- PATCH: Handle log with base ---
    text = replace_log_with_base(text)

    # --- PATCH: Handle log of X -> log(X) ---
    text = re.sub(r'\blog of\s*([-\w\.\+\*/\(\)]+)', r'log(\1)', text, flags=re.IGNORECASE)

    # --- PATCH: Robustly handle "absolute value of ..." BEFORE function name substitutions ---
    text = re.sub(r'absolute value of\s*([-\w\.\+\*/\(\)]+)', r'abs(\1)', text, flags=re.IGNORECASE)

    # --- PATCH: Handle "exponential of X" and "the exponential of X" as "exp(X)" ---
    text = re.sub(r'\b(the\s+)?exponential of\s*([-\w\.\+\*/\(\)]+)', r'exp(\2)', text, flags=re.IGNORECASE)

    # --- PATCH: Handle standalone 'exponential' as 'e' ---
    # Only replace 'exponential' when it is a standalone word (not part of 'exponential of')
    text = re.sub(r'\bexponential\b', 'e', text, flags=re.IGNORECASE)

    # Replace function synonyms with canonical names
    text = re.sub(r'\bsine\b', 'sin', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcosine\b', 'cos', text, flags=re.IGNORECASE)
    text = re.sub(r'\btangent\b', 'tan', text, flags=re.IGNORECASE)

    # Handle 'square root of X plus Y' and 'square root of X minus Y'
    def sqrt_of_x_plus_y(match):
        x = match.group(1)
        y = match.group(2)
        return f"sqrt({x}) + {y}"

    def sqrt_of_x_minus_y(match):
        x = match.group(1)
        y = match.group(2)
        return f"sqrt({x}) - {y}"

    # Non-greedy matching for X, and Y can contain spaces/digits/operators
    text = re.sub(
        r'square root of\s*([-\w\.\+\*/\(\)]+?)\s+plus\s+([-\w\.\+\*/\(\)]+)',
        sqrt_of_x_plus_y,
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'square root of\s*([-\w\.\+\*/\(\)]+?)\s+minus\s+([-\w\.\+\*/\(\)]+)',
        sqrt_of_x_minus_y,
        text,
        flags=re.IGNORECASE
    )

    # Now handle the generic 'square root of X'
    text = re.sub(r'square root of\s*([-\w\.\+\*/\(\)]+)', r'sqrt(\1)', text, flags=re.IGNORECASE)
    # Also handle 'cube root of X'
    text = re.sub(r'cube root of\s*([-\w\.\+\*/\(\)]+)', r'cbrt(\1)', text, flags=re.IGNORECASE)

    # --- PATCH: Handle nth root of X (e.g., 'fourth root of 81' -> 'root(81, 4)') ---
    # Accept both ordinal and cardinal numbers (e.g., '4th', 'fourth', '5', 'fifth')
    # Use NUM_WORDS for word-to-number conversion
    def nth_root_replacer(match):
        n_word = match.group(1)
        x = match.group(2)
        # Try to convert ordinal/cardinal word to number
        try:
            n = int(n_word)
        except ValueError:
            n = NUM_WORDS.get(n_word.lower())
        if n is None:
            # Try to strip 'th', 'st', 'nd', 'rd' and convert again
            n_word_clean = re.sub(r'(st|nd|rd|th)$', '', n_word.lower())
            try:
                n = int(n_word_clean)
            except ValueError:
                n = NUM_WORDS.get(n_word_clean)
        if n is None:
            return match.group(0)  # fallback, no replacement
        return f"root({x}, {n})"

    text = re.sub(
        r'\b(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|hundredth|thousandth|\d+(?:st|nd|rd|th))\s+root of\s*([-\w\.\+\*/\(\)]+)',
        nth_root_replacer,
        text,
        flags=re.IGNORECASE
    )

    # --- PATCH: Fix 'square root of 16 plus 2' to 'sqrt(16) + 2' ---
    # This must be BEFORE the generic 'square root of' regex, and must not be greedy
    def sqrt_of_x_plus_y(match):
        x = match.group(1)
        y = match.group(2)
        return f"sqrt({x}) + {y}"
    text = re.sub(
        r'square root of\s*([-\w\.\+\*/\(\)]+?)\s+plus\s+([-\w\.\+\*/\(\)]+)',
        sqrt_of_x_plus_y,
        text,
        flags=re.IGNORECASE
    )
    def sqrt_of_x_minus_y(match):
        x = match.group(1)
        y = match.group(2)
        return f"sqrt({x}) - {y}"
    text = re.sub(
        r'square root of\s*([-\w\.\+\*/\(\)]+?)\s+minus\s+([-\w\.\+\*/\(\)]+)',
        sqrt_of_x_minus_y,
        text,
        flags=re.IGNORECASE
    )
    # Now handle the generic 'square root of X'
    text = re.sub(r'square root of\s*([-\w\.\+\*/\(\)]+)', r'sqrt(\1)', text, flags=re.IGNORECASE)
    # Also handle 'cube root of X'
    text = re.sub(r'cube root of\s*([-\w\.\+\*/\(\)]+)', r'cbrt(\1)', text, flags=re.IGNORECASE)

    # --- PATCH: Robustly convert 'sin 60' to 'sin(60)' ---
    function_names = list(allowed_functions.keys())
    function_names = sorted(function_names, key=len, reverse=True)
    func_regex = "|".join(re.escape(fn) for fn in function_names)
    text = re.sub(
        rf'\b({func_regex})\b\s*([\-]?\d+(\.\d+)?)(?!\s*\()',
        r'\1(\2)',
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(
        rf'\b({func_regex})\b\s*([a-zA-Z])(?!\s*\()',
        r'\1(\2)',
        text,
        flags=re.IGNORECASE
    )

    # --- PATCH: Robustly unwrap 'sin/cos/tan ... degrees' and do NOT leave 'degrees' as free variable ---
    def trig_of_degrees_replacer(match):
        func = match.group(1).lower()
        arg = match.group(2).strip()
        # Remove 'degrees' from argument (if present)
        arg_clean = re.sub(r"\bdegrees?\b", "", arg, flags=re.IGNORECASE).strip()
        return f"{func}(({arg_clean})*pi/180)"
    # Handles: "sin of 60 degrees"
    text = re.sub(
        r'\b(sin|sine|cos|cosine|tan|tangent)\s+of\s+([-\w\.\+\*/\(\)\s]+?)\s*degrees?\b',
        trig_of_degrees_replacer, text, flags=re.IGNORECASE
    )
    # Handles: "sin 60 degrees"
    text = re.sub(
        r'\b(sin|cos|tan)\s+([\-]?\d+(\.\d+)?)\s*degrees?\b',
        lambda m: f"{m.group(1)}(({m.group(2)})*pi/180)",
        text, flags=re.IGNORECASE
    )

    def trig_of_number_replacer(match):
        func = match.group(1).lower()
        arg = match.group(2)
        if re.match(r"^\d+(\.\d+)?$", arg.strip()):
            return f"{func}(({arg})*pi/180)"
        else:
            return f"{func}({arg})"
    text = re.sub(
        r'\b(sin|sine|cos|cosine|tan|tangent)\s+of\s+([-\d\.]+)\b',
        trig_of_number_replacer, text, flags=re.IGNORECASE
    )

    # PATCH: Handle "N times the square/cube/nth power of x [plus/minus N]" -> "N * x**2 [+/-] N"
    text = re.sub(
        r'(\b\d+\b)\s+times\s+the\s+(square|cube|(\d+)(?:st|nd|rd|th)?\s+power)\s+of\s+([a-zA-Z]\w*)\s*(plus|minus)?\s*(\d+)?',
        lambda m: (
            f"{m.group(1)} * {m.group(4)}**2" if m.group(2) == "square" else
            f"{m.group(1)} * {m.group(4)}**3" if m.group(2) == "cube" else
            f"{m.group(1)} * {m.group(4)}**{m.group(3)}"
        ) + (f" {m.group(5)} {m.group(6)}" if m.group(5) and m.group(6) else ""),
        text, flags=re.IGNORECASE
    )

    text = re.sub(
        r'the\s+square\s+of\s+([a-zA-Z]\w*)\s*(plus|minus)?\s*(\d+)?',
        lambda m: f"{m.group(1)}**2" + (f" {m.group(2)} {m.group(3)}" if m.group(2) and m.group(3) else ""),
        text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'the\s+cube\s+of\s+([a-zA-Z]\w*)\s*(plus|minus)?\s*(\d+)?',
        lambda m: f"{m.group(1)}**3" + (f" {m.group(2)} {m.group(3)}" if m.group(2) and m.group(3) else ""),
        text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'the\s+(\d+)(?:st|nd|rd|th)?\s+power\s+of\s+([a-zA-Z]\w*)\s*(plus|minus)?\s*(\d+)?',
        lambda m: f"{m.group(2)}**{m.group(1)}" + (f" {m.group(3)} {m.group(4)}" if m.group(3) and m.group(4) else ""),
        text, flags=re.IGNORECASE
    )

    text = re.sub(
        r'(\b\d+\b)\s+times\s+([a-zA-Z]\w*)\s+(squared|cubed)\s*(plus|minus)?\s*(\d+)?',
        lambda m: (
            f"{m.group(1)} * {m.group(2)}**2" if m.group(3) == "squared" else
            f"{m.group(1)} * {m.group(2)}**3"
        ) + (f" {m.group(4)} {m.group(5)}" if m.group(4) and m.group(5) else ""),
        text, flags=re.IGNORECASE
    )

    text = re.sub(
        r'(\b\d+\b)\s+times\s+([a-zA-Z]\w*)\s*(plus|minus)?\s*(\d+)?',
        lambda m: f"{m.group(1)} * {m.group(2)}" + (f" {m.group(3)} {m.group(4)}" if m.group(3) and m.group(4) else ""),
        text, flags=re.IGNORECASE
    )

    text = re.sub(
        r'([a-zA-Z]\w*)\s+(squared|cubed)\s*(plus|minus)?\s*(\d+)?',
        lambda m: (
            f"{m.group(1)}**2" if m.group(2) == "squared" else
            f"{m.group(1)}**3"
        ) + (f" {m.group(3)} {m.group(4)}" if m.group(3) and m.group(4) else ""),
        text, flags=re.IGNORECASE
    )

    def nth_root_replacer(match):
        n_word = match.group(1)
        x = match.group(2)
        try:
            n = int(n_word)
        except ValueError:
            n = NUM_WORDS.get(n_word.lower())
        if n is None:
            return match.group(0)
        return f"root({x}, {n})"

    text = re.sub(
        r'(\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|seventieth|eightieth|ninetieth|hundredth|thousandth)[\s\-]*(?:root|th root|st root|nd root|rd root|th root)\s+of\s*([-\w\.\+\*/\(\)]+)',
        nth_root_replacer,
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r'sin[\s\(]*⁻¹\s*\(?([^)]+)\)?', r'asin(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'cos[\s\(]*⁻¹\s*\(?([^)]+)\)?', r'acos(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'tan[\s\(]*⁻¹\s*\(?([^)]+)\)?', r'atan(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'sin[\s\(]*-1\s*\(?([^)]+)\)?', r'asin(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'cos[\s\(]*-1\s*\(?([^)]+)\)?', r'acos(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'tan[\s\(]*-1\s*\(?([^)]+)\)?', r'atan(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'arcsin\s*\(?([^)]+)\)?', r'asin(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'arccos\s*\(?([^)]+)\)?', r'acos(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'arctan\s*\(?([^)]+)\)?', r'atan(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'inverse\s*sine\s*of\s*([-\w\.\+\*/]+)', r'asin(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'inverse\s*cosine\s*of\s*([-\w\.\+\*/]+)', r'acos(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'inverse\s*tangent\s*of\s*([-\w\.\+\*/]+)', r'atan(\1)', text, flags=re.IGNORECASE)
    text = re.sub(r'factorial of\s*([-\w\.\+\*/\(\)]+)', r'factorial(\1)', text, flags=re.IGNORECASE)
    # Remove orphaned 'degrees' or 'degree' at boundaries
    text = re.sub(r'\bdegrees?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

    return text

def parse_natural_language(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    for phrase, symbol in natural_language_operations.items():
        if phrase in text:
            if phrase in ["square root", "cube root", "root"]:
                # e.g., "square root of 9"
                match = re.search(rf"{phrase} of (\d+)", text)
                if match:
                    number = match.group(1)
                    return f"{symbol}({number})"
            elif phrase in ["square", "cube"]:
                # e.g., "square 5" -> 5**2
                match = re.search(rf"{phrase} (\d+)", text)
                if match:
                    number = match.group(1)
                    return f"{number}{symbol}"
            else:
                # e.g., "add 5 and 7"
                match = re.search(rf"{phrase} (\d+) (and|to|by) (\d+)", text)
                if match:
                    num1 = match.group(1)
                    num2 = match.group(3)
                    return f"{num1} {symbol} {num2}"
    return None

# Map allowed function names to SymPy functions
allowed_functions = {
    "sin": sp.sin,
    "sine": sp.sin,
    "cos": sp.cos,
    "cosine": sp.cos,
    "tan": sp.tan,
    "tangent": sp.tan,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "ln": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
    "factorial": sp.factorial,
    "cbrt": lambda x: sp.root(x, 3),
    "root": sp.root,  # <-- Added for nth root support
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
}

# --- PATCH: Constants dictionary ---
allowed_constants = {
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "Pi": sp.pi,
    "pie": sp.pi,  # Accept 'pie' as a synonym for 'pi'
}

def deg_to_rad(match):
    """Convert degrees to radians for trig functions."""
    func = match.group(1)
    value = match.group(2)
    return f"{func}(({value})*pi/180)"

def extract_math_expression(text):
    """
    Extracts and normalizes a mathematical expression from text.
    PATCH: If the input is already a valid math expression, skip NL normalizations.
    """
    # If the input is just a math expression, just convert ^ to ** and strip spaces
    if re.fullmatch(r'^[\d\s\+\-\*/\^\.()\[\]]+$', text.strip()):
        return text.replace('^', '**').strip()

    # PATCH: Handle "factorial of X" FIRST before any other processing
    text = re.sub(r'factorial of\s*([-\w\d\.\+\*/\(\)]+)', r'factorial(\1)', text, flags=re.IGNORECASE)

    # PATCH: NL "inverse [func] of X" for trig
    m = re.match(r"\s*inverse\s*(sine|sin|cosine|cos|tangent|tan)\s*of\s*([\-a-zA-Z0-9_\.\+\*/\(\) ]+)", text, flags=re.IGNORECASE)
    if m:
        arg = m.group(2).strip()
        func = m.group(1).lower()
        if func in ["sin", "sine"]:
            return f"asin({arg})"
        elif func in ["cos", "cosine"]:
            return f"acos({arg})"
        elif func in ["tan", "tangent"]:
            return f"atan({arg})"
    # PATCH: NL "[func] inverse of X"
    m = re.match(r"\s*(sine|sin|cosine|cos|tangent|tan)\s*inverse\s*of\s*([\-a-zA-Z0-9_\.\+\*/\(\) ]+)", text, flags=re.IGNORECASE)
    if m:
        arg = m.group(2).strip()
        func = m.group(1).lower()
        if func in ["sin", "sine"]:
            return f"asin({arg})"
        elif func in ["cos", "cosine"]:
            return f"acos({arg})"
        elif func in ["tan", "tangent"]:
            return f"atan({arg})"

    # --- PATCH: Handle function phrases FIRST before other processing ---
    # Handle "sine of X", "cosine of X", etc. before other transformations
    text = re.sub(r'\b(sine|sin)\s+of\s+([-\w\.\+\*/\(\)]+)', r'sin(\2)', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(cosine|cos)\s+of\s+([-\w\.\+\*/\(\)]+)', r'cos(\2)', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(tangent|tan)\s+of\s+([-\w\.\+\*/\(\)]+)', r'tan(\2)', text, flags=re.IGNORECASE)
    
    # --- PATCH: Normalize 'to the power of' phrases early ---
    # e.g., "3 to the power of 4" -> "3**4"
    text = re.sub(
        r'(\d+(\.\d+)?|\w+)\s+to the power of\s+([\-]?\d+)',
        r'\1**\3',
        text,
        flags=re.IGNORECASE
    )

    # --- PATCH: Normalize power phrases early ---
    text = normalize_power_phrases(text)

    # --- PATCH: If the input is already a simple math expression, return as is ---
    if re.fullmatch(r'[\d\s\+\-\*/\^\.]+', text.strip()):
        return text.strip()

    # --- Replace log10(100) and log_b(100) with log(100, 10) FIRST ---
    text = replace_log_with_base(text)
    
    # --- NEW: Replace number words with digits FIRST ---
    text = replace_number_words(text)

    # --- NEW: Normalize Unicode math constants ---
    text = normalize_math_constants(text)

    # --- PATCH: Normalize ln(x) to log(x) for sympy compatibility ---
    text = re.sub(r'\bln\s*\(', 'log(', text, flags=re.IGNORECASE)

    # --- PATCH: Preprocess for 'N times the square/cube of x' and similar phrases ---
    text = preprocess_math_phrases(text)

    # PATCH: Always strip leading phrases before trig normalization
    text = strip_leading_phrases(text)
    
    # --- PATCH: Always normalize trig functions with numeric arguments to degrees after leading phrases ---
    def trig_numeric_arg_to_degrees(match):
        func = match.group(1)
        arg = match.group(2)
        # Only treat as degrees if arg is a number (not a variable)
        if re.match(r"^-?\d+(\.\d+)?$", arg.strip()):
            return f"{func}(({arg})*pi/180)"
        else:
            return f"{func}({arg})"

    # Always wrap sin/cos/tan followed by a number as a function call in degrees
    text = re.sub(
        r'\b(sin|cos|tan)\s+([\-]?\d+(\.\d+)?)(?!\s*\()',
        lambda m: f"{m.group(1)}(({m.group(2)})*pi/180)",
        text,
        flags=re.IGNORECASE
    )
    # Also handle sin60, cos45, tan30 (no space)
    text = re.sub(
        r'\b(sin|cos|tan)([\-]?\d+(\.\d+)?)(?!\s*\()',
        lambda m: f"{m.group(1)}(({m.group(2)})*pi/180)",
        text,
        flags=re.IGNORECASE
    )

    # --- PATCH: Handle radians explicitly ---
    # If the argument contains 'radian', do NOT convert to degrees, just strip the word 'radian(s)'
    def trig_paren_numeric_arg_to_degrees(match):
        func = match.group(1)
        arg = match.group(2)
        # If argument contains 'radian', remove the word and do not convert
        if "radian" in arg.lower():
            arg_clean = re.sub(r"\s*radians?\s*", "", arg, flags=re.IGNORECASE)
            return f"{func}({arg_clean})"
        # If argument contains 'degree', remove it and convert
        arg_clean = re.sub(r"\s*degrees?\s*", "", arg, flags=re.IGNORECASE)
        if re.match(r"^-?\d+(\.\d+)?$", arg_clean.strip()):
            return f"{func}(({arg_clean})*pi/180)"
        else:
            return f"{func}({arg})"
    text = re.sub(
        r'\b(sin|cos|tan)\s*\(\s*([^\(\)]*?)\s*\)',
        trig_paren_numeric_arg_to_degrees,
        text,
        flags=re.IGNORECASE
    )

    # Also handle cases like 'sin x', 'cos y', etc. (single variable)
    function_names = list(allowed_functions.keys())
    function_names = sorted(function_names, key=len, reverse=True)
    func_regex = "|".join(re.escape(fn) for fn in function_names)
    text = re.sub(
        rf'\b({func_regex})\s*([a-zA-Z])(?!\s*\()',
        r'\1(\2)',
        text,
        flags=re.IGNORECASE
    )

    # --- PATCH: Replace ^ with ** BEFORE inserting implicit multiplication ---
    text = text.replace("^", "**")

    # --- PATCH: Insert * between number and function call (e.g., 3sin(30) -> 3*sin(30)) ---
    # Only match when function name is followed by a parenthesis
    function_names = list(allowed_functions.keys())
    function_names = sorted(function_names, key=len, reverse=True)
    func_regex = "|".join(re.escape(fn) for fn in function_names)
    text = re.sub(
        rf'(\d+(\.\d+)?)(?=\s*({func_regex})\s*\()',
        r'\1*',
        text,
        flags=re.IGNORECASE
    )
    # Now, insert * between number and function name with no space (e.g., 3sin(30))
    text = re.sub(
        rf'(\d+(\.\d+)?)(?=({func_regex})\s*\()',
        r'\1*',
        text,
        flags=re.IGNORECASE
    )

    # --- PATCH: Only insert * between digit and single variable, not before operator words like 'plus' ---
    # Only do this if the next character is a variable (single letter), not an operator or space
    text = re.sub(r"(?<![\)\w])(\d)\s*([a-zA-Z])(?![a-zA-Z])", r"\1*\2", text)

    # --- PATCH: Remove erroneous '*' before operators or end of string ---
    text = re.sub(r'(\))\s*\*\s*([\+\-/])', r'\1 \2', text)
    text = re.sub(r'(\))\s*\*\s*$', r'\1', text)

    # --- FIX: Improved implicit multiplication insertion ---
    # Insert * between closing parenthesis and digit: (x+1)2 -> (x+1)*2
    text = re.sub(r"(\))\s*(\d)", r"\1*\2", text)
    text = re.sub(r"(\))\s*\*\s*(?=[^\d\(])", r"\1 ", text)
    # Insert * between closing parenthesis and opening parenthesis: (x+1)(y+2) -> (x+1)*(y+2)
    text = re.sub(r"(\))\s*\(", r"\1*(", text)
    # Insert * between digit and opening parenthesis: 2(x+1) -> 2*(x+1)
    text = re.sub(r"(\d)\s*\(", r"\1*(", text)

    # --- FIX: Insert * between variable and opening parenthesis, unless preceded by a function name ---
    def insert_mul_var_paren(match):
        var = match.group(1)
        if var.lower() in function_names:
            return match.group(0)
        return f"{var}*("

    text = re.sub(
        rf"\b(?!{func_regex}\b)([a-zA-Z])\s*\(",
        insert_mul_var_paren,
        text,
        flags=re.IGNORECASE
    )

    # Insert * between closing parenthesis and variable: (x)y -> (x)*y
    text = re.sub(r"(\))\s*([a-zA-Z])", r"\1*\2", text)
    # Insert * between digit and variable: 2x -> 2*x (but not at start, and not before power operator)
    # --- PATCH: Only do this if the next character is a variable and not part of a number or operator ---
    text = re.sub(r"(?<!^)(\d)\s*([a-zA-Z])(?=\W|$)", r"\1*\2", text)

    # --- PATCH: Ensure operators are surrounded by spaces to avoid merging tokens ---
    text = re.sub(r'([+\-*/=])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)

    # --- PATCH: Replace NL operators with math symbols ---
    nl_replacements = [
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\btimes\b", "*"),
        (r"\bmultiplied by\b", "*"),
        (r"\bdivided by\b", "/"),
        (r"\bover\b", "/"),
        (r"\bmodulus\b", "%"),
        (r"\bmod\b", "%"),
        (r"\bremainder\b", "%"),
        (r"\bpower\b", "**"),
        (r"\bequals\b", "="),
    ]
    for pat, sym in nl_replacements:
        text = re.sub(pat, sym, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)

    # --- PATCH: Remove any leading '*' that may have been inserted by mistake ---
    text = re.sub(r"^\s*\*", "", text)

    # --- PATCH: If the expression is just numbers and operators, preserve them as is ---
    # This ensures '20 - 7' stays as '20 - 7'
    text = text.strip()
    return text

def split_expressions(text):
    """
    Split input into separate math expressions.
    inverse_funcs = {"asin", "acos", "atan", "arcsin", "arccos", "arctan"}
    Splits only on ';' or ',' that are not inside parentheses.
    """
    parts = []
    current = []
    depth = 0
    for c in text:
        if c == '(':
            depth += 1
            current.append(c)
        elif c == ')':
            depth -= 1
            current.append(c)
        elif c in ',;' and depth == 0:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(c)
    part = ''.join(current).strip()
    if part:
        parts.append(part)
    return parts
    
def split_repeated_leading_phrases(text):
    """
    Splits input text on repeated leading phrases (e.g., 'Calculate', 'What is', etc.).
    Returns a list of separated statements.
    """
    phrases = [
        r'what is', r'calculate', r'find', r'compute', r'evaluate', r'please',
        r'can you', r'could you', r'show me', r'give me', r'tell me',
        r'how much is', r'what\'s', r'whats', r'solve', r'determine', r'work out',
        r'i need', r'i want', r'i\'d like', r'id like', r'help me'
    ]
    pattern = r'(' + '|'.join(phrases) + r')'
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if len(matches) <= 1:
        return [text]
    split_points = [m.start() for m in matches[1:]]
    prev = 0
    result = []
    for idx in split_points:
        result.append(text[prev:idx].strip())
        prev = idx
    result.append(text[prev:].strip())
    return [s for s in result if s]

def calculate(expression):
    """
    Evaluates a mathematical expression.
    PATCH: Only handles direct math functions and numeric evaluation. Algebraic variable/symbolic processing is NOT supported.

    PATCH (FIX): Allow evaluation of expressions like 'cos(pi/3)' that use known constants but no unknown symbols.
    PATCH (2024-06): Output for trig functions is now always one line and readable, showing angles in degrees if applicable.

    PATCH (2024-06): Show user-friendly domain errors for trig inverse functions (asin/acos) if input is not in [-1, 1].
    PATCH (2024-07): For trig functions like sin(X°), output only the numeric result (not nested 'The result of ...' strings).
    """
    try:

        # Helper to check if it's a trig-function call with a numeric arg and (optionally) already in degrees
        def detect_trig_numeric(expr_str):
            m = re.match(r'^\s*(sin|cos|tan)\s*\(\s*([\-]?\d+(\.\d+)?)\s*\)\s*$', expr_str)
            if m:
                func = m.group(1)
                value = float(m.group(2))
                return func, value
            return None, None

        # --- PATCH: Detect trig inverse functions with numeric input and check domain ---
        def detect_trig_inverse_numeric(expr_str):
            m = re.match(r'^\s*(asin|arcsin|acos|arccos|atan|arctan)\s*\(\s*([\-]?\d+(\.\d+)?)\s*\)\s*$', expr_str)
            if m:
                func = m.group(1)
                value = float(m.group(2))
                return func, value
            return None, None

        if isinstance(expression, str):
            # Handle trig numeric in degrees
            norm_expr = expression.strip()
            trig_func, val = detect_trig_numeric(norm_expr)
            if trig_func:
                val_rad = math.radians(val)
                res = {
                    "sin": math.sin(val_rad),
                    "cos": math.cos(val_rad),
                    "tan": math.tan(val_rad),
                }[trig_func]
                # Only return the numeric result, not wrapped in another phrase
                return f"{round(res, 4)}"

            # Handle inverse trig function domain errors
            inverse_func, value = detect_trig_inverse_numeric(norm_expr)
            if inverse_func:
                # Normalize synonym names
                inverse_base = {
                    "asin": "asin", "arcsin": "asin",
                    "acos": "acos", "arccos": "acos",
                    "atan": "atan", "arctan": "atan"
                }[inverse_func]
                # Domain checks
                if inverse_base == "asin" or inverse_base == "arcsin":
                    if not -1 <= value <= 1:
                        return "Error: asin (inverse sine) is only defined for values between -1 and 1."
                elif inverse_base == "acos" or inverse_base == "arccos":
                    if not -1 <= value <= 1:
                        return "Error: acos (inverse cosine) is only defined for values between -1 and 1."
                # atan is defined for all real numbers
                # let sympy handle valid values as usual

            # If it's a simple numeric expr
            if re.fullmatch(r'^[\d\s\+\-\*/\^\.()\[\]]+$', norm_expr):
                expr = norm_expr.replace('^', '**')
                result = eval(expr)
                rounded = round(result, 4)
                return f"{rounded}"

        # Strip leading phrases if expression is a string
        if isinstance(expression, str):
            expression = strip_leading_phrases(expression)

        allowed_funcs_only = {k: v for k, v in allowed_functions.items() if callable(v)}
        sympy_locals = {}
        sympy_locals.update(allowed_funcs_only)
        sympy_locals.update(allowed_constants)

        def normalize_func_names(expr):
            for func in allowed_functions:
                expr = re.sub(rf'\b{func}\b', func, expr, flags=re.IGNORECASE)
            return expr

        if isinstance(expression, str):
            expression = normalize_func_names(expression)
        expr = sp.sympify(expression, locals=sympy_locals)

        # --- PATCH: Only forbid expressions with true unknown variables ---
        if getattr(expr, 'is_Symbol', False):
            # This is for a single variable like just "x"
            if str(expr) not in allowed_constants:
                return "Error: Variables and symbolic algebra are not supported."
        # If free_symbols contains only 'pi', 'E', "e", "Pi", allow evaluation; else block
        if getattr(expr, 'free_symbols', set()):
            safe_syms = {'pi', 'e', 'E', 'Pi'}
            syms = {str(s) for s in expr.free_symbols}
            if len(syms) == 0:
                pass  # no variables
            elif syms <= safe_syms:
                # All symbols are known constants: allow numeric evaluation
                pass
            else:
                return "Error: Variables and symbolic algebra are not supported."
        # Otherwise, proceed to numeric evaluation as before

        # Detect trig function with numeric arg that is already normalized to degrees
        if expr.is_Function and expr.func in [sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan]:
            arg = expr.args[0]
            func_name = expr.func.__name__
            display_name = display_func_names.get(func_name, func_name)
            try:
                # --- PATCH: Domain checks for inverse trig ---
                if func_name in ["asin", "arcsin"]:
                    try:
                        arg_val = float(arg)
                        if not -1 <= arg_val <= 1:
                            return "Error: asin (inverse sine) is only defined for values between -1 and 1."
                    except Exception:
                        pass
                elif func_name in ["acos", "arccos"]:
                    try:
                        arg_val = float(arg)
                        if not -1 <= arg_val <= 1:
                            return "Error: acos (inverse cosine) is only defined for values between -1 and 1."
                    except Exception:
                        pass
                # PATCH: check if arg is numeric (no free symbols!) and does not contain pi
                if hasattr(arg, 'free_symbols') and len(arg.free_symbols) == 0:
                    # Try to also detect "k*pi/180" patterns (so we can report nice "sin(80°)" output)
                    if (
                        arg.is_Mul
                        and any(a == sp.pi for a in arg.args)
                        and any(a == sp.Integer(1)/sp.Integer(180) or a == sp.Rational(1, 180) for a in arg.args)
                    ):
                        # Try to extract the numeric coefficient (the degree value)
                        for factor in arg.args:
                            if (
                                factor.is_Mul
                                and any(ff == sp.pi for ff in factor.args)
                                and any(ff == 80 for ff in factor.args)
                            ):
                                deg_val = [ff for ff in factor.args if ff != sp.pi][0]
                                deg_val = float(deg_val)
                                numeric = expr.evalf()
                                return f"{numeric:.4f}"
                        try:
                            arg_float = float(arg)
                            deg_val = arg_float * 180 / math.pi
                            numeric = expr.evalf()
                            return f"{numeric:.4f}"
                        except Exception:
                            pass
                    try:
                        numeric = expr.evalf()
                        return f"{numeric:.4f}"
                    except Exception:
                        pass
                numeric = expr.evalf()
                return f"{numeric:.4f}"
            except Exception:
                return f"Error: Couldn't numerically evaluate {func_name}({arg})."

        result = expr.evalf()
        if (
            result.is_infinite
            or result.is_real is False
            or result == sp.zoo
            or result == sp.oo
            or result == -sp.oo
            or result == sp.nan
        ):
            return "Error: Division by zero or undefined result."
        if result.is_real:
            try:
                float_result = float(result)
                rounded = round(float_result, 4)
                if rounded == float_result:
                    return f"{rounded}"
                else:
                    return f"{rounded} (full: {float_result})"
            except Exception:
                return str(result)
        else:
            return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero."
    except sp.SympifyError:
        return "Error: Invalid mathematical expression"
    except Exception as e:
        errstr = str(e)
        # --- PATCH: Domain error strings from sympy ---
        if "asin" in errstr and "out of range" in errstr:
            return "Error: asin (inverse sine) is only defined for values between -1 and 1."
        if "acos" in errstr and "out of range" in errstr:
            return "Error: acos (inverse cosine) is only defined for values between -1 and 1."
        if "ZeroDivisionError" in errstr or "division by zero" in errstr:
            return "Error: Division by zero."
        return f"Error: Invalid expression ({e})"

def evaluate_statements(statement, show_full=False):
    """
    Evaluates multiple math expressions separated by commas or semicolons.
    Returns a string with all results, one per line.
    Variable assignment, symbolic simplification/expansion/factoring is disabled.
    PATCH: Each part/segment is first checked for natural language geometry/physics/unit/date/etc.
    PATCH (2024-06-15): Each part is normalized with number word and NL operator phrase conversion before extraction and evaluation.
    PATCH (2024-07): Result lines use digit/operator form of expression on the left (e.g. 2 + 2 = 4), not word-form.
    PATCH (2024-07-13): Don't sum or combine values from different parts inadvertently; output each as its own line.
    PATCH (2024-07-14): Fix: ensure the result for each part is correct and independent (no reuse of previous or next).
    PATCH (2024-07-15): Fix: Do not over-normalize simple infix expressions such as '3-2'; always display both the input and its result, not just '5 = 5'.
    """
    def smart_split(s):
        parts = []
        current = []
        depth = 0
        for c in s:
            if c == '(':
                depth += 1
                current.append(c)
            elif c == ')':
                depth -= 1
                current.append(c)
            elif c in ',;' and depth == 0:
                part = ''.join(current).strip()
                if part:
                    parts.append(part)
                current = []
            else:
                current.append(c)
        part = ''.join(current).strip()
        if part:
            parts.append(part)
        return parts

    allowed_funcs_only = {k: v for k, v in allowed_functions.items() if callable(v)}
    sympy_locals = {}
    sympy_locals.update(allowed_funcs_only)
    sympy_locals.update(allowed_constants)

    parts = smart_split(statement)
    output_lines = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part[0] == '#':
            output_lines.append(part)
            continue

        # --- PATCH: natural language shape/physics/other handling ---
        nl_res = handle_natural_language(part)
        if nl_res:
            output_lines.append(nl_res)
            continue

        # --- EARLY PATCH: if it's already a simple numeric infix expression, evaluate directly,
        # and skip further normalization to avoid normalizing '3-2' into '5'
        infix_math_pattern = r'^\s*\d+(\.\d+)?\s*[\+\-\*/]\s*\d+(\.\d+)?\s*$'
        if re.fullmatch(infix_math_pattern, part):
            try:
                expr_obj = sp.sympify(part, locals=sympy_locals)
                val = expr_obj.evalf()
                rounded = round(float(val), 4)
                if abs(rounded - int(rounded)) < 1e-12:
                    result_str = str(int(rounded))
                else:
                    result_str = str(rounded)
                expr_for_output = part.replace('**2', '²').replace('**3', '³')
                expr_for_output = re.sub(r'\s+', ' ', expr_for_output).strip()
                output_lines.append(f"{expr_for_output} = {result_str}")
            except Exception as e:
                output_lines.append(f"Error evaluating expression '{part}': {e}")
            continue

        # --- NEW PATCH: normalize each segment using number words and NL operator mapping
        # First: convert number words to digits
        part_numwords = replace_number_words(part)
        # Second: convert NL operator/verb phrases to math symbols/expressions
        part_nl = nl_simple_math_to_expression(part_numwords)
        # Now extract/normalize as a math expression
        expr = extract_math_expression(part_nl)

        try:
            expr_obj = sp.sympify(expr, locals=sympy_locals)
            # Ensure we only evaluate direct numeric expressions (not variable assignment or algebra)
            if getattr(expr_obj, 'is_Symbol', False) or getattr(expr_obj, 'free_symbols', set()):
                output_lines.append(f"Error: Variables and symbolic algebra are not supported.")
                continue
            val = expr_obj.evalf()
            rounded = round(float(val), 4)
            if abs(rounded - int(rounded)) < 1e-12:
                result_str = str(int(rounded))
            else:
                result_str = str(rounded)
            expr_for_output = expr.replace('**2', '²').replace('**3', '³')
            expr_for_output = re.sub(r'\s+', ' ', expr_for_output).strip()
            output_lines.append(f"{expr_for_output} = {result_str}")
        except Exception as e:
            output_lines.append(f"Error evaluating expression '{part}': {e}")
    if output_lines:
        return '\n'.join(output_lines)
    else:
        return "No output expression found."

def format_trig_solution(trig_func, rhs, var='x'):
    """
    Formats the general and numeric solutions for equations like sin(x) = a, cos(x) = a, tan(x) = a.
    Returns a Markdown string.
    """
    import sympy as sp
    rhs = sp.sympify(rhs)
    n = sp.Symbol('n', integer=True)
    pi = sp.pi

    def latex(expr):
        return sp.latex(expr, mul_symbol='dot')

    def evalf(expr, n_val):
        try:
            return float(expr.subs(n, n_val).evalf())
        except Exception:
            return None

    # Special cases for sin(x) = ±1, cos(x) = ±1
    if trig_func == 'sin':
        if rhs == 1:
            gen_sol = pi/2 + 2*pi*n
            latex_sol = f"{latex(pi/2)} + 2{latex(pi)} n"
        elif rhs == -1:
            gen_sol = 3*pi/2 + 2*pi*n
            latex_sol = f"{latex(3*pi/2)} + 2{latex(pi)} n"
        else:
            # General case: x = arcsin(a) + 2πn, x = π - arcsin(a) + 2πn
            arcsin = sp.asin(rhs)
            gen_sol_1 = arcsin + 2*pi*n
            gen_sol_2 = pi - arcsin + 2*pi*n
            latex_sol_1 = f"{latex(arcsin)} + 2{latex(pi)} n"
            latex_sol_2 = f"{latex(pi)} - {latex(arcsin)} + 2{latex(pi)} n"
            numeric_1 = [evalf(gen_sol_1, i) for i in [-1, 0, 1]]
            numeric_2 = [evalf(gen_sol_2, i) for i in [-1, 0, 1]]
            out = [
                "**✅ General solution:**",
                f"{var} = {latex_sol_1}",
                f"{var} = {latex_sol_2}",
                "where $n$ is any integer ($n \\in \\mathbb{Z}$)\n",
                "**🔢 Example numeric values (for n = -1, 0, 1):**",
                f"- n = -1 → x ≈ {numeric_1[0]:.3f}",
                f"- n = 0 → x ≈ {numeric_1[1]:.3f}",
                f"- n = 1 → x ≈ {numeric_1[2]:.3f}",
                "",
                f"- n = -1 → x ≈ {numeric_2[0]:.3f}",
                f"- n = 0 → x ≈ {numeric_2[1]:.3f}",
                f"- n = 1 → x ≈ {numeric_2[2]:.3f}",
            ]
            return "\n".join(out)
        numeric = [evalf(gen_sol, i) for i in [-1, 0, 1]]
        out = [
            "**✅ General solution:**",
            f"{var} = {latex_sol}",
            "where $n$ is any integer ($n \\in \\mathbb{Z}$)\n",
            "**🔢 Example numeric values (for n = -1, 0, 1):**",
            f"- n = -1 → x ≈ {numeric[0]:.3f}",
            f"- n = 0 → x ≈ {numeric[1]:.3f}",
            f"- n = 1 → x ≈ {numeric[2]:.3f}",
        ]
        return "\n".join(out)

    elif trig_func == 'cos':
        if rhs == 1:
            gen_sol = 0 + 2*pi*n
            latex_sol = f"0 + 2{latex(pi)} n"
        elif rhs == -1:
            gen_sol = pi + 2*pi*n
            latex_sol = f"{latex(pi)} + 2{latex(pi)} n"
        else:
            arccos = sp.acos(rhs)
            gen_sol_1 = arccos + 2*pi*n
            gen_sol_2 = -arccos + 2*pi*n
            latex_sol_1 = f"{latex(arccos)} + 2{latex(pi)} n"
            latex_sol_2 = f"-{latex(arccos)} + 2{latex(pi)} n"
            numeric_1 = [evalf(gen_sol_1, i) for i in [-1, 0, 1]]
            numeric_2 = [evalf(gen_sol_2, i) for i in [-1, 0, 1]]
            out = [
                "**✅ General solution:**",
                f"{var} = {latex_sol_1}",
                f"{var} = {latex_sol_2}",
                "where $n$ is any integer ($n \\in \\mathbb{Z}$)\n",
                "**🔢 Example numeric values (for n = -1, 0, 1):**",
                f"- n = -1 → x ≈ {numeric_1[0]:.3f}",
                f"- n = 0 → x ≈ {numeric_1[1]:.3f}",
                f"- n = 1 → x ≈ {numeric_1[2]:.3f}",
                "",
                f"- n = -1 → x ≈ {numeric_2[0]:.3f}",
                f"- n = 0 → x ≈ {numeric_2[1]:.3f}",
                f"- n = 1 → x ≈ {numeric_2[2]:.3f}",
            ]
            return "\n".join(out)
        numeric = [evalf(gen_sol, i) for i in [-1, 0, 1]]
        out = [
            "**✅ General solution:**",
            f"{var} = {latex_sol}",
            "where $n$ is any integer ($n \\in \\mathbb{Z}$)\n",
            "**🔢 Example numeric values (for n = -1, 0, 1):**",
            f"- n = -1 → x ≈ {numeric[0]:.3f}",
            f"- n = 0 → x ≈ {numeric[1]:.3f}",
            f"- n = 1 → x ≈ {numeric[2]:.3f}",
        ]
        return "\n".join(out)

    elif trig_func == 'tan':
        arctan = sp.atan(rhs)
        gen_sol = arctan + pi*n
        latex_sol = f"{latex(arctan)} + {latex(pi)} n"
        numeric = [evalf(gen_sol, i) for i in [-1, 0, 1]]
        out = [
            "**✅ General solution:**",
            f"{var} = {latex_sol}",
            "where $n$ is any integer ($n \\in \\mathbb{Z}$)\n",
            "**🔢 Example numeric values (for n = -1, 0, 1):**",
            f"- n = -1 → x ≈ {numeric[0]:.3f}",
            f"- n = 0 → x ≈ {numeric[1]:.3f}",
            f"- n = 1 → x ≈ {numeric[2]:.3f}",
        ]
        return "\n".join(out)
    else:
        return "Unsupported trig function."
    
# --- PATCH: Update is_function_name to only check allowed_functions and inverse function synonyms---
def is_function_name(expr):
    """Check if the input is just a function name (e.g., 'sin', 'log', 'atan', or 'inverse of tan')."""
    expr = expr.strip().lower()
    # Now match e.g. 'tan^-1', 'atan', 'inverse tan', etc.
    inverse_funcs = ['asin', 'acos', 'atan', 'arcsin', 'arccos', 'arctan']
    if expr in allowed_functions:
        return callable(allowed_functions[expr])
    if expr in inverse_funcs:
        return True
    # Special: 'inverse of tan'
    if re.fullmatch(r"inverse of (sin|cos|tan)", expr):
        return True
    if re.fullmatch(r"(sin|cos|tan)[\s\-]*[\^−\-]?1", expr):
        return True
    return False

def solve_function_zeros(expr):
    """
    Attempt to find zeros of a single-variable function expression.
    Returns a user-friendly string with the solutions, or None if not applicable.
    Variable assignment is not supported.
    PATCH: If the expression is a constant (no variables), do not attempt to find zeros.
    """
    try:
        # Find variable names in the expression
        var_names = set(re.findall(r'[a-zA-Z]\w*', expr))
        # Exclude known functions and constants
        known_names = set(allowed_functions.keys()) | {"e", "pi", "E", "Pi"}
        var_names = [v for v in var_names if v not in known_names]
        if len(var_names) == 0:
            # If there are no variables, check if the expression is a constant
            import sympy as sp
            try:
                val = float(sp.sympify(expr, locals=allowed_functions))
                return f"'{expr}' is a constant value ({val}). There are no zeros to find."
            except Exception:
                return None  # Not a valid constant, skip
        if len(var_names) != 1:
            return None  # Only handle single-variable functions

        var = sp.Symbol(var_names[0], real=True)
        locals_dict = {**allowed_functions, var_names[0]: var}
        sym_expr = sp.sympify(expr, locals=locals_dict)

        # Set up the equation: expr == 0
        eq = sp.Eq(sym_expr, 0)
        sol = sp.solveset(eq, var, domain=sp.S.Reals)

        if sol is sp.S.EmptySet or getattr(sol, "is_empty", False):
            return f"No real zeros found for {var}."

        # If solution is a FiniteSet, show all solutions numerically
        if isinstance(sol, sp.FiniteSet):
            numeric_solutions = [sp.N(s, 6) for s in sol]
            numeric_str = ', '.join([f"{float(s):.3f}" for s in numeric_solutions])
            return (
                f"Zero(s) for {var}: {', '.join([str(s) for s in sol])}\n"
                f"Numeric: {numeric_str}"
            )

        # For infinite sets (e.g., periodic trig solutions), show symbolic and a few numeric examples
        try:
            n = sp.Symbol('n', integer=True)
            if isinstance(sol, sp.Union):
                reps = []
                for s in sol.args:
                    if hasattr(s, 'lamda'):
                        reps.append(s.lamda)
                    elif hasattr(s, 'base_set') and hasattr(s, 'function'):
                        reps.append(s.function)
                    else:
                        reps.append(s)
            else:
                reps = [sol]
            numeric_examples = []
            for rep in reps:
                for i in [0, 1, -1]:
                    try:
                        val = rep.subs('n', i).evalf()
                        numeric_examples.append(val)
                    except Exception:
                        continue
            numeric_examples = list(dict.fromkeys(numeric_examples))
            numeric_str = ', '.join([f"{float(x):.3f}" for x in numeric_examples if x is not None])
            return (
                f"General zeros for {var}:\n{sp.pretty(sol)}\n"
                + (f"Example numeric zeros (for n=0,1,-1): {numeric_str}" if numeric_str else "")
            )
        except Exception:
            return f"General zeros for {var}:\n{sp.pretty(sol)}"

    except Exception:
        return None
    
def handle_natural_language(query):
    q = query.lower()

    # --- PATCH: Handle temperature conversion phrasing for Kelvin ---
    m = re.match(r'^\s*(?:convert\s*)?([0-9.\-]+)\s*(kelvin|k)\s*(is\s*)?(to|in|into)\s*(celsius|centigrade|degc)\s*$', q)
    if m:
        kelvin = float(m.group(1))
        celsius = kelvin - 273.15
        from_unit = m.group(2)
        to_unit = m.group(5) if m.lastindex >= 5 else "celsius"
        from_unit_symbol = pretty_temp_unit(from_unit)
        to_unit_symbol = pretty_temp_unit(to_unit)
        return f"{kelvin:.2f} {from_unit_symbol} is equal to {celsius:.2f} {to_unit_symbol}"

    # --- NEW: Patch for Celsius to Kelvin ---
    m = re.match(r'^\s*(?:convert\s*)?([0-9.\-]+)\s*(c|celsius|centigrade|degc)\s*(is\s*)?(to|in|into)\s*(kelvin|k)\s*$', q)
    if m:
        celsius = float(m.group(1))
        kelvin = celsius + 273.15
        from_unit = m.group(2)
        to_unit = m.group(5) if m.lastindex >= 5 else "kelvin"
        from_unit_symbol = pretty_temp_unit(from_unit)
        to_unit_symbol = pretty_temp_unit(to_unit)
        return f"{celsius:.2f} {from_unit_symbol} is equal to {kelvin:.2f} {to_unit_symbol}"

    # --- Patch for Fahrenheit to Kelvin ---
    m = re.match(r'^\s*(?:convert\s*)?([0-9.\-]+)\s*(f|fahrenheit|degf)\s*(is\s*)?(to|in|into)\s*(kelvin|k)\s*$', q)
    if m:
        fahrenheit = float(m.group(1))
        kelvin = (fahrenheit - 32) * 5 / 9 + 273.15
        from_unit = m.group(2)
        to_unit = m.group(5) if m.lastindex >= 5 else "kelvin"
        from_unit_symbol = pretty_temp_unit(from_unit)
        to_unit_symbol = pretty_temp_unit(to_unit)
        return f"{fahrenheit:.2f} {from_unit_symbol} is equal to {kelvin:.2f} {to_unit_symbol}"

    # --- Patch for Kelvin to Fahrenheit ---
    m = re.match(r'^\s*(?:convert\s*)?([0-9.\-]+)\s*(kelvin|k)\s*(is\s*)?(to|in|into)\s*(f|fahrenheit|degf)\s*$', q)
    if m:
        kelvin = float(m.group(1))
        fahrenheit = (kelvin - 273.15) * 9 / 5 + 32
        from_unit = m.group(2)
        to_unit = m.group(5) if m.lastindex >= 5 else "fahrenheit"
        from_unit_symbol = pretty_temp_unit(from_unit)
        to_unit_symbol = pretty_temp_unit(to_unit)
        return f"{kelvin:.2f} {from_unit_symbol} is equal to {fahrenheit:.2f} {to_unit_symbol}"
    
    # Celsius to Fahrenheit (very flexible)
    match = re.search(
        r'(?:convert\s*)?(?:what\s*)?(?:is\s*)?(?:the\s*)?(\d+(\.\d+)?)\s*(c|celsius|centigrade)\s*(is\s*)?(to|in)?\s*f(ahrenheit)?',
        query)
    if match:
        celsius = float(match.group(1))
        fahrenheit = (celsius * 9/5) + 32
        from_unit = match.group(3)
        to_unit = "fahrenheit"
        from_unit_symbol = pretty_temp_unit(from_unit)
        to_unit_symbol = pretty_temp_unit(to_unit)
        return f"{celsius:.2f} {from_unit_symbol} is equal to {fahrenheit:.2f} {to_unit_symbol}"

    # Fahrenheit to Celsius (very flexible)
    match = re.search(
        r'(?:convert\s*)?(?:what\s*)?(?:is\s*)?(?:the\s*)?(\d+(\.\d+)?)\s*f(ahrenheit)?\s*(is\s*)?(to|in)?\s*c(el(si)?us)?',
        query)
    if match:
        fahrenheit = float(match.group(1))
        celsius = (fahrenheit - 32) * 5/9
        from_unit = "fahrenheit"
        to_unit = "celsius"
        from_unit_symbol = pretty_temp_unit(from_unit)
        to_unit_symbol = pretty_temp_unit(to_unit)
        return f"{fahrenheit:.2f} {from_unit_symbol} is equal to {celsius:.2f} {to_unit_symbol}"

    # NEW: Handle "change X degrees to Fahrenheit" or "change X degrees to Celsius" without explicit C/F
    match = re.search(
        r'change\s*(\d+(\.\d+)?)\s*degrees?\s*to\s*(fahrenheit|celsius|centigrade)',
        query)
    if match:
        value = float(match.group(1))
        target = match.group(3)
        # Guess source unit: if target is fahrenheit, assume celsius, and vice versa
        if target.startswith('f'):
            from_unit = "celsius"
            to_unit = "fahrenheit"
            fahrenheit = (value * 9/5) + 32
            from_unit_symbol = pretty_temp_unit(from_unit)
            to_unit_symbol = pretty_temp_unit(to_unit)
            # PATCH HERE: Add a space between value and symbol in output string
            return f"{value:.2f} {from_unit_symbol} is equal to {fahrenheit:.2f} {to_unit_symbol}"
        else:
            from_unit = "fahrenheit"
            to_unit = "celsius"
            celsius = (value - 32) * 5/9
            from_unit_symbol = pretty_temp_unit(from_unit)
            to_unit_symbol = pretty_temp_unit(to_unit)
            # PATCH HERE: Add a space between value and symbol in output string
            return f"{value:.2f} {from_unit_symbol} is equal to {celsius:.2f} {to_unit_symbol}"

    # --- PATCH: Handle "subtract X from Y" or "what do I get if I subtract X from Y"
    match = re.search(r'(?:subtract|minus)\s*(\d+(\.\d+)?)\s*from\s*(\d+(\.\d+)?)', q)
    if match:
        subtrahend = float(match.group(1))
        minuend = float(match.group(3))
        result = minuend - subtrahend
        return f"{minuend} minus {subtrahend} is {result:.2f}"

    match = re.search(r'what do i get if i subtract\s*(\d+(\.\d+)?)\s*from\s*(\d+(\.\d+)?)', q)
    if match:
        subtrahend = float(match.group(1))
        minuend = float(match.group(3))
        result = minuend - subtrahend
        return f"{minuend} minus {subtrahend} is {result:.2f}"
    
    # Abbrev/typo mapping
    abbr = [
        (r"whts", "what's"),
        (r"\bwht\b", "what"),
        (r"\br\b", "radius"),
        (r"=\s*", "="),
        (r"\?", ""),
    ]
    for src, dst in abbr:
        q = re.sub(src, dst, q, flags=re.IGNORECASE)

    # Flexible circle area
    circle_area_pattern = re.search(
        r"(?:area (?:of )?circle|circle area)[^0-9a-zA-Z]*(?:radius|r)?[^\d-]*([0-9.\-]+)", q)
    if circle_area_pattern:
        radius = float(circle_area_pattern.group(1))
        area = math.pi * radius ** 2
        out = f"The area of a circle with radius {radius} is {area:.2f} units²"
        if not out.endswith('.'):
            out += '.'
        return out

    # Also accept phrases like "area circle = 20"
    m = re.match(r"area (?:of )?circle[^\d]*=([0-9.\-]+)", q)
    if m:
        radius = float(m.group(1))
        area = math.pi * radius ** 2
        out = f"The area of a circle with radius {radius} is {area:.2f} units²"
        if not out.endswith('.'):
            out += '.'
        return out

    # Area of a circle
    match = re.search(r'area of (a )?circle.*radius[^\d]*(\d+(\.\d+)?)', query)
    if match:
        radius = float(match.group(2))
        area = math.pi * radius ** 2
        out = f"The area of a circle with radius {radius} is {area:.2f} units²"
        if not out.endswith('.'):
            out += '.'
        return out

    # Volume of a sphere
    match = re.search(r'volume of (a )?sphere.*radius[^\d]*(\d+(\.\d+)?)', query)
    if match:
        radius = float(match.group(2))
        volume = (4/3) * math.pi * radius ** 3
        out = f"The volume of a sphere with radius {radius} is {volume:.2f} units³"
        if not out.endswith('.'):
            out += '.'
        return out

    # Surface area of a sphere (allow diameter or radius)
    match = re.search(r'surface area of (a )?sphere.*diameter[^\d]*(\d+(\.\d+)?)', query)
    if match:
        diameter = float(match.group(2))
        radius = diameter / 2
        area = 4 * math.pi * radius ** 2
        out = f"The surface area of a sphere with diameter {diameter} (radius {radius}) is {area:.2f} units²"
        if not out.endswith('.'):
            out += '.'
        return out
    match = re.search(r'surface area of (a )?sphere.*radius[^\d]*(\d+(\.\d+)?)', query)
    if match:
        radius = float(match.group(2))
        area = 4 * math.pi * radius ** 2
        out = f"The surface area of a sphere with radius {radius} is {area:.2f} units²"
        if not out.endswith('.'):
            out += '.'
        return out

    # Volume of a cylinder (allow radius/diameter and height)
    match = re.search(
        r'volume of (a )?cylinder.*radius[^\d]*(\d+(\.\d+)?).*height[^\d]*(\d+(\.\d+)?)', query)
    if match:
        radius = float(match.group(2))
        height = float(match.group(4))
        volume = math.pi * radius ** 2 * height
        out = f"The volume of a cylinder with radius {radius} and height {height} is {volume:.2f} units³"
        if not out.endswith('.'):
            out += '.'
        return out
    match = re.search(
        r'volume of (a )?cylinder.*diameter[^\d]*(\d+(\.\d+)?).*height[^\d]*(\d+(\.\d+)?)', query)
    if match:
        diameter = float(match.group(2))
        radius = diameter / 2
        height = float(match.group(4))
        volume = math.pi * radius ** 2 * height
        out = f"The volume of a cylinder with diameter {diameter} (radius {radius}) and height {height} is {volume:.2f} units³"
        if not out.endswith('.'):
            out += '.'
        return out

    # --- PATCH: Improved general perimeter/area/volume regex support for chatty forms ---
    # E.g. "if a rectangle has length 8 and width 3, what's the perimeter"

    # Perimeter of rectangle (flexible)
    match = re.search(
        r"(?:perimeter.*rectangle.*length[^\d]*(\d+(\.\d+)?).*width[^\d]*(\d+(\.\d+)?))|"
        r"(?:rectangle.*length[^\d]*(\d+(\.\d+)?).*width[^\d]*(\d+(\.\d+)?).*(?:perimeter|what.?s the perimeter|what is the perimeter|find the perimeter))|"
        r"(?:if a rectangle has length[^\d]*(\d+(\.\d+)?).*width[^\d]*(\d+(\.\d+)?)[^?]*perimeter)",
        query)
    if match:
        # Find all numbers: the first is length, the second is width
        nums = [float(g) for g in match.groups() if g and re.match(r'^\d+(\.\d+)?$', g)]
        if len(nums) >= 2:
            length = nums[0]
            width = nums[1]
            perimeter = 2 * (length + width)
            out = f"The perimeter of a rectangle with length {length} and width {width} is {perimeter:.2f} units"
            if not out.endswith('.'):
                out += '.'
            return out
        # fallback, not enough matches

    # The rest of the function unchanged, except:  
    # MOVE the original rectangle perimeter block below this PATCH  
    # The following is the original snippet moved to not shadow the improved match  
    match = re.match(r'perimeter of (a )?rectangle.*length[^\d]*(\d+(\.\d+)?).*width[^\d]*(\d+(\.\d+)?)', query)
    if match:
        length = float(match.group(2))
        width = float(match.group(4))
        perimeter = 2 * (length + width)
        out = f"The perimeter of a rectangle with length {length} and width {width} is {perimeter:.2f} units"
        if not out.endswith('.'):
            out += '.'
        return out

    # Perimeter of a rectangle
    match = re.search(r'perimeter of (a )?rectangle.*length[^\d]*(\d+(\.\d+)?).*width[^\d]*(\d+(\.\d+)?)', query)
    if match:
        length = float(match.group(2))
        width = float(match.group(4))
        perimeter = 2 * (length + width)
        out = f"The perimeter of a rectangle with length {length} and width {width} is {perimeter:.2f} units"
        if not out.endswith('.'):
            out += '.'
        return out

    # Area of a triangle (very flexible)
    match = re.search(
        r'(?:calculate\s*)?(?:the\s*)?area of (?:a\s*)?triangle.*?(?:base\s*(?:is)?\s*(\d+(\.\d+)?)).*?(?:height\s*(?:is)?\s*(\d+(\.\d+)?))',
        query)
    if match:
        base = float(match.group(1))
        height = float(match.group(3))
        area = 0.5 * base * height
        out = f"The area of a triangle with base {base} and height {height} is {area:.2f} units²"
        if not out.endswith('.'):
            out += '.'
        return out

    # Also handle "triangle area" or "area of triangle" with base/height in any order
    match = re.search(
        r'(triangle area|area of (a )?triangle)[^0-9a-zA-Z]*(?:base[^\d]*(\d+(\.\d+)?)[^0-9a-zA-Z]+height[^\d]*(\d+(\.\d+)?)|height[^\d]*(\d+(\.\d+)?)[^0-9a-zA-Z]+base[^\d]*(\d+(\.\d+)?))',
        query)
    if match:
        nums = [float(g) for g in match.groups() if g and re.match(r'^\d+(\.\d+)?$', g)]
        if len(nums) == 2:
            area = 0.5 * nums[0] * nums[1]
            out = f"The area of a triangle with base {nums[0]} and height {nums[1]} is {area:.2f} units²"
            if not out.endswith('.'):
                out += '.'
            return out

    return None

# PATCH: Fix numword_to_digit to always return a string, and improve handling in nl_simple_math_to_expression

def numword_to_digit(s):
    s = s.strip()
    val = words_phrase_to_number(s)
    if val is not None:
        return str(val)
    # PATCH: If not a number word, return the original string (not None)
    return s

# --- PATCH: Move 'add ... to ...' and 'subtract ... from ...' BEFORE aggregate 'add ... and ...' and similar ---
def nl_simple_math_to_expression(text):
    # --- PATCH: If the expression is already a numeric infix expression with an operator, return as is ---
    if re.fullmatch(r"\s*[\d]+\s*[\+\-\*/]\s*[\d]+\s*", text):
        return text

    text = replace_number_words(text)
    text = text.lower().strip()

    # 1. Robust: "quotient of X divided by Y" or "quotient of X divide by Y" (X / Y)
    m = re.fullmatch(r'quotient of\s+(.+?)\s+divid(?:e|ed) by\s+(.+)', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} / {right_expr}"

    # 2. Robust: "quotient of X and Y" (X / Y)
    m = re.fullmatch(r'quotient of\s+(.+?)\s+and\s+(.+)', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} / {right_expr}"

    # --- PATCH: Handle "remainder of X divided by Y" correctly as X % Y ---
    m = re.match(r'(?:what(\'s| is)? )?remainder of\s+(.+?)\s+divid(?:e|ed) by\s+(.+)', text)
    if m:
        x = m.group(2).strip()
        y = m.group(3).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} % {y_expr}"

    # ... [entire rest of your existing original function unchanged] ...

    # PATCH: Handle "remainder when X is divided by Y", "remainder when X divided by Y", and "modulus of X and Y"
    m = re.match(r'(?:what(\'s| is)? )?remainder when\s+(.+?)\s+(?:is )?divid(?:e|ed) by\s+(.+)', text)
    if m:
        x = m.group(2).strip()
        y = m.group(3).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} % {y_expr}"

    # --- PATCH: Handle 'add X to Y' and 'subtract X from Y' with greedy matching EARLY ---
    m = re.match(r'^add\s+(.+?)\s+to\s+(.+)$', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{right_expr} + {left_expr}"
    m = re.match(r'^subtract\s+(.+?)\s+from\s+(.+)$', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{right_expr} - {left_expr}"

    # --- PATCH: Handle "half/double/triple of X and [multiply/divide/add/subtract] by Y" as sequential operation ---
    m = re.match(
        r"^(half|double|triple)\s+of\s+([a-zA-Z0-9\.\- ]+?)\s+and\s+(multipl(?:y|ied)|divide|add|subtract)(?: by)?\s+([a-zA-Z0-9\.\- ]+)$",
        text
    )
    if m:
        op1 = m.group(1)
        x_raw = m.group(2).strip()
        op2 = m.group(3)
        y_raw = m.group(4).strip()
        x_val = words_phrase_to_number(x_raw)
        y_val = words_phrase_to_number(y_raw)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x_raw)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y_raw)
        if op1 == "half":
            left = f"({x_expr}) / 2"
        elif op1 == "double":
            left = f"({x_expr}) * 2"
        elif op1 == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        op_map = {
            "multiplied": "*", "multiply": "*",
            "divided": "/", "divide": "/",
            "add": "+", "subtract": "-",
        }
        # Accept both 'multiply' and 'multiplied', 'divide' and 'divided'
        op2_key = op2
        if op2_key.endswith("ed"):
            op2_key = op2_key[:-2]
        op_sym = op_map.get(op2_key, None)
        if op_sym is None:
            op_sym = op_map.get(op2, "?")
        return f"{left} {op_sym} {y_expr}"

    # --- PATCH: Handle "[half|double|triple] N and [add|subtract|multiply|divide] by M" robustly ---
    # e.g., "half 98 and multiplied by 2" -> (98/2)*2
    m = re.match(r'^(half|double|triple)\s+([-\w\s\.]+?)\s+and\s+((?:multipl(?:y|ied)|divide|add|subtract)(?: by)?)\s+([-\w\s\.]+)$', text)
    if m:
        op1 = m.group(1)
        num1_raw = m.group(2)
        op2 = m.group(3)
        num2_raw = m.group(4)
        num1_val = words_phrase_to_number(num1_raw)
        num2_val = words_phrase_to_number(num2_raw)
        num1_expr = str(num1_val) if num1_val is not None else numword_to_digit(num1_raw)
        num2_expr = str(num2_val) if num2_val is not None else numword_to_digit(num2_raw)

        # Compose left value
        if op1 == "half":
            left = f"({num1_expr}) / 2"
        elif op1 == "double":
            left = f"({num1_expr}) * 2"
        elif op1 == "triple":
            left = f"({num1_expr}) * 3"
        else:
            left = num1_expr

        # Compose second (sequential) operation: map verbal op to symbol
        op_map = {
            "multiplied by": "*",
            "multiply by": "*",
            "multiply": "*",
            "divided by": "/",
            "divide by": "/",
            "divide": "/",
            "add": "+",
            "added by": "+",
            "subtract": "-",
            "subtracted by": "-",
        }
        op2_key = op2.strip()
        if op2_key not in op_map and op2_key.endswith('ed by'):
            # Fix typos for e.g. "multiplied by"
            op2_key = op2_key.replace('ed by', 'ied by')
        op_sym = op_map.get(op2_key, None)
        if op_sym is None:
            # fallback: try to remove trailing " by"
            op2_key = op2_key.replace(' by', '')
            op_sym = op_map.get(op2_key, None)
        if op_sym is None:
            return f"{left} ? {num2_expr}"
        return f"{left} {op_sym} {num2_expr}"
    
    # --- PATCH START: Handle patterns like 'half 98 and [operation] by [number]' as sequential ---
    # E.g., 'half 98 and divide by 2' → ((98)/2)/2, 'half 98 and multiply by 2' → ((98)/2)*2
    m = re.match(r'^(half|double|triple)\s+([-\w\s\.]+?)\s+and\s+(divide|multiply|add|subtract)\s+by\s+([-\w\s\.]+)$', text)
    if m:
        op1, num1_raw, op2, num2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        num1_val = words_phrase_to_number(num1_raw)
        num2_val = words_phrase_to_number(num2_raw)
        num1_expr = str(num1_val) if num1_val is not None else numword_to_digit(num1_raw)
        num2_expr = str(num2_val) if num2_val is not None else numword_to_digit(num2_raw)
        # Compose left value
        if op1 == "half":
            left = f"({num1_expr}) / 2"
        elif op1 == "double":
            left = f"({num1_expr}) * 2"
        elif op1 == "triple":
            left = f"({num1_expr}) * 3"
        else:
            left = num1_expr
        # Compose second (sequential) operation
        op_map = {
            "divide": "/",
            "multiply": "*",
            "add": "+",
            "subtract": "-",
        }
        op_sym = op_map[op2]
        return f"({left}) {op_sym} {num2_expr}"

    # --- PATCH END ---
    
    # --- PATCH: Robustly handle 'half|double|triple of X and plus|minus|...' as '(X/2) op Y' ---
    # Before:
    # m = re.match(r'^(half|double|triple)\s+of\s+([a-zA-Z0-9\.\- ]+?)\s+and\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\- ]+)$', text)
    # Now:
    m = re.match(
        r'^(half|double|triple)\s+of\s+([a-zA-Z0-9\.\- ]+?)\s+and\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\- ]+)$',
        text
    )
    if m:
        op = m.group(1)
        # PATCH: Strip trailing "and" from X (if any); avoids X='98 and'
        x = m.group(2).strip()
        if x.endswith(" and"):
            x = x[:-4].rstrip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        # Now apply the second operation
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        elif op2 in ["times", "multiply by", "multiplied by"]:
            return f"{left} * {y_expr}"
        elif op2 in ["divide by", "divided by", "over"]:
            return f"{left} / {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # --- PATCH: Robustly handle 'half|double|triple of X plus|minus|...' as '(X/2) op Y' ---
    m = re.match(r'^(half|double|triple)\s+of\s+([a-zA-Z0-9\.\- ]+?)\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        # Now apply the second operation
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        elif op2 in ["times", "multiply by", "multiplied by"]:
            return f"{left} * {y_expr}"
        elif op2 in ["divide by", "divided by", "over"]:
            return f"{left} / {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # --- PATCH: Robustly handle 'half|double|triple X plus|minus|...' as '(X/2) op Y' ---
    m = re.match(r'^(half|double|triple)\s+([a-zA-Z0-9\.\- ]+?)\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        elif op2 in ["times", "multiply by", "multiplied by"]:
            return f"{left} * {y_expr}"
        elif op2 in ["divide by", "divided by", "over"]:
            return f"{left} / {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # --- PATCH: Handle even more combined patterns above here as needed ---

    # -------- *Now* handle 'half/double/triple of X' *alone* (must come after above) ------
    m = re.match(r'^(half|double|triple)\s+of\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        if op == "half":
            return f"({x_expr}) / 2"
        elif op == "double":
            return f"({x_expr}) * 2"
        elif op == "triple":
            return f"({x_expr}) * 3"
        else:
            return x_expr

    # -------- Also handle 'half/double/triple X' *alone* ------
    m = re.match(r'^(half|double|triple)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        if op == "half":
            return f"({x_expr}) / 2"
        elif op == "double":
            return f"({x_expr}) * 2"
        elif op == "triple":
            return f"({x_expr}) * 3"
        else:
            return x_expr

    # ... (rest of the function remains unchanged, as per instructions) ...

    # PATCH: Robustly handle 'subtract X from Y' and 'X subtracted from Y' for all digit/number-word phrases
    m = re.match(r'^subtract\s+([a-zA-Z0-9\s\-\.]+)\s+from\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{y_expr} - {x_expr}"

    m = re.match(r'^([a-zA-Z0-9\s\-\.]+)\s+subtracted from\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{y_expr} - {x_expr}"

    # --- PATCH: Handle 'subtract X by Y' and 'minus X by Y' as 'X - Y' (bugfix for e.g. 'subtract five hundred by fifty') ---
    m = re.match(r'^(subtract|minus)\s+([-\w\s\.]+)\s+by\s+([-\w\s\.]+)$', text)
    if m:
        x = m.group(2).strip()
        y = m.group(3).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} - {y_expr}"

    # --- PATCH: Handle 'subtract X and Y' or 'minus X and Y' as 'X - Y' ---
    m = re.match(r'^(subtract|minus)\s+(.+?)\s+and\s+(.+)$', text)
    if m:
        left = m.group(2).strip()
        right = m.group(3).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        # Only handle if both sides are numbers
        if left_val is not None and right_val is not None:
            left_expr = str(left_val)
            right_expr = str(right_val)
            return f"{left_expr} - {right_expr}"
        # else fall through to other logic

    # --- NEW PATCH: Handle 'adding X and Y', 'subtracting X and Y', etc. ---
    m = re.match(r'^(adding|subtracting|multiplying|dividing)\s+(.+?)\s+and\s+(.+)$', text)
    if m:
        op = m.group(1)
        left = m.group(2).strip()
        right = m.group(3).strip()
        op_map = {
            'adding': '+',
            'subtracting': '-',
            'multiplying': '*',
            'dividing': '/',
        }
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        # PATCH: Only handle if both sides are numbers; else fall through
        if left_val is not None and right_val is not None:
            left_expr = str(left_val)
            right_expr = str(right_val)
            return f"{left_expr} {op_map[op]} {right_expr}"
        # else, treat as a single phrase, let below logic handle

    # --- PATCH: Handle 'sum of X and Y' and similar patterns at the start ---
    m = re.match(r'^(sum|product|difference|quotient) of\s+(.+?)\s+and\s+(.+)$', text, flags=re.IGNORECASE)
    if m:
        op_word = m.group(1).lower()
        left = m.group(2).strip()
        right = m.group(3).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        op_map = {
            "sum": "+",
            "product": "*",
            "difference": "-",
            "quotient": "/"
        }
        op = op_map[op_word]
        return f"{left_expr} {op} {right_expr}"

    # --- PATCH: Only use the add/sum/plus ... and ... shortcut if BOTH parts are clearly numbers ---
    m = re.match(r'^(add|sum|plus)\s+(.+?)\s+and\s+(.+)$', text)
    if m:
        op = m.group(1)
        left = m.group(2).strip()
        right = m.group(3).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        if left_val is not None and right_val is not None:
            left_expr = str(left_val)
            right_expr = str(right_val)
            return f"{left_expr} + {right_expr}"
        # else, fall through -- treat as a single number phrase

    # --- Ensure operator words are ALWAYS replaced with math symbols ---
    # Order matters: longest phrases first!
    op_replace = [
        ("multiplied by", "*"),
        ("multiply by", "*"),
        ("divide by", "/"),
        ("divided by", "/"),
        ("added to", "+"),
        ("minus", "-"),
        ("plus", "+"),
        ("times", "*"),
        ("over", "/"),
        ("add", "+"),
        ("subtract", "-"),
    ]
    # Greedy replace, longest first
    for phrase, op in op_replace:
        text = re.sub(rf"\b{re.escape(phrase)}\b", f" {op} ", text)
    # Remove redundant spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

    # PATCH: Handle "modulus of X and Y"
    m = re.match(r'(?:what(\'s| is)? )?modulus of\s+(.+?)\s+and\s+(.+)', text)
    if m:
        x = m.group(2).strip()
        y = m.group(3).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} % {y_expr}"

    # PATCH: Handle "modulus X and Y"
    m = re.match(r'(?:what(\'s| is)? )?modulus\s+(.+?)\s+and\s+(.+)', text)
    if m:
        x = m.group(2).strip()
        y = m.group(3).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} % {y_expr}"

    # PATCH: Handle "X mod Y"
    m = re.match(r'(.+?)\s+mod\s+(.+)', text)
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} % {y_expr}"

    # PATCH: Handle "remainder of X and Y" (another phrasing)
    m = re.match(r'(?:what(\'s| is)? )?remainder of\s+(.+?)\s+and\s+(.+)', text)
    if m:
        x = m.group(2).strip()
        y = m.group(3).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} % {y_expr}"

    # ... rest of the original implementation unchanged ...

    # 3. "sum of X and Y"
    m = re.fullmatch(r'sum of\s+(.+?)\s+and\s+(.+)', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} + {right_expr}"

    # 4. "product of X multiplied by Y" or "product of X times Y"
    m = re.fullmatch(r'product of\s+(.+?)\s+(?:multipl(?:y|ied) by|times)\s+(.+)', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} * {right_expr}"

    # 5. "product of X and Y"
    m = re.fullmatch(r'product of\s+(.+?)\s+and\s+(.+)', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} * {right_expr}"

    # 6. "difference of X and Y"
    m = re.fullmatch(r'difference of\s+(.+?)\s+and\s+(.+)', text)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} - {right_expr}"

    # ...rest of your original code...

    # Fallback: remove known trailing non-math phrases and retry
    fallback_patterns = [
        r'(.*)\s+now it says this',
        r'(.*)\s+please',
        r'(.*)\s+thanks',
        r'(.*)\s+thank you',
        r'(.*)\s+show me',
    ]
    for pat in fallback_patterns:
        m = re.match(pat, text)
        if m:
            text = m.group(1).strip()
            # Try again, recursively, after stripping trailing phrase
            return nl_simple_math_to_expression(text)

    # If still not a direct math expression, return as is
    return text

    # --- PATCH: Handle 'double of X and minus Y' and similar patterns ---
    m = re.match(r'^(double|triple|half)\s+of\s+([a-zA-Z0-9\.\- ]+?)\s+and\s+(minus|plus|add|subtract)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        #  --- FIX: Use correct multiplier/divider for double/triple/half ---
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        # Apply second operation
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # --- PATCH: Handle 'double X and minus Y' and similar patterns ---
    m = re.match(r'^(double|triple|half)\s+([a-zA-Z0-9\.\- ]+?)\s+and\s+(minus|plus|add|subtract)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # --- PATCH: Handle 'triple/double/half of X plus/minus Y' ---
    m = re.match(r'^(double|triple|half)\s+of\s+([a-zA-Z0-9\.\- ]+?)\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        elif op2 in ["times", "multiply by", "multiplied by"]:
            return f"{left} * {y_expr}"
        elif op2 in ["divide by", "divided by", "over"]:
            return f"{left} / {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # --- PATCH: Handle 'triple/double/half of X' alone ---
    m = re.match(r'^(double|triple|half)\s+of\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        if op == "half":
            return f"({x_expr}) / 2"
        elif op == "double":
            return f"({x_expr}) * 2"
        elif op == "triple":
            return f"({x_expr}) * 3"
        else:
            return x_expr

    # --- PATCH: Handle 'triple/double/half X' alone ---
    m = re.match(r'^(double|triple|half)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        if op == "half":
            return f"({x_expr}) / 2"
        elif op == "double":
            return f"({x_expr}) * 2"
        elif op == "triple":
            return f"({x_expr}) * 3"
        else:
            return x_expr

    # --- PATCH: Handle 'divide X by Y' and 'multiply X by Y' EARLY ---
    m = re.match(r'^divide\s+([a-zA-Z0-9\s\-\.]+)\s+by\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} / {y_expr}"
    m = re.match(r'^multiply\s+([a-zA-Z0-9\s\-\.]+)\s+by\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} * {y_expr}"

    # --- PATCH: Handle 'double/triple/half X minus Y' and 'double/triple/half X plus Y' ---
    # MOVE THIS BLOCK ABOVE THE SIMPLER 'double X' BLOCK
    m = re.match(r'^(double|triple|half)\s+([a-zA-Z0-9\.\- ]+?)\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        # Always use 2 for double, 3 for triple, 0.5 for half
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        # Now apply the second operation
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        elif op2 in ["times", "multiply by", "multiplied by"]:
            return f"{left} * {y_expr}"
        elif op2 in ["divide by", "divided by", "over"]:
            return f"{left} / {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # --- PATCH: Handle 'double/triple/half X' alone ---
    m = re.match(r'^(double|triple|half)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        if op == "half":
            return f"({x_expr}) / 2"
        elif op == "double":
            return f"({x_expr}) * 2"
        elif op == "triple":
            return f"({x_expr}) * 3"
        else:
            return x_expr

    # --- PATCH: Handle "double X and minus Y" and similar patterns ---
    m = re.match(
        r"^(double|triple|half)\s+([-\w\s\.]+)\s+and\s+(add|plus|subtract|minus|multiply|times|divide|over)\s+([-\w\s\.]+)$",
        text)
    if m:
        op1, num1_raw, op2, num2_raw = m.group(1), m.group(2), m.group(3), m.group(4)
        num1 = words_phrase_to_number(num1_raw)
        num2 = words_phrase_to_number(num2_raw)
        if num1 is None or num2 is None:
            return None
        # Apply first operation
        if op1 == "double":
            val = num1 * 2
        elif op1 == "triple":
            val = num1 * 3
        elif op1 == "half":
            val = num1 / 2
        else:
            val = num1
        # Apply second operation
        if op2 in ["add", "plus", "multiply", "times"]:
            result = f"{val} + {num2}" if op2 in ["add", "plus"] else f"{val} * {num2}"
        elif op2 in ["subtract", "minus", "divide", "over"]:
            result = f"{val} - {num2}" if op2 in ["subtract", "minus"] else f"{val} / {num2}"
        else:
            return None
        return result

    # --- PATCH: Fix for 'double/triple/half X plus/minus Y' ---
    m = re.match(r'^(double|triple|half)\s+([a-zA-Z0-9\.\- ]+?)\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\- ]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        # Always use 2 for double, 3 for triple, 0.5 for half
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        # Now apply the second operation
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        elif op2 in ["times", "multiply by", "multiplied by"]:
            return f"{left} * {y_expr}"
        elif op2 in ["divide by", "divided by", "over"]:
            return f"{left} / {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # PATCH: Handle 'double/triple/half <number>' alone
    m = re.match(r'^(half|double|triple)\s+of\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        if op == "half":
            return f"({x_expr}) / 2"
        elif op == "double":
            return f"({x_expr}) * 2"
        elif op == "triple":
            return f"({x_expr}) * 3"
        else:
            return x_expr

    # --- PATCH: Handle 'divide X and Y' and 'multiply X and Y' (less common, but for completeness) ---
    m = re.match(r'^divide\s+([a-zA-Z0-9\s\-\.]+)\s+and\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} / {y_expr}"
    m = re.match(r'^multiply\s+([a-zA-Z0-9\s\-\.]+)\s+and\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        return f"{x_expr} * {y_expr}"

    # --- PATCH: Handle 'double/triple/half <number> multiply by <number>' and similar patterns ---
    op_words = ["plus", "minus", "add", "subtract", "multiply by", "multiplied by", "times", "divide by", "divided by", "over"]
    for op_word in op_words:
        op_pat = rf'^(double|triple|half)\s+([a-zA-Z0-9\.\-]+)\s+{op_word}\s+([a-zA-Z0-9\.\-]+)$'
        m = re.match(op_pat, text)
        if m:
            op = m.group(1)
            x = m.group(2).strip()
            y = m.group(3).strip()
            x_val = words_phrase_to_number(x)
            y_val = words_phrase_to_number(y)
            x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
            y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
            # Always use 2 for double, 3 for triple, 0.5 for half
            if op == "half":
                left = f"({x_expr}) / 2"
            elif op == "double":
                left = f"({x_expr}) * 2"
            elif op == "triple":
                left = f"({x_expr}) * 3"
            else:
                left = x_expr
            if op_word in ["plus", "add"]:
                return f"{left} + {y_expr}"
            elif op_word in ["minus", "subtract"]:
                return f"{left} - {y_expr}"
            elif op_word in ["multiply by", "multiplied by", "times"]:
                return f"{left} * {y_expr}"
            elif op_word in ["divide by", "divided by", "over"]:
                return f"{left} / {y_expr}"
            else:
                return f"{left} {op_word} {y_expr}"

    # PATCH: Handle 'double/triple/half <number> plus/minus <number>' and similar patterns correctly
    m = re.match(r'^(double|triple|half)\s+([a-zA-Z0-9\.\-]+)\s+(plus|minus|add|subtract|times|multiply by|multiplied by|divide by|divided by|over)\s+([a-zA-Z0-9\.\-]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        op2 = m.group(3)
        y = m.group(4).strip()
        x_val = words_phrase_to_number(x)
        y_val = words_phrase_to_number(y)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        y_expr = str(y_val) if y_val is not None else numword_to_digit(y)
        # Always use 2 for double, 3 for triple, 0.5 for half
        if op == "half":
            left = f"({x_expr}) / 2"
        elif op == "double":
            left = f"({x_expr}) * 2"
        elif op == "triple":
            left = f"({x_expr}) * 3"
        else:
            left = x_expr
        # Now apply the second operation
        if op2 in ["plus", "add"]:
            return f"{left} + {y_expr}"
        elif op2 in ["minus", "subtract"]:
            return f"{left} - {y_expr}"
        elif op2 in ["times", "multiply by", "multiplied by"]:
            return f"{left} * {y_expr}"
        elif op2 in ["divide by", "divided by", "over"]:
            return f"{left} / {y_expr}"
        else:
            return f"{left} {op2} {y_expr}"

    # PATCH: Handle 'half/double/triple of X' alone
    m = re.match(r'^(half|double|triple)\s+of\s+([a-zA-Z0-9\s\-\.]+)$', text)
    if m:
        op = m.group(1)
        x = m.group(2).strip()
        x_val = words_phrase_to_number(x)
        x_expr = str(x_val) if x_val is not None else numword_to_digit(x)
        if op == "half":
            return f"({x_expr}) / 2"
        elif op == "double":
            return f"({x_expr}) * 2"
        elif op == "triple":
            return f"({x_expr}) * 3"
        else:
            return x_expr

    # --- PATCH: Handle 'sum/product/difference/quotient of X and Y' ---
    sum_pat = re.match(r'^(sum|product|difference|quotient)\s+of\s+(.+?)\s+and\s+(.+)$', text)
    if sum_pat:
        op_word = sum_pat.group(1)
        left = sum_pat.group(2).strip()
        right = sum_pat.group(3).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        op_map = {
            "sum": "+",
            "product": "*",
            "difference": "-",
            "quotient": "/"
        }
        op = op_map[op_word]
        return f"{left_expr} {op} {right_expr}"

    # PATCH: Handle "X minus Y" (and similar) BEFORE addition!
    m = re.match(r'^(.+?)\s+(minus|subtract|subtracted by|less)\s+(.+)$', text)
    if m:
        left = m.group(1).strip()
        right = m.group(3).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} - {right_expr}"

    # PATCH: Greedily match multi-word numbers for both X and Y
    m = re.match(r'^(add|sum|plus)\s+(.+?)\s+and\s+(.+)$', text)
    if m:
        op = m.group(1)
        left = m.group(2).strip()
        right = m.group(3).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} + {right_expr}"

    # PATCH: Handle "X plus Y"
    m = re.match(r'^(.+?)\s+(plus|add|added to)\s+(.+)$', text)
    if m:
        left = m.group(1).strip()
        right = m.group(3).strip()
        left_val = words_phrase_to_number(left)
        right_val = words_phrase_to_number(right)
        left_expr = str(left_val) if left_val is not None else numword_to_digit(left)
        right_expr = str(right_val) if right_val is not None else numword_to_digit(right)
        return f"{left_expr} + {right_expr}"

    # Supported operators and their precedence
    op_map = {
        "plus": ("+", 1),
        "add": ("+", 1),
        "added to": ("+", 1),
        "minus": ("-", 1),
        "subtract": ("-", 1),
        "subtracted from": ("-", 1),
        "times": ("*", 2),
        "multiplied by": ("*", 2),
        "multiply by": ("*", 2),
        "multiply": ("*", 2),
        "divided by": ("/", 2),
        "divide by": ("/", 2),
        "divide": ("/", 2),
        "over": ("/", 2),
    }
    op_words = sorted(op_map.keys(), key=lambda x: -len(x))
    tokens = []
    i = 0
    words = text.split()
    while i < len(words):
        matched = False
        for op in op_words:
            op_len = len(op.split())
            if i + op_len <= len(words) and ' '.join(words[i:i+op_len]) == op:
                tokens.append(op)
                i += op_len
                matched = True
                break
        if not matched:
            tokens.append(words[i])
            i += 1

    output = []
    for t in tokens:
        if t in op_map:
            output.append(op_map[t][0])
        else:
            output.append(t)

    expr = ' '.join(output)
    expr = re.sub(r'\s+', ' ', expr).strip()

    function_names = list(allowed_functions.keys())
    if expr.strip() in function_names:
        return expr.strip()

    for func in function_names:
        pat = rf'\b{func}\s+of\s+([-\w\.\+\*/\(\) ]+)$'
        m = re.match(pat, expr)
        if m:
            arg = m.group(1).strip()
            return f"{func}({arg})"

    m = re.match(r'([a-zA-Z0-9\s\-\.]+)\s+added to\s+([a-zA-Z0-9\s\-\.]+)$', expr)
    if m:
        left = numword_to_digit(m.group(1))
        right = numword_to_digit(m.group(2))
        return f"{right} + {left}"

    if re.search(r'\d|\(|\)|\+|\-|\*|\/|\^', expr):
        return expr

    return expr

# Main chatbot loop
def chatbot_calculator():
    print("Hello! I'm your AI chatbot calculator. How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # NEW: Always pass to core logic and print response directly!
        response = chatbot_calculator_logic(user_input)
        print(f"Chatbot: {response}")

def chatbot_calculator_logic(user_input):
    # --- PATCH: Answer 'Who are you?' or variants with a natural language explanation ---
    input_lc = user_input.strip().lower()
    if input_lc in {"who are you", "who are you?", "what are you"}:
        return "I am your AI math chatbot calculator."
    # --- PATCH: Answer 'What do you do?' or 'What do you solve?' with a natural language explanation ---
    input_lc = user_input.strip().lower()
    if input_lc in {
        "what do you do", "what do you do?", "what do you solve", "what do you solve?", 
        "what can you solve", "what can you solve?"
    }:
        return ("I solve math the way you speak it. From simple calculations like addition and subtraction, "
                "to advanced functions like trigonometry, logarithms, and factorials — I’ve got it covered. "
                "I can also handle geometry problems, convert units and temperatures, and even calculate dates. "
                "In short, I make solving problems feel natural, whether it’s for school, work, or everyday life. ")
    try:
        user_input = normalize_nl_input(user_input)
        # --- PATCH: Handle inverse function phrases BEFORE anything else ---
        inverse_match = re.match(
            r"\s*inverse\s*(sine|sin|cosine|cos|tangent|tan)\s*of\s*([\-a-zA-Z0-9_\.\+\*/\(\) ]+)",
            user_input, flags=re.IGNORECASE
        )
        if inverse_match:
            arg = inverse_match.group(2).strip()
            func = inverse_match.group(1).lower()
            if func in ["sin", "sine"]:
                extracted = f"asin({arg})"
            elif func in ["cos", "cosine"]:
                extracted = f"acos({arg})"
            elif func in ["tan", "tangent"]:
                extracted = f"atan({arg})"
            else:
                return f"Sorry, I couldn't understand the function."
            result = calculate(extracted)
            return f"The result of {extracted} is {result}."

        # --- Date offset queries ---
        offset_result = handle_date_offset_query(user_input)
        if offset_result:
            return offset_result

        # --- Geometry/physics/other NL queries ---
        nl_result = handle_natural_language(user_input)
        if nl_result:
            return nl_result

        # --- Date difference queries ---
        date_diff_result = handle_date_difference(user_input)
        if date_diff_result:
            return date_diff_result

        # --- Handle unit conversion queries ---
        unit_conv_result = handle_unit_conversion(user_input)
        if unit_conv_result:
            return unit_conv_result

        cleaned_input = user_input
        corrected_input, was_corrected = correct_math_typos(cleaned_input)
        if was_corrected:
            cleaned_input = corrected_input

        # --- Remove algebraic equality solving ---
        eq_result = handle_equality_comparison_query(cleaned_input)
        if eq_result:
            return eq_result

        # --- PATCH: Handle multi-step NL math queries ---
        multi_step_result = handle_nl_multi_step_math(cleaned_input)
        if multi_step_result:
            return multi_step_result

        # --- PATCH: Targeted fix for 'What do I get if I add ... and ...' natural language queries
        m = re.match(r"what do i get if i (add|sum|plus)\s+(.+?)\s+(?:and|to)\s+(.+)", user_input, flags=re.IGNORECASE)
        if m:
            left_raw = m.group(2).strip()
            right_raw = m.group(3).strip()
            # Prefer number conversion for both, but allow fall-through
            left = replace_number_words(left_raw)
            right = replace_number_words(right_raw)
            try:
                left_val = float(left)
            except Exception:
                left_val = left
            try:
                right_val = float(right)
            except Exception:
                right_val = right
            result = calculate(f"{left_val} + {right_val}")
            return f"The result of {left_val} + {right_val} is {result}."

        # -------- PATCH: Prefer nl_simple_math_to_expression when it yields a valid math expression --------
        nl_expr = nl_simple_math_to_expression(cleaned_input)
        # Only short-circuit if it clearly produced a math expression that evaluates to a non-error/non-passthrough result
        if nl_expr and (re.search(r'[\+\-\*/]', nl_expr) or '(' in nl_expr or ')' in nl_expr):
            result = calculate(nl_expr)
            # Only return if result is a proper number, not a passthrough/error (let fallback logic run otherwise)
            if (
                isinstance(result, (int, float))
                or (
                    isinstance(result, str)
                    and not result.strip().lower().startswith(("error", "sorry", "the result of"))
                    and re.match(r'^-?[\d\.]+$', result.strip())
                )
            ):
                return f"The result of {nl_expr} is {result}."
        # Otherwise, let normal logic continue!

        nl_expr = nl_simple_math_to_expression(cleaned_input)
        if nl_expr and re.fullmatch(r'[\d\s\+\-\*/\^\.]+', nl_expr):
            result = calculate(nl_expr)
            return f"The result of {nl_expr} is {result}."
        
        user_input_stripped = cleaned_input

        expr, result = enhanced_nl_math_parser(user_input_stripped)
        if expr is not None:
            if result is None and isinstance(expr, str):
                # FIX: Strip math command words like "this:" before extracting expression
                expr_clean = strip_math_command_words(expr)
                extracted = extract_math_expression(expr_clean)
                result2 = calculate(extracted)
                return f"The result of '{expr}' is {result2}."
            return f"The result of '{expr}' is {result}."

        extracted = extract_math_expression(strip_math_command_words(nl_expr))
        if isinstance(extracted, str) and extracted.startswith("'") and "inverse" in extracted:
            return extracted

        extracted = extract_math_expression(strip_math_command_words(user_input_stripped))
        if isinstance(extracted, str) and extracted.startswith("'") and "inverse" in extracted:
            return extracted

        if ',' in user_input_stripped or ';' in user_input_stripped:
            cleaned_input2 = user_input_stripped
            result = evaluate_statements(cleaned_input2)
            return result

        if re.match(r'^difference\s+between\s+(.+?)\s+and\s+(.+)$', user_input_stripped.lower()):
            m = re.match(r'^difference\s+between\s+(.+?)\s+and\s+(.+)$', user_input_stripped, re.IGNORECASE)
            if m:
                expr1 = m.group(1).strip()
                expr2 = m.group(2).strip()
                response = difference_between_expressions(expr1, expr2)
                return response

        comparison_result = handle_comparison_queries(user_input_stripped)
        if comparison_result:
            return comparison_result

        greetings = ["hi", "hello", "yo"]
        if user_input_stripped.lower() in greetings:
            return "Hello! How may I help you today?"

        if user_input_stripped.lower() in ["exit", "quit", "bye"]:
            return "Goodbye!"

        if user_input_stripped.lower() in ["help", "what can you do?"]:
            return ("I can do arithmetic, unit conversions, date calculations, geometry, physics, and much more! (Algebraic equation solving, symbolic simplification/expansion are not supported.)")

        parsed_nl = parse_natural_language(user_input_stripped)
        if parsed_nl:
            extracted = extract_math_expression(strip_math_command_words(parsed_nl))
            result = calculate(extracted)
            return f"The result of {extracted} is {result}."

        simple_expr = nl_simple_math_to_expression(user_input_stripped)
        if simple_expr:
            if not is_function_name(simple_expr.strip()):
                extracted = extract_math_expression(strip_math_command_words(simple_expr))
                result = calculate(extracted)
                return f"The result of {extracted} is {result}."

        fallback_expr2 = nl_simple_math_to_expression(user_input_stripped)
        if fallback_expr2 and fallback_expr2 != user_input_stripped:
            if not is_function_name(fallback_expr2.strip()):
                extracted = extract_math_expression(strip_math_command_words(fallback_expr2))
                result = calculate(extracted)
                return f"The result of {extracted} is {result}."

        extracted = extract_math_expression(strip_math_command_words(user_input_stripped))
        if is_function_name(extracted.strip()) and extracted.strip() == user_input_stripped.strip():
            return f"'{extracted.strip()}' is a mathematical function. Please provide an argument, e.g., {extracted.strip()}(x) or {extracted.strip()}(number)."
        if extracted:
            result = calculate(extracted)
            if isinstance(result, str) and (
                result.strip().lower().startswith("the result of")
                or result.strip().lower().startswith("error")
                or result.strip().lower().startswith("sorry")
            ):
                return result
            else:
                return f"The result of {extracted} is {result}."

        return "Sorry, I couldn't find a valid math expression in your input. Please try again."

    except Exception as e:
        return f"Oops! I ran into a problem: {e}"

# Run the chatbot
if __name__ == "__main__":
    chatbot_calculator()
