import os
import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Available UPOS tags
TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
    "SCONJ", "VERB", "X", "SYM"
]


def robust_json_parse(text, expected_words):
    """Multiple strategies for parsing JSON from model output"""
    
    print(f"\n=== DEBUGGING JSON PARSE ===")
    print(f"Raw model output: '{text}'")
    print(f"Expected words: {expected_words}")
    
    # Strategy 1: Direct parsing
    try:
        result = json.loads(text.strip())
        print(f"Strategy 1 - Direct JSON parse successful: {result}")
        if validate_json_structure(result, expected_words):
            print(f"Strategy 1 - Validation passed")
            return result
        else:
            print(f"Strategy 1 - Validation failed")
    except Exception as e:
        print(f"Strategy 1 - Direct JSON parse failed: {e}")
    
    # Strategy 2: Extract JSON block with multiple patterns
    json_patterns = [
        r'\[.*?\]',  # Find array (non-greedy)
        r'\[.*\]',   # Find array (greedy)
        r'\{.*?\}',  # Find single object (non-greedy)
        r'\{.*\}',   # Find single object (greedy)
    ]
    
    for i, pattern in enumerate(json_patterns):
        matches = re.findall(pattern, text, re.DOTALL)
        print(f"Strategy 2.{i+1} - Pattern '{pattern}' found {len(matches)} matches")
        for j, match in enumerate(matches):
            print(f"  Match {j+1}: '{match[:100]}{'...' if len(match) > 100 else ''}'")
            try:
                cleaned = fix_json_formatting(match)
                print(f"  Cleaned: '{cleaned[:100]}{'...' if len(cleaned) > 100 else ''}'")
                result = json.loads(cleaned)
                print(f"  Parsed successfully: {result}")
                if validate_json_structure(result, expected_words):
                    print(f"  Validation passed - using this result")
                    return result
                else:
                    print(f"  Validation failed")
            except Exception as e:
                print(f"  Parse failed: {e}")
    
    # Strategy 3: Parse structured text (fallback)
    try:
        result = parse_structured_text(text, expected_words)
        if result:
            print(f"Strategy 3 - Structured text parse successful: {result}")
            return result
        else:
            print(f"Strategy 3 - Structured text parse failed")
    except Exception as e:
        print(f"Strategy 3 - Exception: {e}")
    
    # Strategy 4: Parse line by line
    try:
        result = parse_line_by_line(text, expected_words)
        if result:
            print(f"Strategy 4 - Line by line parse successful: {result}")
            return result
        else:
            print(f"Strategy 4 - Line by line parse failed")
    except Exception as e:
        print(f"Strategy 4 - Exception: {e}")
    
    # Final fallback - return unknown tags
    print(f"All parsing strategies failed - using UNKNOWN tags")
    return create_default_tags(expected_words)


def fix_json_formatting(text):
    """Fix common JSON formatting issues"""
    print(f"Fixing JSON formatting for: '{text}'")
    
    fixes = [
        # Basic cleanup
        (r'^\s*```json\s*', ''),  # Remove code block markers
        (r'\s*```\s*$', ''),
        (r'^\s*```\s*', ''),
        
        # Quote fixes
        (r"'", '"'),  # Single to double quotes
        (r'(\w+):\s*([^",\[\]{}]+)(?=[,\]}])', r'"\1": "\2"'),  # Unquoted keys and values
        
        # Structural fixes
        (r'}\s*{', '},{'),  # Missing commas between objects
        (r',\s*}', '}'),    # Trailing commas in objects
        (r',\s*]', ']'),    # Trailing commas in arrays
        (r':\s*,', ': null,'),  # Empty values
        
        # Word/UPOS specific fixes
        (r'"word":\s*([^",\]]+?)(?=\s*[,}])', r'"word": "\1"'),  # Unquoted word values
        (r'"UPOS":\s*([A-Z]+)(?=\s*[,}])', r'"UPOS": "\1"'),    # Unquoted UPOS values
        
        # Array fixes
        (r'^\s*([^[]*)', ''),  # Remove text before first [
        (r'([^]]*)\s*$', ''),  # Remove text after last ]
    ]
    
    original = text
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
    
    if original != text:
        print(f"JSON formatting changed from: '{original}' to: '{text}'")
    
    return text.strip()


def validate_json_structure(result, expected_words):
    """Validate that JSON result matches expected structure"""
    print(f"Validating JSON structure:")
    print(f"  Result type: {type(result)}")
    print(f"  Result: {result}")
    print(f"  Expected words: {expected_words}")
    
    if not isinstance(result, list):
        print(f"  FAIL: Result is not a list")
        return False
    
    if len(result) != len(expected_words):
        print(f"  FAIL: Length mismatch - got {len(result)}, expected {len(expected_words)}")
        return False
    
    for i, item in enumerate(result):
        print(f"  Checking item {i}: {item}")
        if not isinstance(item, dict):
            print(f"    FAIL: Item is not a dict")
            return False
        if 'word' not in item or 'UPOS' not in item:
            print(f"    FAIL: Missing 'word' or 'UPOS' key")
            return False
        
        # Check UPOS validity
        upos = item['UPOS']
        if upos not in TAGS:
            print(f"    WARNING: UPOS '{upos}' not in valid tags {TAGS}")
            # Don't fail validation for invalid UPOS, we'll handle it later
        
        # Optional: check if word matches expected (with some flexibility for case/punctuation)
        actual_word = item['word'].lower().replace('"', '').replace("'", '')
        expected_word = expected_words[i].lower().replace('"', '').replace("'", '')
        if actual_word != expected_word:
            print(f"    WARNING: Word mismatch - got '{actual_word}', expected '{expected_word}'")
            # Allow some flexibility, don't fail validation
    
    print(f"  PASS: JSON structure is valid")
    return True


def parse_structured_text(text, expected_words):
    """Parse structured text that's not valid JSON"""
    print(f"Parsing structured text: '{text}'")
    results = []
    
    # Pattern 1: word: TAG format
    pattern1 = r'(\w+):\s*([A-Z]+)'
    matches = re.findall(pattern1, text)
    print(f"Pattern 1 (word: TAG) found {len(matches)} matches: {matches}")
    if len(matches) == len(expected_words):
        for (word, tag), expected_word in zip(matches, expected_words):
            results.append({"word": expected_word, "UPOS": tag if tag in TAGS else "UNKNOWN"})
        print(f"Pattern 1 successful: {results}")
        return results
    
    # Pattern 2: word/TAG format
    pattern2 = r'(\w+)/([A-Z]+)'
    matches = re.findall(pattern2, text)
    print(f"Pattern 2 (word/TAG) found {len(matches)} matches: {matches}")
    if len(matches) == len(expected_words):
        for (word, tag), expected_word in zip(matches, expected_words):
            results.append({"word": expected_word, "UPOS": tag if tag in TAGS else "UNKNOWN"})
        print(f"Pattern 2 successful: {results}")
        return results
    
    # Pattern 3: Extract just the tags in order
    pattern3 = r'\b([A-Z]{2,})\b'
    matches = re.findall(pattern3, text)
    print(f"Pattern 3 (uppercase words) found {len(matches)} matches: {matches}")
    valid_tags = [tag for tag in matches if tag in TAGS]
    print(f"Pattern 3 valid tags: {valid_tags}")
    if len(valid_tags) == len(expected_words):
        for word, tag in zip(expected_words, valid_tags):
            results.append({"word": word, "UPOS": tag})
        print(f"Pattern 3 successful: {results}")
        return results
    
    print(f"Structured text parsing failed")
    return None


def parse_line_by_line(text, expected_words):
    """Parse text line by line looking for word-tag pairs"""
    print(f"Parsing line by line: '{text}'")
    results = []
    lines = text.strip().split('\n')
    print(f"Found {len(lines)} lines: {lines}")
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        print(f"Processing line {line_num}: '{line}'")
        if not line:
            continue
            
        # Try different line formats
        patterns = [
            r'"word":\s*"([^"]+)",?\s*"UPOS":\s*"([^"]+)"',  # JSON-like
            r'(\w+)\s*:\s*([A-Z]+)',  # word: TAG
            r'(\w+)\s*/\s*([A-Z]+)',  # word/TAG
            r'"([^"]+)"\s*[,:]?\s*"([A-Z]+)"',  # "word", "TAG"
        ]
        
        for pattern_num, pattern in enumerate(patterns):
            match = re.search(pattern, line)
            if match:
                word, tag = match.groups()
                print(f"  Pattern {pattern_num+1} matched: word='{word}', tag='{tag}'")
                if len(results) < len(expected_words):
                    expected_word = expected_words[len(results)]
                    results.append({
                        "word": expected_word, 
                        "UPOS": tag if tag in TAGS else "UNKNOWN"
                    })
                    print(f"  Added result: {results[-1]}")
                break
        else:
            print(f"  No pattern matched for line: '{line}'")
    
    print(f"Line by line parsing result: {results}")
    return results if len(results) == len(expected_words) else None


def create_default_tags(expected_words):
    """Create default tags when all parsing fails"""
    return [{"word": word, "UPOS": "UNKNOWN"} for word in expected_words]


def get_confidence_score(raw_output, parsed_result):
    """Estimate confidence in the parsing result"""
    score = 0.0
    
    # JSON formatting quality (30%)
    if raw_output.strip().startswith('[') and raw_output.strip().endswith(']'):
        score += 0.3
    elif '[' in raw_output and ']' in raw_output:
        score += 0.15
    
    # All tags are valid (40%)
    if all(tag["UPOS"] in TAGS for tag in parsed_result):
        score += 0.4
    else:
        valid_ratio = sum(1 for tag in parsed_result if tag["UPOS"] in TAGS) / len(parsed_result)
        score += 0.4 * valid_ratio
    
    # No UNKNOWN tags (30%)
    if not any(tag["UPOS"] == "UNKNOWN" for tag in parsed_result):
        score += 0.3
    else:
        known_ratio = sum(1 for tag in parsed_result if tag["UPOS"] != "UNKNOWN") / len(parsed_result)
        score += 0.3 * known_ratio
    
    return score


def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Successfully loaded text with {len(text)} characters", flush=True)
        print(f"Sample text (first 100 chars): {text[:100]}", flush=True)
        return text
    except Exception as e:
        print(f"Error loading file {file_path}: {e}", flush=True)
        return ""


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_text(text):
    return text.split()


def split_into_chunks(tokens, chunk_size=20):
    # Smaller chunks to reduce GPU memory usage
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]


def batch_generate_text(prompts, tokenizer, model):
    batch_size = 8  
    all_generated_texts = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_i = batch_idx * batch_size
        batch_prompts = prompts[start_i:start_i+batch_size]
        print(f"Generating batch {batch_idx+1}/{total_batches} with {len(batch_prompts)} prompts...", flush=True)
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():  # Add this to save memory
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,          # Beam search with 5 beams
                #temperature=0.9,      # Sampling temperature
                #top_k=50,             # Top-k sampling
                #top_p=0.95,           # Nucleus (top-p) sampling
                #do_sample=True        # Enable sampling for top_k, top_p, temperature
            )
        
        torch.cuda.empty_cache()
        
        for j, prompt in enumerate(batch_prompts):
            gen = tokenizer.decode(output_ids[j], skip_special_tokens=True)
            raw = gen[len(prompt):] if gen.startswith(prompt) else gen
            all_generated_texts.append(raw.strip())
        
        print(f"Completed batch {batch_idx+1}/{total_batches}", flush=True)
    
    return all_generated_texts


def get_pos_tags_chunk(chunk, tokenizer, model, max_words_per_request=5):
    few_shot = """[ {"word": "bo", "UPOS": "ADJ"}, {"word": "volch", "UPOS": "VERB"}, {"word": "seyor", "UPOS": "NOUN"}, {"word": "homps", "UPOS": "NOUN"}, {"word": "sant", "UPOS": "ADJ"}, {"word": "iorn", "UPOS": "NOUN"}, {"word": "ilz", "UPOS": "PRON"}, {"word": "addicions", "UPOS": "NOUN"}, {"word": "deffendre", "UPOS": "VERB"}]
        """
    base_prompt = (
        "You are a linguistic expert in Medieval Romance languages. "
        "Analyze the given text and assign Universal Dependencies Part-of-Speech tags (UPOS) to each token. "
        f"Available tags: {', '.join(TAGS)}."
        "Return a JSON array of objects, each with only 'word' and 'UPOS' keys. " 
        "Output only the JSON array, properly formatted and closed, with no extra text or explanation. "
        #"Consider syntactic and semantic relationships, including agreement, word order, and morphology. "
        #"Medieval Romance languages often exhibit significant spelling variation; for example, Old Occitan: 'ansy', 'eynsi', or 'anes'; Old Catalan: 'fiyl', or 'conseyl'; Middle French: 'norryr' or 'norrir'. "
        #f"\nExample format:\n{few_shot}\n"
    )
    
    all_pos_tags = []
    sub_chunks = []
    prompts = []

    # Break into smaller sub-chunks
    for i in range(0, len(chunk), max_words_per_request):
        sub = chunk[i:i + max_words_per_request]
        sub_chunks.append(sub)
        token_list = ", ".join([f'"{word}"' for word in sub])
        prompt = f"{base_prompt}Now tag these {len(sub)} words: [{token_list}]\nOutput:"
        prompts.append(prompt)

    print(f"Prepared {len(prompts)} prompts for chunk of {len(chunk)} words", flush=True)

    outputs = batch_generate_text(prompts, tokenizer, model)

    # Process each sub-chunk with improved JSON parsing
    for sub_idx, (sub, raw) in enumerate(zip(sub_chunks, outputs)):
        print(f"\n{'='*50}")
        print(f"PROCESSING SUB-CHUNK {sub_idx+1}/{len(sub_chunks)}")
        print(f"Words: {sub}")
        print(f"Raw model output (full): '{raw}'")
        print(f"Raw model output length: {len(raw)} characters")
        print(f"{'='*50}")
        
        try:
            # Use the improved JSON parsing
            pos_list = robust_json_parse(raw, sub)
            confidence = get_confidence_score(raw, pos_list)
            
            print(f"\nFINAL PARSED RESULT:")
            print(f"Confidence: {confidence:.2f}")
            print(f"Parsed tags: {pos_list}")
            
            # Collect tagged words with enhanced validation
            for idx, word in enumerate(sub):
                if idx < len(pos_list):
                    upos = pos_list[idx].get("UPOS", "UNKNOWN")
                    # Enhanced UPOS validation with logging
                    if upos not in TAGS:
                        print(f"WARNING: Invalid UPOS '{upos}' for word '{word}', setting to UNKNOWN")
                        print(f"Valid tags are: {TAGS}")
                        upos = "UNKNOWN"
                    else:
                        print(f"Valid UPOS '{upos}' for word '{word}'")
                else:
                    print(f"No tag found for word '{word}', setting to UNKNOWN")
                    upos = "UNKNOWN"
                
                final_tag = {"word": word, "UPOS": upos}
                print(f"Final tag for '{word}': {final_tag}")
                all_pos_tags.append(final_tag)
                
        except Exception as e:
            print(f"ERROR parsing subchunk {sub}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Fallback to UNKNOWN tags
            for w in sub:
                fallback_tag = {"word": w, "UPOS": "UNKNOWN"}
                print(f"Fallback tag for '{w}': {fallback_tag}")
                all_pos_tags.append(fallback_tag)
    
    return all_pos_tags


def save_json(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    torch.cuda.empty_cache()
    base_dir = "prompting"
    pos_dir = os.path.join(base_dir, "PoS_Tagging_pretest")
    os.makedirs(pos_dir, exist_ok=True)
    
    model_name = "google/gemma-3-12b-it"
    auth_token = "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"
    use_cached = False

    # Initialize variables
    tokenizer = None
    model = None

    if not use_cached:
        try:
            print(f"Loading tokenizer from {model_name}...", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=auth_token,
                force_download=True
            )
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"Loading model from {model_name}...", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=auth_token,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for Gemma
                device_map="auto",
                #force_download=True
            )
            
            print("Setting model to evaluation mode...", flush=True)
            model.eval()
            print("Model and tokenizer loaded successfully!", flush=True)
            
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return
    else:
        print("Skipping model load; using cached results.", flush=True)
        return  # Exit if using cached mode

    # Check if model and tokenizer are loaded
    if model is None or tokenizer is None:
        print("Model or tokenizer failed to load. Exiting.", flush=True)
        return

    # Load text
    text = load_text(os.path.join(base_dir, "Llibre_reference.txt"))
    if not text or len(text) < 100:
        text = (
            "Ayssi comensan las prophetias del payres Nostradamus. "
            "Consideran las vanas prophetias et totalement rejectadas, "
            "las quales sus se divulguero per los indits escriches, ..."
        )

    tokens = tokenize_text(preprocess_text(text))
    print(f"Token count: {len(tokens)}", flush=True)

    chunks = split_into_chunks(tokens)
    print(f"Using chunk size={len(chunks[0]) if chunks else 0} -> {len(chunks)} chunks", flush=True)

    all_tags = []
    start = time.time()
    
    # Process only first chunk for debugging
    for i, chunk in enumerate(chunks, 1):  # Only process first chunk
        print(f"\n--- Processing chunk {i}/{len(chunks)} ---", flush=True)
        tags = get_pos_tags_chunk(chunk, tokenizer, model)
        print(f"Chunk {i} tags:", end=' ', flush=True)
        for w in tags:
            print(f"{w['word']}/{w['UPOS']}", end=' ', flush=True)
        print(flush=True)
        all_tags.extend(tags)

    print("\nComplete tagging output:", flush=True)
    for w in all_tags:
        print(f"{w['word']}/{w['UPOS']}", end=' ', flush=True)
    print(flush=True)

    save_json(all_tags, os.path.join(pos_dir, "tagging_b1_cat_zero_gemma3_5.json"))
    print("Saved tags to disk.", flush=True)

    # Enhanced statistics
    counts = {}
    confidence_stats = {"high": 0, "medium": 0, "low": 0}
    
    for w in all_tags:
        tag = w['UPOS']
        counts[tag] = counts.get(tag, 0) + 1
    
    print("\nTag counts:", flush=True)
    for t, c in sorted(counts.items()):
        print(f"{t}: {c}", flush=True)
    
    unknown_ratio = counts.get("UNKNOWN", 0) / len(all_tags)
    print(f"\nUnknown tag ratio: {unknown_ratio:.2%}", flush=True)

    duration = time.time() - start
    print(f"Total time: {duration:.2f}s", flush=True)


if __name__ == "__main__":
    main()