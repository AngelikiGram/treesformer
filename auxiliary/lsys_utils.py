import torch

def validate_lstring_structure(lstring):
    """Check if L-string follows basic structural rules."""
    issues = []
    # Check bracket balance
    depth = 0
    for i, char in enumerate(lstring):
        if char == '[': depth += 1
        elif char == ']':
            depth -= 1
            if depth < 0: issues.append(f"Negative depth at position {i}")
    if depth != 0: issues.append(f"Unbalanced brackets: {depth} unclosed")
    
    # Check if starts with A->
    if not lstring.startswith("A->"): issues.append("Does not start with A->")
    
    # Check for empty brackets []
    if "[]" in lstring: issues.append("Contains empty brackets")
    
    return len(issues) == 0, issues
def check_and_fix_model_nans(model, check_only=False):
    """Detect NaNs in model weights and replace them with zeros if not in check_only mode."""
    has_nan = False
    with torch.no_grad():
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                if not check_only:
                    print(f"[REPAIR] NaN detected in {name}, zeroing out...")
                    param.nan_to_num_(0.0)
                has_nan = True
    return has_nan
