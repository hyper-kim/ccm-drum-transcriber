with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    model.eval()'''

replacement = '''    model.eval()
    
    # CRITICAL FIX: The model was trained with Batch Size 384. 
    # Its BatchNorm running statistics expect highly varied Groove datasets.
    # Inference is Batch Size 1 on quiet Demucs stems. 
    # By forcing the first BatchNorm to recompute statistics on the test input (Instance Norm behavior),
    # we completely bypass the domain shift caused by absolute volume/variance differences.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train() # Force BN to use batch statistics instead of running stats!
'''

text = text.replace(target, replacement)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Forced BatchNorm to use batch stats")
