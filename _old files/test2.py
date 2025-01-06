from transformers import AutoTokenizer

def test_transformers():
    try:
        # Try to load a simple tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("Transformers package is working!")
        
        # Test basic tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer(test_text)
        print("\nTest tokenization:")
        print(f"Input text: {test_text}")
        print(f"Tokens: {tokens}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_transformers()